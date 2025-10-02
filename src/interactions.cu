#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ glm::vec3 randomInUnitSphere(
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    while (true)
    {
        float x = u01(rng) * 2.0f - 1.0f;
        float y = u01(rng) * 2.0f - 1.0f;
        float z = u01(rng) * 2.0f - 1.0f;
        glm::vec3 p(x, y, z);
        if (glm::dot(p, p) < 1.0f)
            return p;
    }
}

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    const float bias = EPSILON;
    glm::vec3 wi = glm::normalize(pathSegment.ray.direction);

    // Refractive (glass) material
    if (m.hasRefractive > 0.0f && m.indexOfRefraction > 0.0f)
    {
        // Orientation and indices
        glm::vec3 inDir = wi;
        bool inside = glm::dot(inDir, normal) > 0.0f;
        float etaI = inside ? m.indexOfRefraction : 1.0f;
        float etaT = inside ? 1.0f : m.indexOfRefraction;
        float eta = etaI / etaT;
        glm::vec3 N = inside ? -normal : normal;

        // Directions
        float cosThetaI = glm::clamp(-glm::dot(inDir, N), 0.f, 1.f);
        glm::vec3 reflectDir = glm::reflect(inDir, N);
        glm::vec3 refractDir = glm::refract(inDir, N, eta);

        // Fresnel via Schlick
        float r0 = (etaT - etaI) / (etaT + etaI);
        r0 = r0 * r0;
        float fresnel = r0 + (1.0f - r0) * powf(1.0f - cosThetaI, 5.0f);

        thrust::uniform_real_distribution<float> u01(0, 1);
        bool chooseReflect = (u01(rng) < fresnel) || (glm::dot(refractDir, refractDir) < 1e-12f);

        glm::vec3 newDir = chooseReflect ? reflectDir : refractDir;
        pathSegment.ray.direction = glm::normalize(newDir);
        pathSegment.ray.origin = intersect + (chooseReflect ? N : -N);

        pathSegment.color *= m.color;
        pathSegment.remainingBounces -= 1;
        return;
    }

    // Subsurface scattering
    if (m.hasSubsurface > 0.0f)
    {
        glm::vec3 n = glm::normalize(normal);
        glm::vec3 directionNotNormal = (fabsf(n.x) < SQRT_OF_ONE_THIRD) ? glm::vec3(1, 0, 0)
            : (fabsf(n.y) < SQRT_OF_ONE_THIRD) ? glm::vec3(0, 1, 0) : glm::vec3(0, 0, 1);
        glm::vec3 t1 = glm::normalize(glm::cross(n, directionNotNormal));
        glm::vec3 t2 = glm::normalize(glm::cross(n, t1));

        // mfp = 1 / sigma_t
        glm::vec3 sigmaT = m.sigmaA + m.sigmaS;
        float sigmaT_scalar = fmaxf(1e-6f, (sigmaT.x + sigmaT.y + sigmaT.z) / 3.0f);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float u1 = u01(rng);
        float u2 = u01(rng);
        float rd = 1.0f / sigmaT_scalar;
        float r = -rd * logf(fmaxf(1e-6f, 1.0f - u1));
        float phi = TWO_PI * u2;
        glm::vec3 lateral = r * (cosf(phi) * t1 + sinf(phi) * t2);
        glm::vec3 newPoint = intersect + lateral;

        // Cosine-weighted outgoing direction
        glm::vec3 newDir = glm::normalize(calculateRandomDirectionInHemisphere(n, rng));

        glm::vec3 albedo = glm::vec3(
            sigmaT.x > 0 ? m.sigmaS.x / sigmaT.x : 0.0f,
            sigmaT.y > 0 ? m.sigmaS.y / sigmaT.y : 0.0f,
            sigmaT.z > 0 ? m.sigmaS.z / sigmaT.z : 0.0f);
        glm::vec3 att = glm::exp(-m.sigmaA * r);
        pathSegment.color *= (albedo * att * m.color);

        pathSegment.ray.origin = newPoint + n * bias;
        pathSegment.ray.direction = newDir;
        pathSegment.remainingBounces -= 1;
        return;
    }

    // Reflective / glossy material
    if (m.hasReflective > 0.0f)
    {
        glm::vec3 R = glm::reflect(wi, normal);
        float rough = m.specular.exponent; // interpret exponent as roughness [0..1]
        if (rough > 0.0f)
        {
            glm::vec3 fuzz = rough * randomInUnitSphere(rng);
            R = glm::normalize(R + fuzz);
            if (glm::dot(R, normal) < 0.0f)
            {
                R = glm::reflect(R, normal);
            }
        }
        pathSegment.color *= m.color;
        pathSegment.ray.origin = intersect + normal * bias;
        pathSegment.ray.direction = glm::normalize(R);
        pathSegment.remainingBounces -= 1;
        return;
    }

    // Diffuse scatter
    glm::vec3 newDir = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
    pathSegment.color *= m.color;
    pathSegment.ray.origin = intersect + normal * bias;
    pathSegment.ray.direction = newDir;
    pathSegment.remainingBounces -= 1;
}
