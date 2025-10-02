#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    // Mesh-only metadata
    int triStart = -1;
    int triCount = 0;
    glm::vec3 bboxMin = glm::vec3(0.0f);
    glm::vec3 bboxMax = glm::vec3(0.0f);
};

struct Triangle
{
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
    int materialId;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
    // Subsurface scattering parameters (approximate)
    float hasSubsurface = 0.0f;
    glm::vec3 sigmaA = glm::vec3(0.0f); // absorption
    glm::vec3 sigmaS = glm::vec3(0.0f); // scattering
    // Procedural Marble texture
    float hasMarble = 0.0f;
    float marbleScale = 1.0f;
    float marbleFrequency = 5.0f;
    float marbleWarp = 1.0f;
    int   marbleOctaves = 5;
    glm::vec3 marbleColor1 = glm::vec3(1.0f); // vein color
    glm::vec3 marbleColor2 = glm::vec3(0.8f); // base color
    // Procedural Wood rings texture
    float hasWood = 0.0f;
    glm::vec3 woodLightColor = glm::vec3(0.8f, 0.7f, 0.5f);
    glm::vec3 woodDarkColor  = glm::vec3(0.4f, 0.2f, 0.1f);
    float woodScale = 1.0f;
    float woodFrequency = 8.0f;
    float woodNoiseAmp = 0.5f;
    int   woodOctaves = 4;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};
