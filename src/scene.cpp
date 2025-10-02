#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/norm.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <sstream>
#include <filesystem>

using namespace std;
using json = nlohmann::json;

static void loadOBJMesh(const std::string& filename,
                        const glm::mat4& transform,
                        int materialId,
                        std::vector<Triangle>& outTris,
                        glm::vec3& outMin,
                        glm::vec3& outMax)
{
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Failed to open OBJ file: " << filename << std::endl;
        outMin = glm::vec3(0.0f);
        outMax = glm::vec3(0.0f);
        return;
    }
    std::vector<glm::vec3> verts;
    outMin = glm::vec3(FLT_MAX);
    outMax = glm::vec3(-FLT_MAX);

    std::string line;
    while (std::getline(in, line))
    {
        if (line.size() < 2) continue;
        if (line[0] == 'v' && line[1] == ' ')
        {
            std::istringstream ss(line.substr(2));
            float x, y, z; ss >> x >> y >> z;
            verts.emplace_back(x, y, z);
        }
    }
    in.clear();
    in.seekg(0, std::ios::beg);
    while (std::getline(in, line))
    {
        if (line.size() < 2) continue;
        if (line[0] == 'f' && line[1] == ' ')
        {
            std::istringstream ss(line.substr(2));
            std::vector<int> vidx;
            std::string tok;
            while (ss >> tok)
            {
                size_t slash = tok.find('/');
                int vi = 0;
                if (slash == std::string::npos) vi = std::stoi(tok);
                else vi = std::stoi(tok.substr(0, slash));
                vidx.push_back(vi - 1); // OBJ is 1-based
            }
            if (vidx.size() < 3) continue;
            for (size_t i = 1; i + 1 < vidx.size(); ++i)
            {
                glm::vec3 v0 = verts[vidx[0]];
                glm::vec3 v1 = verts[vidx[i]];
                glm::vec3 v2 = verts[vidx[i + 1]];
                glm::vec3 w0 = glm::vec3(transform * glm::vec4(v0, 1.0f));
                glm::vec3 w1 = glm::vec3(transform * glm::vec4(v1, 1.0f));
                glm::vec3 w2 = glm::vec3(transform * glm::vec4(v2, 1.0f));
                outTris.push_back(Triangle{ w0, w1, w2, materialId });
                outMin = glm::min(outMin, glm::min(w0, glm::min(w1, w2)));
                outMax = glm::max(outMax, glm::max(w0, glm::max(w1, w2)));
            }
        }
    }
}

// Generate a trefoil knot tube by sweeping a circular cross section along a trefoil curve
static void generateTrefoilTube(const glm::mat4& transform,
                                int materialId,
                                int segments,
                                int ringSegments,
                                float knotScale,
                                float tubeRadius,
                                std::vector<Triangle>& outTris,
                                glm::vec3& outMin,
                                glm::vec3& outMax)
{
    segments = std::max(segments, 3);
    ringSegments = std::max(ringSegments, 3);
    const float twoPi = TWO_PI;
    const float dt = twoPi / segments;

    std::vector<glm::vec3> centers(segments);
    std::vector<glm::vec3> tangents(segments);
    std::vector<glm::vec3> normals(segments);
    std::vector<glm::vec3> binormals(segments);

    auto trefoil = [knotScale](float t) {
        float c3 = cosf(3.0f * t);
        float s3 = sinf(3.0f * t);
        float c2 = cosf(2.0f * t);
        float s2 = sinf(2.0f * t);
        glm::vec3 p;
        p.x = (2.0f + c3) * c2;
        p.y = (2.0f + c3) * s2;
        p.z = s3;
        return knotScale * p;
    };

    // Centers and tangents
    for (int i = 0; i < segments; ++i)
    {
        float t = i * dt;
        int ip = (i + 1) % segments;
        int im = (i - 1 + segments) % segments;
        centers[i] = trefoil(t);
        glm::vec3 cNext = trefoil((i + 1) * dt);
        glm::vec3 cPrev = trefoil((i - 1 + segments) * dt);
        tangents[i] = glm::normalize(cNext - cPrev);
    }

    // Initialize frame with a reasonable up
    glm::vec3 up = glm::vec3(0, 1, 0);
    if (fabsf(glm::dot(up, tangents[0])) > 0.9f) up = glm::vec3(1, 0, 0);
    normals[0] = glm::normalize(glm::cross(up, tangents[0]));
    binormals[0] = glm::normalize(glm::cross(tangents[0], normals[0]));

    // Parallel transport frame
    for (int i = 1; i < segments; ++i)
    {
        glm::vec3 Ti = tangents[i];
        glm::vec3 Ni_1 = normals[i - 1];
        glm::vec3 Ni = Ni_1 - Ti * glm::dot(Ti, Ni_1);
        if (glm::length2(Ni) < 1e-8f)
        {
            // fallback
            glm::vec3 v = fabsf(Ti.y) < 0.9f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
            Ni = glm::normalize(glm::cross(v, Ti));
        }
        else
        {
            Ni = glm::normalize(Ni);
        }
        glm::vec3 Bi = glm::normalize(glm::cross(Ti, Ni));
        normals[i] = Ni;
        binormals[i] = Bi;
    }

    // Precompute ring directions in local frame
    std::vector<glm::vec2> ringDirs(ringSegments);
    for (int j = 0; j < ringSegments; ++j)
    {
        float a = (j * twoPi) / ringSegments;
        ringDirs[j] = glm::vec2(cosf(a), sinf(a));
    }

    // Generate vertices per ring
    std::vector<std::vector<glm::vec3>> verts(segments, std::vector<glm::vec3>(ringSegments));
    outMin = glm::vec3(FLT_MAX);
    outMax = glm::vec3(-FLT_MAX);
    for (int i = 0; i < segments; ++i)
    {
        const glm::vec3& C = centers[i];
        const glm::vec3& N = normals[i];
        const glm::vec3& B = binormals[i];
        for (int j = 0; j < ringSegments; ++j)
        {
            glm::vec3 p = C + tubeRadius * (ringDirs[j].x * N + ringDirs[j].y * B);
            glm::vec3 pw = glm::vec3(transform * glm::vec4(p, 1.0f));
            verts[i][j] = pw;
            outMin = glm::min(outMin, pw);
            outMax = glm::max(outMax, pw);
        }
    }

    // Emit triangles
    outTris.reserve(outTris.size() + segments * ringSegments * 2);
    for (int i = 0; i < segments; ++i)
    {
        int i1 = (i + 1) % segments;
        for (int j = 0; j < ringSegments; ++j)
        {
            int j1 = (j + 1) % ringSegments;
            const glm::vec3& v00 = verts[i][j];
            const glm::vec3& v01 = verts[i][j1];
            const glm::vec3& v10 = verts[i1][j];
            const glm::vec3& v11 = verts[i1][j1];
            outTris.push_back(Triangle{ v00, v10, v11, materialId });
            outTris.push_back(Triangle{ v00, v11, v01, materialId });
        }
    }
}

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const std::filesystem::path baseDir = std::filesystem::path(jsonName).parent_path();
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = newMaterial.color;
            const float roughness = p.value("ROUGHNESS", 0.0f);
            // Treat as reflective with roughness controlling glossiness
            newMaterial.hasReflective = 1.0f;
            newMaterial.specular.exponent = roughness; // reuse exponent as roughness [0..1]
        }
        else if (p["TYPE"] == "Glass")
        {
            const auto& col = p.value("RGB", std::vector<float>{1.0f, 1.0f, 1.0f});
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p.value("IOR", 1.5f);
        }
        else if (p["TYPE"] == "Subsurface")
        {
            // Simple SSS: specify absorption (sigma_a) and scattering (sigma_s)
            auto a = p.value("SIGMA_A", std::vector<float>{0.1f, 0.1f, 0.1f});
            auto s = p.value("SIGMA_S", std::vector<float>{1.0f, 1.0f, 1.0f});
            newMaterial.sigmaA = glm::vec3(a[0], a[1], a[2]);
            newMaterial.sigmaS = glm::vec3(s[0], s[1], s[2]);
            newMaterial.hasSubsurface = 1.0f;
            // Optional tint color multiplies final throughput
            const auto& col = p.value("RGB", std::vector<float>{1.0f, 1.0f, 1.0f});
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Marble")
        {
            // Procedural marble texture parameters
            newMaterial.hasMarble = 1.0f;
            auto c1 = p.value("RGB1", std::vector<float>{1.0f, 1.0f, 1.0f});
            auto c2 = p.value("RGB2", std::vector<float>{0.8f, 0.8f, 0.8f});
            newMaterial.marbleColor1 = glm::vec3(c1[0], c1[1], c1[2]);
            newMaterial.marbleColor2 = glm::vec3(c2[0], c2[1], c2[2]);
            newMaterial.marbleScale = p.value("SCALE", 1.0f);
            newMaterial.marbleFrequency = p.value("FREQ", 6.0f);
            newMaterial.marbleWarp = p.value("WARP", 2.0f);
            newMaterial.marbleOctaves = p.value("OCTAVES", 5);
            // Base color acts as additional tint
            const auto& col = p.value("RGB", std::vector<float>{1.0f, 1.0f, 1.0f});
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "mesh")
        {
            newGeom.type = MESH;
        }
        else if (type == "trefoil_knot")
        {
            // Represent as a mesh of triangles
            newGeom.type = MESH;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
        if (newGeom.type == MESH)
        {
            if (type == "mesh")
            {
                std::string fileRel = p.value("FILE", std::string());
                if (fileRel.empty()) {
                    std::cerr << "Mesh object missing FILE path. Skipping.\n";
                } else {
                    std::filesystem::path full = fileRel;
                    if (full.is_relative()) full = baseDir / full;
                    newGeom.triStart = static_cast<int>(triangles.size());
                    glm::vec3 bbMin, bbMax;
                    loadOBJMesh(full.string(), newGeom.transform, newGeom.materialid, triangles, bbMin, bbMax);
                    newGeom.triCount = static_cast<int>(triangles.size()) - newGeom.triStart;
                    newGeom.bboxMin = bbMin;
                    newGeom.bboxMax = bbMax;
                }
            }
            else if (type == "trefoil_knot")
            {
                int segments = p.value("SEGMENTS", 256);
                int ringSegments = p.value("RING_SEGMENTS", 16);
                float knotScale = p.value("KNOT_SCALE", 2.0f);
                float tubeRadius = p.value("RADIUS", 0.15f);
                newGeom.triStart = static_cast<int>(triangles.size());
                glm::vec3 bbMin, bbMax;
                generateTrefoilTube(newGeom.transform, newGeom.materialid,
                                    segments, ringSegments, knotScale, tubeRadius,
                                    triangles, bbMin, bbMax);
                newGeom.triCount = static_cast<int>(triangles.size()) - newGeom.triStart;
                newGeom.bboxMin = bbMin;
                newGeom.bboxMax = bbMax;
            }
        }
        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);
    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
