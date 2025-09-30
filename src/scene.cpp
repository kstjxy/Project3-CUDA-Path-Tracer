#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cfloat>

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
        return;
    }
    std::vector<glm::vec3> verts;
    outMin = glm::vec3(FLT_MAX);
    outMax = glm::vec3(-FLT_MAX);

    std::string line;
    while (std::getline(in, line))
    {
        if (line.size() < 2) continue;
        if (line[0] == 'v' && (line[1] == ' '))
        {
            std::istringstream ss(line.substr(2));
            float x, y, z; ss >> x >> y >> z;
            glm::vec3 v = glm::vec3(x, y, z);
            verts.push_back(v);
        }
    }
    // Rewind to parse faces
    in.clear();
    in.seekg(0, std::ios::beg);
    while (std::getline(in, line))
    {
        if (line.size() < 2) continue;
        if (line[0] == 'f' && (line[1] == ' '))
        {
            std::istringstream ss(line.substr(2));
            std::vector<int> vidx;
            std::string tok;
            while (ss >> tok)
            {
                // token formats: v, v/t, v//n, v/t/n
                size_t slash = tok.find('/');
                int vi = 0;
                if (slash == std::string::npos) {
                    vi = std::stoi(tok);
                } else {
                    vi = std::stoi(tok.substr(0, slash));
                }
                // OBJ indices are 1-based
                vidx.push_back(vi - 1);
            }
            if (vidx.size() < 3) continue;
            // Triangulate fan
            for (size_t i = 1; i + 1 < vidx.size(); ++i)
            {
                glm::vec3 v0 = verts[vidx[0]];
                glm::vec3 v1 = verts[vidx[i]];
                glm::vec3 v2 = verts[vidx[i + 1]];
                // Transform to world
                glm::vec3 w0 = glm::vec3(transform * glm::vec4(v0, 1.0f));
                glm::vec3 w1 = glm::vec3(transform * glm::vec4(v1, 1.0f));
                glm::vec3 w2 = glm::vec3(transform * glm::vec4(v2, 1.0f));
                Triangle t{ w0, w1, w2, materialId };
                outTris.push_back(t);
                outMin = glm::min(outMin, glm::min(w0, glm::min(w1, w2)));
                outMax = glm::max(outMax, glm::max(w0, glm::max(w1, w2)));
            }
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
            std::string objPath = p.value("FILE", std::string());
            if (objPath.empty())
            {
                std::cerr << "Mesh object missing FILE path. Skipping.\n";
            }
            else
            {
                std::filesystem::path full = objPath;
                if (full.is_relative()) { full = baseDir / full; }
                newGeom.triStart = static_cast<int>(triangles.size());
                glm::vec3 bbMin, bbMax;
                loadOBJMesh(full.string(), newGeom.transform, newGeom.materialid, triangles, bbMin, bbMax);
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
