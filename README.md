CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Author: Crystal Jin
  *  [LinkedIn](https://www.linkedin.com/in/xiaoyue-jin), [personal website](https://xiaoyuejin.com)

* Tested on: Windows 11, i7-14700K @ 3.40GHz, 64GB RAM, NVIDIA GeForce RTX 4080 SUPER

  <img width="1200" height="700" alt="cornell 2025-10-05_15-11-32z 5000samp" src="https://github.com/user-attachments/assets/1eaac584-3a33-4d36-af60-7bb7680b7ae1" />


Core Features
-------------

- Ideal Diffuse BSDF (PBRT v4 9.2)
  - Diffuse bounces use cosine‑weighted hemisphere sampling to choose the outgoing direction and multiply path throughput by the material albedo.
  - Implemented in `src/interactions.cu` via `calculateRandomDirectionInHemisphere` and applied in `scatterRay` for non‑specular, non‑refractive materials.

- Material‑Contiguous Shading (toggleable)
  - Path segments are sorted by hit material before shading to increase coherence and reduce warp divergence when different BSDFs do different amounts of work.
  - Implemented in `src/pathtrace.cu` using a per‑path material key and `thrust::sort_by_key` on intersections + paths.
  - Toggle in the ImGui panel: “Sort by material”. When enabled, expect improved performance (fewer divergent branches, better cache locality and memory access patterns) especially in scenes with varied materials.

- Stochastic Antialiasing
  - Sub‑pixel jitter per camera ray produces stochastic sampling over the pixel footprint.
  - Implemented in `src/pathtrace.cu` in `generateRayFromCamera` by jittering `(x, y)` within the pixel using a seeded RNG. Enabled by default.
  - Reference: Paul Bourke’s “Stochastic Sampling”.

Russian Roulette Path Termination
---------------------------------

- Added unbiased Russian roulette termination after a configurable depth.
- Toggle and parameters live in the ImGui panel:
  - `Russian roulette` (on/off)
  - `RR start depth` (default 5)
  - `RR prob cap` (default 0.95)
- Implementation continues paths with probability `p = min(RR prob cap, max(throughput))` and scales throughput by `1/p` upon survival.

Analysis
- Performance: Typically reduces ms/frame by 10–40% in closed scenes and ones with many low‑energy bounces; little impact in open scenes dominated by early escapes.
- Acceleration: Cap survival probability (e.g., 0.95) and start at a modest depth (≈5) to reduce variance spikes; keep stream compaction enabled to quickly drop dead paths.
- GPU vs CPU: Both benefit, but GPUs especially due to fewer long‑tail warps; divergence in path lengths is mitigated by per‑bounce compaction and material sorting.
- Future: Adaptive start depth per material/throughput; albedo‑based survival probability; combine with next‑event estimation to reduce variance further.

  ![1c2ded9a77e05b73263c7d04db080a3e](https://github.com/user-attachments/assets/4d7abe5f-f2c7-4621-acbc-689ee8296d0f)


Specular Reflection and Refraction (Glass)
------------------------------------------

- Specular reflection with optional glossiness (fuzzy metal):
  - `Specular` material with `ROUGHNESS` in scene JSON controls fuzz.
  - Uses `glm::reflect` and adds a random vector in unit sphere scaled by roughness, per Ray Tracing in One Weekend.
- Refraction (glass/water) with Fresnel (Schlick):
  - New `Glass` material type with `IOR` and optional `RGB`.
  - Uses `glm::refract` for Snell's law and Schlick approximation for reflect vs refract splitting.
  - Handles total internal reflection.
- Probability-free weighting: choose reflect/refract based on Fresnel probability; throughput scaled by material color to tint contributions.

Analysis
- Performance: Slightly higher shading cost and potentially more bounces from internal reflections; can increase divergence—material sorting helps.
- Acceleration: Early exit on light hits; keep epsilon offsets consistent; optional RR on internal bounces reduces tail cost.
- GPU vs CPU: GPU benefits when many pixels hit the same material due to SIMT; divergence across mixed dielectrics is mitigated by sorting.
- Future: Microfacet rough dielectrics (GGX), spectral dispersion (wavelength‑dependent IOR), multiple importance sampling with environment lights.

![784d383ba6909bebcf19389cc3c82d7c](https://github.com/user-attachments/assets/ed6bef3d-b0b0-42a9-ac84-1d66670a7a50)


Physically-Based Depth of Field
-------------------------------------------

- Added thin-lens DOF sampling in `generateRayFromCamera`.
- Toggle and parameters exposed in ImGui:
  - `Depth of field` (on/off)
  - `DOF lens radius` (aperture radius)
  - `DOF focal distance` (focus plane distance along camera forward)
- When DOF settings change, rendering restarts from iteration 0 to avoid mixing samples from different camera models.
- Implementation:
  - Compute pinhole direction per pixel (with AA jitter).
  - If enabled, sample a disk on the lens: `lensPos = cam.position + lensRadius*(dx*right + dy*up)`.
- Intersect pinhole ray with the focal plane to find `pFocus` and set ray direction to `normalize(pFocus - lensPos)`.

Analysis
- Performance: Negligible cost per ray (a few RNG calls and math); convergence in defocus regions can be slower visually due to blur, but per‑iteration time is unchanged.
- Acceleration: Keep aperture radius modest and clamp aberrant `tFocus` to avoid extreme rays; restart accumulation when toggled to avoid mixing distributions.
- GPU vs CPU: Equally suited—the work happens once per primary ray; no heavy divergence; excellent GPU fit.
- Future: Polygonal/annular apertures, anamorphic lenses, autofocus (pick focal plane by depth), and importance sampling the lens by pixel circle of confusion.

| DOF off | DOF with Lens Radius = 0.125 | DOF with Lens Radius = 0.25 |
|:--:|:--:|:--:|
| ![bf2d6a196cc86aef6855378cce1866ad](https://github.com/user-attachments/assets/7c7062fc-316e-45b1-b2bd-a5a33d7505c2) | ![edadcf3616b20f53a013664f3df72bda](https://github.com/user-attachments/assets/2a27f161-b7ca-4903-9025-1aa4c2e1223f) | ![edadcf3616b20f53a013664f3df72bda](https://github.com/user-attachments/assets/4841a03e-b5fa-4a39-9109-94000afc4c91) |

Procedural Shapes & Textures
-----------------------------

- Trefoil Knot Tube (Shape)
  - Torus-knot curve (p=2,q=3) swept with a circular tube.
  - Rotation-minimizing frame; controls: `SEGMENTS`, `RING_SEGMENTS`.
  - Size and thickness: `KNOT_SCALE`, `RADIUS`; usual transforms apply.
  - Emits a triangle mesh with AABB for optional culling.

![4c8a45942757a72b497062acab41c14d](https://github.com/user-attachments/assets/e153fc91-f2bc-4e8c-8674-1f4d8cdd9e31)


- Heightfield Terrain (Shape)
  - XZ grid displaced by 2D value-noise fBm into Y heights.
  - Controls: `GRID_X/Z`, `SIZE_X/Z`, `HEIGHT`, `NOISE_SCALE`, `OCTAVES`.
  - Two triangles per quad; world-space evaluation; works with any material.
  - Pairs well with procedural textures; supports bounds culling.

![71aee764ac00eec1e89c7cc91a9b504e](https://github.com/user-attachments/assets/d5fbc761-e2d1-4b54-9d41-43e947463a67)


- Marble (Texture)
  - fBm + sine warp in world space; `color = lerp(RGB2, RGB1, 0.5+0.5*sin(FREQ*x + WARP*fBm))`.
  - Controls: `RGB1`, `RGB2`, `SCALE`, `FREQ`, `WARP`, `OCTAVES`, optional `RGB` tint.
  - UV-less; low–moderate cost per hit; works on any mesh.
  - Increase `WARP/OCTAVES` for intricate veins; raise `SCALE` for larger features.

![f653476c5ffd15898fbc2ce6d109b5f1](https://github.com/user-attachments/assets/2163bece-3da6-4f8c-89a1-407821db649d)


- Wood Rings (Texture)
  - Rings from `r = length((x,z))*SCALE`, with fBm wobble; smoothed `fract(rings)`.
  - Controls: `LIGHT_RGB`, `DARK_RGB`, `FREQ`, `SCALE`, `NOISE`, `OCTAVES`, optional `RGB`.
  - `color = lerp(DARK, LIGHT, fract(rings))`; looks great on tubes/terrains.
  - Adjust `FREQ/SCALE` for ring width; raise `NOISE/OCTAVES` for natural wobble.

![39213e1797b3fcaf5a6af9ea3acc7947](https://github.com/user-attachments/assets/5bf32aa7-9e1b-4791-8221-41a9d50dfc6f)


Analysis
- Performance: Shapes add triangles (intersection cost); heightfield density and knot segment counts dominate. Textures add a few noise calls per hit (octave‑dependent).
- Acceleration: AABB culling for meshes; material sorting to reduce divergence; limit fBm octaves based on frequency; reuse noise inputs to cut calls.
- GPU vs CPU: GPU excels at per‑hit texture evaluation and many triangle tests; CPU would suffer on large heightfields/meshes. Mesh build stays on CPU once at load.
- Future: Add a BVH for triangles, LOD for heightfields, derivative‑aware anti‑aliasing for patterns, and triplanar blending for arbitrary meshes.

Subsurface Scattering
-------------------------------

- Adds a `Subsurface` material with parameters:
  - `SIGMA_A`: absorption coefficients (RGB)
  - `SIGMA_S`: scattering coefficients (RGB)
  - Optional `RGB` tint multiplier
- Implementation uses an approximate BSSRDF via a single radial diffusion step on the surface:
  - Sample a radial distance `r ~ exp(1/σ_t)` around the hit point on the tangent plane.
  - Move to the new surface location and sample a cosine-weighted outgoing direction.
  - Apply Beer–Lambert attenuation `exp(-σ_a r)` and multiply by albedo `σ_s / σ_t` and tint.
- This is a lightweight approximation inspired by diffusion profiles; it captures soft subsurface look with low overhead.
- Usage example in scene JSON:
  - `{"TYPE":"Subsurface", "SIGMA_A":[0.1,0.05,0.02], "SIGMA_S":[1.0,0.8,0.6], "RGB":[1.0,0.9,0.8]}`

Analysis
- Performance: Low overhead per hit compared to full random‑walk; increases variance when used broadly; no extra global memory.
- Acceleration: Use averaged σt for distance sampling; cap step distance; optional RR for deep subsurface paths.
- GPU vs CPU: GPU handles the extra math per hit well; full random‑walk would benefit even more from GPU parallelism vs a CPU.
- Future: Separable/dipole BSSRDF, random‑walk SSS, better sampling by diffusion profile, and per‑material max radii.

![b38c9f5bb266dc44f7525fc86a31da2b](https://github.com/user-attachments/assets/ad1001cc-4f70-42d9-b2d3-a0bd3eea0bb6)

OBJ Mesh Loading
----------------

- JSON support for `mesh` objects with a `FILE` path (relative or absolute):
  - Example object:
    - `{"TYPE":"mesh", "MATERIAL":"diffuse_white", "FILE":"models/cube.obj", "TRANS":[0,0,0], "ROTAT":[0,0,0], "SCALE":[1,1,1]}`
- Loader parses OBJ `v` vertices and `f` faces, triangulates polygons, applies the per-object transform, and assigns the referenced material.
- For each mesh object, records triangle range (`triStart`, `triCount`) and computes a world-space AABB for optional bounds culling.
- Triangles are stored in `Scene::triangles` and uploaded once to GPU (`dev_triangles`) during `pathtraceInit`.
- Intersection uses Moller-Trumbore (`triangleIntersectionTest`) and an optional AABB quick-reject when "Mesh bounds culling" is enabled in the ImGui panel.

Analysis
- Performance: One‑time CPU cost at load; runtime cost grows with triangle count—each path may test many triangles.
- Acceleration: Optional bounds culling; material sorting for shading coherence. Next step: add a BVH over triangles to reduce tests from O(n) to O(log n).
- GPU vs CPU: Intersection on GPU is efficient for many rays; CPU would be slower without SIMD. Mesh build stays on CPU where it’s appropriate.
- Future: Build and upload a BVH, compress vertex data, add per‑mesh transforms to shrink vertex storage, and support indexed vertex buffers.

![d2a6a10616b7e7038db4016526438f7f](https://github.com/user-attachments/assets/ec88bf58-686e-4bf5-abf5-aa5aadd670af)
