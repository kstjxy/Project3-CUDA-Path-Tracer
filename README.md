CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

Russian Roulette Path Termination
---------------------------------

- Added unbiased Russian roulette termination after a configurable depth.
- Toggle and parameters live in the ImGui panel:
  - `Russian roulette` (on/off)
  - `RR start depth` (default 5)
  - `RR prob cap` (default 0.95)
- Implementation continues paths with probability `p = min(RR prob cap, max(throughput))` and scales throughput by `1/p` upon survival.

How To Evaluate Performance
---------------------------

- Scenes: use closed scenes like `scenes/cornell.json` and `scenes/cornell_mesh.json`.
- Run the renderer with a fixed iteration count (e.g., 500–2000).
- Record the average frame time from the ImGui overlay (ms/frame) or compute total time to finish iterations.
- Compare with `Russian roulette` enabled vs disabled, keeping all other settings identical.
- Expected: noticeable speedup in closed scenes due to early termination of low-throughput paths, with negligible bias (estimator remains unbiased by survival scaling).

Physically-Based Depth of Field (Thin Lens)
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

Marble Texture (Procedural)
---------------------------

- Material type: `Marble` with parameters:
  - `RGB1`: vein color (e.g., `[1.0, 1.0, 1.0]`)
  - `RGB2`: base color (e.g., `[0.8, 0.8, 0.8]`)
  - `SCALE`: spatial scale of the pattern (default 1.0–2.0)
  - `FREQ`: base band frequency (default ~6.0)
  - `WARP`: sine warp amplitude from fBm (default ~2.0)
  - `OCTAVES`: fBm octaves (default 5)
  - Optional `RGB`: overall tint multiplier
- Implementation: 3D value-noise fBm + sine warp evaluated in world space at the hit point:
  - `wobble = WARP * fBm(p * SCALE, OCTAVES)`
  - `t = FREQ * p.x + wobble * 2π`
  - `mix = 0.5 + 0.5 sin(t)` → `color = lerp(RGB2, RGB1, mix) * RGB`
- Usage example in scene JSON:
  - `{"TYPE":"Marble","RGB1":[1,1,1],"RGB2":[0.8,0.8,0.8],"SCALE":1.2,"FREQ":6,"WARP":2,"OCTAVES":5}`
- Notes:
  - World-space evaluation avoids UVs and works on any mesh (spheres, cubes, procedural shapes).
  - Increase `WARP` or `OCTAVES` for more intricate veins; increase `SCALE` for larger features.
  - Performance impact is low to moderate; each diffuse hit evaluates a handful of noise calls per octave.

Wood Rings (Procedural)
-----------------------

- Material type: `Wood` with parameters:
  - `LIGHT_RGB`: lighter ring color (e.g., `[0.85, 0.7, 0.5]`)
  - `DARK_RGB`: darker ring color (e.g., `[0.4, 0.2, 0.1]`)
  - `SCALE`: spatial scale (default 1.0–1.5)
  - `FREQ`: ring frequency (default ~8–12)
  - `NOISE`: wobble amplitude added via fBm (default ~0.4–0.6)
  - `OCTAVES`: fBm octaves (default 4)
  - Optional `RGB`: overall tint multiplier
- Implementation: rings around the Y axis using world-space coordinates at the hit point:
  - `r = length((x, z)) * SCALE`
  - `rings = r * FREQ + NOISE * fBm(p, OCTAVES)`
  - `t = fract(rings)` then smoothed; `color = lerp(DARK_RGB, LIGHT_RGB, t) * RGB`
- Usage example in scene JSON:
  - `{"TYPE":"Wood","LIGHT_RGB":[0.85,0.7,0.5],"DARK_RGB":[0.4,0.2,0.1],"SCALE":1.2,"FREQ":10,"NOISE":0.4,"OCTAVES":4}`
- Notes:
  - Looks great on curved tubes and terrains; adjust `FREQ`/`SCALE` to control ring width.
  - Higher `NOISE` and `OCTAVES` add more natural wobble; lower values give clean rings.
  - Wood currently overrides Marble if both are enabled on the same material.

Trefoil Knot Tube (Procedural)
------------------------------

- Object type: `trefoil_knot` generated at load time, represented as a triangle mesh.
- Parameters (JSON):
  - `SEGMENTS` (default 256): samples along the knot curve
  - `RING_SEGMENTS` (default 16-24): samples around the tube cross section
  - `KNOT_SCALE` (default 2.0): overall knot size (world units)
  - `RADIUS` (default 0.15): tube radius (world units)
- Transform fields (`TRANS`, `ROTAT`, `SCALE`) are applied as usual.
- Implementation outline:
  - Use a torus-knot parametric curve; trefoil is the (p=2, q=3) case.
  - Construct a rotation-minimizing (parallel transport) frame along the curve to sweep a circle with minimal twist.
  - Emit triangle strips between adjacent rings; compute a world-space AABB for optional bounds culling.
- References:
  - Torus Knot — general (p, q) formulation (trefoil is p=2, q=3): https://en.wikipedia.org/wiki/Torus_knot
  - PBRT v4 — Curves (rotation-minimizing frame): https://pbr-book.org/4ed/Shapes/Curves

Procedural Heightfield Terrain
-----------------------------

- Object type: `heightfield` generated at load time as a triangle mesh.
- Parameters (JSON):
  - `GRID_X`, `GRID_Z`: vertex resolution along X/Z (controls mesh density)
  - `SIZE_X`, `SIZE_Z`: world-space extent of the terrain
  - `HEIGHT`: vertical amplitude (peak/trough height)
  - `NOISE_SCALE`: frequency of the driving noise (higher → finer features)
  - `OCTAVES`: number of fBm layers (higher → more small-scale detail)
  - Standard `TRANS`, `ROTAT`, `SCALE` are supported
- Implementation: 2D value-noise fBm drives height y over an XZ grid; two triangles per quad are emitted and a world-space AABB is computed for optional culling.
- Usage example:
  - `{ "TYPE":"heightfield", "MATERIAL":"diffuse_white", "GRID_X":64, "GRID_Z":64, "SIZE_X":6.0, "SIZE_Z":6.0, "HEIGHT":0.6, "NOISE_SCALE":1.2, "OCTAVES":5, "TRANS":[0,0.6,-3], "ROTAT":[0,0,0], "SCALE":[1,1,1] }`
- Notes:
  - Increase `GRID_X/Z` for smoother silhouettes; this increases triangle count.
  - Pair `SIZE_X/Z` changes with `NOISE_SCALE` to keep features visually consistent.
  - Apply any material, including procedural ones (e.g., Marble), since evaluation is in world/object space.

Subsurface Scattering (Approx.)
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


OBJ Mesh Loading
----------------

- JSON support for `mesh` objects with a `FILE` path (relative or absolute):
  - Example object:
    - `{"TYPE":"mesh", "MATERIAL":"diffuse_white", "FILE":"models/cube.obj", "TRANS":[0,0,0], "ROTAT":[0,0,0], "SCALE":[1,1,1]}`
- Loader parses OBJ `v` vertices and `f` faces, triangulates polygons, applies the per-object transform, and assigns the referenced material.
- For each mesh object, records triangle range (`triStart`, `triCount`) and computes a world-space AABB for optional bounds culling.
- Triangles are stored in `Scene::triangles` and uploaded once to GPU (`dev_triangles`) during `pathtraceInit`.
- Intersection uses Moller-Trumbore (`triangleIntersectionTest`) and an optional AABB quick-reject when "Mesh bounds culling" is enabled in the ImGui panel.
- Notes:
  - OBJ normals/UVs are ignored in this minimal loader; shading uses geometric normals from triangles.
  - Large meshes will increase intersection cost; consider enabling material sorting and bounds culling for better coherence.
