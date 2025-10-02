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
