# reCAD: GLB to analytic CAD with CUDA segmentation

Native C++20/CUDA/OpenCascade reconstruction from tessellated GLB input. The
production path fits connected surface patches rather than assuming a particular
box/sphere Boolean expression. No filenames, reference dimensions, or CSG trees
are used to reconstruct geometry.

## Pipeline

1. Decode the GLB scene, apply transforms, convert units, weld positions, and
   repair winding. Preserve raw shared-index components before welding: CAD
   exporters often duplicate vertices at original face boundaries.
2. Compute triangle features and classify adjacent triangle pairs on CUDA.
   Classifications use dihedral angle, optional geometric guards, and source
   boundaries when useful. Deterministic CPU union-find assembles the GPU merge
   decisions into regions. Triangle soups with one source component per triangle
   automatically use geometric segmentation instead of that uninformative prior.
3. Fit plane, cylinder, sphere, and ring-torus candidates to each region using
   area-weighted least squares, robust Huber/LM refinement, positional and normal
   consistency, and a BIC-style model-complexity penalty. PCA, normal directions,
   shared planar directions, and spherical direction sampling initialize axes.
   Fixed-axis torus refinement precedes full refinement. Unresolved mixed regions
   can be split by deterministic local hypotheses and connected inliers.
4. Recover oriented boundary loops and holes; project and unwrap them in fitted
   surface coordinates. OpenCascade builds trimmed analytic faces, sews them with
   a fit/chord-aware tolerance, heals shells, and orients closed solids.
5. If no primitive explains a bounded chart, try OpenCascade
   `GeomPlate_BuildPlateSurface` with boundary curves and scattered interior
   constraints, then `GeomPlate_MakeApprox` to obtain a smooth B-spline surface.
   Check all source vertices, including those omitted from the fitting system.
   Folded or failed charts retain trimmed degree-one B-spline triangle faces.
   If the combined result cannot form a valid solid, the complete triangle
   B-spline fallback preserves the mesh geometry.
6. Validate topology and bidirectional mesh/BRep distance, then write STEP/BREP.
   Only after export can an optional STEP reference be loaded for independent
   topology, mass-property, bounds, and sampled boundary-distance comparison.

The NURBS path uses [OpenCascade's plate fitting and approximation APIs](https://dev.opencascade.org/doc/refman/html/package_geomplate.html).
The final triangle fallback is a faithful mesh representation, not a claim that
the original smooth CAD surface has been recovered.

## Accuracy contract

Exact analytic surface types and parameters can often be recovered from mesh
vertices. Original trimming curves, manufacturing dimensions, and feature history
cannot be guaranteed from a finite tessellation. Surface recognition coverage is
not a geometric accuracy percentage.

Every run writes `reconstruction.json`, `report.txt`, `segmentation.ply`,
`reconstructed.brep`, and `reconstructed.step`. Reports distinguish analytic,
smooth NURBS, and triangle-fallback output and record residuals and CUDA timings.

The five `basic_geometry` GLB/STEP pairs were tested locally with CUDA 13.2,
OpenCascade 8.0.0, and an RTX 5060. Every sample produced one valid closed solid,
100% analytic patch coverage, and the same face-type counts as its BRep reference:

| Sample | Planes | Cylinders | Spheres | Tori | Surface RMS (mm) | Sampled maximum incl. edges (mm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| box/profile double cut | 21 | 24 | 0 | 0 | 0.0000631 | 0.01770 |
| box/sphere | 6 | 0 | 1 | 0 | 0.00000091 | 0.00089 |
| hollow extrusion | 9 | 9 | 0 | 0 | 0.0003572 | 0.02087 |
| curved hollow extrusion | 5 | 5 | 0 | 8 | 0.0000448 | 0.02251 |
| curved extrusion/sphere | 6 | 4 | 1 | 8 | 0.0004196 | 0.09438 |

These are sampled bidirectional distances to the reference STEP boundary, not
certified Hausdorff bounds. Sampling uses up to 512 area-stratified surface
points plus 512 edge/vertex points per direction. Surface RMS is area weighted;
edge samples contribute to the maximum. Face counts match, but fragmented trim
edges mean full topological entity counts need not match. Relative volume error
across these samples was at most 0.032%.

The STEP CSG files describe the expected primitive families and were inspected
for test coverage. Neither STEP nor STEP CSG geometry enters fitting, trimming,
or reconstruction selection.

## Units and coordinates

glTF uses metres and a Y-up frame. The executable defaults to `--input-scale 1000`
and writes STEP in millimetres while retaining the GLB frame. Use
`--input-scale 1` for legacy files authored directly in millimetres, including
`testData/box_sphere_test-01.glb`.

`--reference-z-up` explicitly rotates a CAD STEP reference `(x,y,z)` to the GLB
frame `(x,z,-y)` for comparison. This reference transform does not alter output
geometry. It is used by the `basic_geometry` test runner.

## Build and run

Open **`reCAD.sln`** in Visual Studio 2026. Select **Debug | x64** or
**Release | x64**, then Build Solution. The `reCAD` project is the application;
`reCAD.Tests` builds the test executable. F5 starts the application with the
included legacy GLB fixture and its correct unit scale.

Both projects use native MSBuild C++/CUDA integration with the v145 toolset.
Executables and runtime DLLs go to `build/native/bin/<Configuration>`; intermediate
files stay under `build/native/obj`. The solution and projects live in the source
repository, so Visual Studio uses the root Git repository.

Install the VS 2026 Desktop development with C++ workload and CUDA 13.2 with
Visual Studio integration. Eigen3, nlohmann/json, tinygltf, and OpenCascade
dependencies live under `.deps`. For a fresh checkout, run
`scripts/bootstrap-dependencies.ps1` once, or let the build helper install missing
dependencies.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-vs2026.ps1 -Configuration Release

.\build\native\bin\Release\mesh2brep.exe `
  --input model.glb --output output\model --verify
```

Use `--help` for options, including `--fit-tolerance`, `--smooth-angle`,
`--ignore-source-patches`, `--force-nurbs`, and `--verify-analytic`.
`--force-nurbs` explicitly chooses the triangle-preserving fallback.
The native projects target CUDA architecture 120. Override the
`RecadCudaArchitecture` MSBuild property for another GPU, for example
`/p:RecadCudaArchitecture=89`. Shared compiler, linker, and runtime settings live
in `vs/NativeProject.props` and `vs/NativeProject.targets`.

To compare a reconstructed model against a Z-up STEP reference:

```powershell
.\build\native\bin\Release\mesh2brep.exe `
  --input model.glb --output output\model `
  --reference-step reference.step --reference-z-up --verify-analytic
```

`--verify` checks solid validity and scale-relative mesh fidelity. When a
reference is supplied, it also requires comparison to complete without failed
samples. The dataset runner additionally checks surface signatures and reference
distance limits; reference geometry never determines the reconstructed result.

## Tests

```powershell
powershell -ExecutionPolicy Bypass -File .\tests\run-integration.ps1 -Configuration Release

powershell -ExecutionPolicy Bypass -File .\tests\run-samples.ps1 `
  -DatasetRoot C:\Users\YuriyV\Downloads\basic_geometry `
  -OutputRoot .\output\basic_geometry
```

The native test runner covers CUDA merge/split decisions, source boundaries, deterministic labels,
invalid adjacency, rotated cylinder/torus fitting, a smooth free-form NURBS patch
inside a closed solid, triangle-preserving fallback, known reference distances,
and the original GLB fixture. The dataset runner requires every sample to be
fully analytic, closed, valid, and within its reference tolerances.

## Main implementation files

| File | Responsibility |
| --- | --- |
| `src/MeshLoader.cpp` | GLB transforms, unit scale, source provenance, welding |
| `src/CudaGeometry.cu` | GPU features, adjacency classification, residual kernels |
| `src/SurfaceFitting.cpp` | Robust primitive fitting and model selection |
| `src/PatchReconstruction.cpp` | Analytic trimming, smooth NURBS fitting, sewing |
| `src/GeneralReconstruction.cpp` | Mesh-preserving B-spline fallback |
| `src/ReferenceValidation.cpp` | Independent post-export STEP comparison |
| `src/Distance.cpp` | Mesh/BRep distance and mesh BVH |
| `src/main.cpp` | CLI, orchestration, reports, validation/export |

Remaining limits include unsupported analytic families such as cones, unproven
robustness on noisy scans or folded free-form charts, approximate trim curves,
and no recovery of an original CSG construction history. Unsupported surfaces
use the NURBS/fallback path instead of being forced into a known primitive.
