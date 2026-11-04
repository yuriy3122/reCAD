# reCAD: GLB to analytic CAD with CUDA segmentation

Native C++20/CUDA/OpenCascade reconstruction from tessellated GLB input. The
experimental reconstruction path fits connected surface patches rather than assuming a particular
box/sphere Boolean expression. No filenames, reference dimensions, or CSG trees
are used to reconstruct geometry.

## Pipeline

1. Decode the GLB scene, including Draco compression, apply transforms to positions
   and normals, convert units, weld positions, conform T-junctions, and repair
   winding. Preserve raw shared-index components before welding: CAD
   exporters often duplicate vertices at original face boundaries.
2. Compute triangle features and classify adjacent triangle pairs on CUDA.
   Classifications use dihedral angle, smooth-normal curvature, geometric guards, and source
   boundaries when useful. Deterministic CPU union-find assembles the GPU merge
   decisions into regions. Triangle soups with one source component per triangle
   automatically use geometric segmentation instead of that uninformative prior.
3. Fit plane, cylinder, cone, sphere, and ring-torus candidates to each region using
   area-weighted least squares, robust Huber/LM refinement, positional and normal
   consistency, and a BIC-style model-complexity penalty. PCA, normal directions,
   shared planar directions, and spherical direction sampling initialize axes.
   Fixed-axis torus refinement precedes full refinement. Unresolved mixed regions
   can be split by deterministic local hypotheses and connected inliers.
   Model-based growth and label refinement join fragmented seeds; physical
   support area prevents tessellation slivers from dominating model selection.
   Area-weighted metric Potts alpha expansion jointly assigns triangles to
   fitted surfaces, with a boundary-length penalty. Seeds smaller than the
   spatial noise footprint are deferred to joint assignment; unexplained regions
   are fitted afterward. The solver uses iterative push/relabel, without recursion.
4. Recover oriented boundary loops and holes; project and unwrap them in fitted
   surface coordinates. OpenCascade builds trimmed analytic faces, sews them with
   a fit/chord-aware tolerance, heals shells, and orients closed solids.
5. If no primitive explains a bounded chart, try OpenCascade
   `GeomPlate_BuildPlateSurface` with boundary curves and scattered interior
   constraints, then `GeomPlate_MakeApprox` to obtain a smooth B-spline surface.
   Check all source vertices, including those omitted from the fitting system.
   Failed fitting, trimming, or solid assembly terminates reconstruction with
   an explicit error. There is no triangle-face substitute.
6. Validate topology and bidirectional mesh/BRep distance, then write STEP/BREP.
   Only after export can an optional STEP reference be loaded for independent
   topology, mass-property, bounds, and sampled boundary-distance comparison.

The NURBS path uses [OpenCascade's plate fitting and approximation APIs](https://dev.opencascade.org/doc/refman/html/package_geomplate.html).
Fitting, label refinement, and OpenCascade construction run on the CPU. CUDA
accelerates initial triangle features, curvature, and adjacency classification.
Graph optimization follows [Boykov, Veksler, and Zabih (PAMI 2001)](https://www.cs.cornell.edu/rdz/Papers/BVZ-pami01-final.pdf).

## Accuracy contract

Exact analytic surface types and parameters can often be recovered from mesh
vertices. Original trimming curves, manufacturing dimensions, and feature history
cannot be guaranteed from a finite tessellation. Surface recognition coverage is
not a geometric accuracy percentage.

Successful runs write `reconstruction.json`, `report.txt`, `segmentation.ply`,
`reconstructed.brep`, and `reconstructed.step`. Reports distinguish analytic and
smooth NURBS output and record residuals and CUDA timings. `surface_fitting.json`
is available even when solid construction fails. Check `run_status.json` when
reusing an output directory: older exports may remain after a failed run.

The five `glb-smooth` GLB/STEP CSG pairs were verified on 2026-11-04 with
CUDA 13.2, OpenCascade 8.0.0, and an RTX 5060. Every output is one valid closed
solid with analytic surfaces and zero B-spline faces. No reference geometry is
available to the fitting or construction stages.

| Sample | Analytic faces | Reference RMS (mm) | Sampled maximum (mm) | Volume error (%) | CPU fitting (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CSG intersection/subtraction | 18 | 0.003000 | 0.083159 | 0.118577 | 3.428 |
| Plate with holes | 11 | 0.001204 | 0.009414 | 0.024092 | 0.033 |
| Bracket with fillet | 9 | 0.000465 | 0.003005 | 0.003255 | 0.001 |
| Rounded box | 26 | 0.002609 | 0.045201 | 0.165818 | 4.581 |
| Pulley with groove | 7 | 0.001332 | 0.010052 | 0.020843 | 0.135 |

The rounded box contains six planes, twelve cylinders, and eight spheres. The
CSG sample's spherical surface uses multiple parameter charts; face and edge
counts can differ from the reference even when the geometry is close.

The accepted validation run uses an explicit volume-error limit of **0.166%**,
surface RMS at most **0.005 mm**, maximum sampled distance at most **0.1 mm**,
and the runner's additional relative-distance limits. Sampling uses up to 2,048
area-stratified surface points plus 2,048 boundary points per direction. RMS is
area weighted; edge samples contribute to the maximum. OCCT mass properties
use adaptive integration. These measurements are not certified Hausdorff bounds.
Fitting times exclude construction, export, and independent reference validation.

Detailed local reports are in `output/validated-glb-smooth`. The bundled original
GLB also passes `--verify-analytic`. The older `basic_geometry` dataset was
unavailable for this regression run. This five-sample result does not establish
reliability on hundreds of unseen models.

STEP CSG references are evaluated independently using OCCT primitive and Boolean
builders. Neither STEP nor STEP CSG geometry enters fitting, trimming, or
reconstruction selection.

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
Visual Studio integration. Eigen3, nlohmann/json, tinygltf, Draco, and OpenCascade
dependencies live under `.deps`. For a fresh checkout, run
`scripts/bootstrap-dependencies.ps1` once, or let the build helper install missing
dependencies.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-vs2026.ps1 -Configuration Release

.\build\native\bin\Release\mesh2brep.exe `
  --input model.glb --output output\model --verify
```

Use `--help` for options, including `--fit-tolerance`, `--smooth-angle`,
`--ignore-source-patches`, and `--verify-analytic`.
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
samples. The dataset runner additionally checks reference distance and volume
limits; `-RequireSurfaceSignature` also requires matching surface-type counts.
Reference geometry never determines the reconstructed result. Use
`--reference-samples 2048` (or `-ReferenceSamples 2048` in the runner) for denser
sampling. Sampled errors are not certified Hausdorff bounds.

## Tests

```powershell
powershell -ExecutionPolicy Bypass -File .\tests\run-integration.ps1 -Configuration Release

powershell -ExecutionPolicy Bypass -File .\tests\run-samples.ps1 `
  -DatasetRoot C:\Users\YuriyV\Downloads\glb-smooth `
  -OutputRoot .\output\validated-glb-smooth `
  -ReferenceSamples 2048 -MaximumRelativeVolumeError 0.00166
```

The dataset runner discovers GLBs recursively and matches STEP reference stems,
including exporter suffixes. Use `-ReferenceRoot` to resolve ambiguous references.
It continues after failures, saves metrics after each file, and enforces
`-TimeoutSeconds` (default 300). `-ReuseCompleted` rechecks fresh completed reports
without rerunning reconstruction; `Reused` identifies those rows and `Seconds`
measures the current runner invocation. Limits and paths are saved in
`validation_settings.json`.

The native test runner covers CUDA merge/split decisions, source boundaries, deterministic labels,
exact graph-cut checks against exhaustive search and dynamic programming,
invalid adjacency, rotated cylinder/torus fitting, rotated and scaled cones,
a smooth free-form NURBS patch inside a closed solid, explicit reconstruction
failure, known reference distances, trimmed periodic surface sampling,
and the original GLB fixture. The dataset runner requires every sample to be
fully analytic, closed, valid, and within its reference tolerances.

## Main implementation files

| File | Responsibility |
| --- | --- |
| `src/MeshLoader.cpp` | GLB transforms, unit scale, source provenance, welding |
| `src/CudaGeometry.cu` | GPU features, adjacency classification, residual kernels |
| `src/SurfaceFitting.cpp` | Robust primitive fitting and model selection |
| `src/PatchReconstruction.cpp` | Analytic trimming, smooth NURBS fitting, sewing |
| `src/ReferenceValidation.cpp` | Independent post-export STEP comparison |
| `src/Distance.cpp` | Mesh/BRep distance and mesh BVH |
| `src/main.cpp` | CLI, orchestration, reports, validation/export |

Remaining limits include unproven robustness on noisy scans or folded free-form
charts, approximate trim curves, and no recovery of an original CSG construction
history. An unsupported primitive requires a validated fitted NURBS surface;
otherwise reconstruction fails. Dataset results do not establish reliability
on arbitrary unseen GLBs.
