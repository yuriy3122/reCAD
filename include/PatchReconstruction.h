#pragma once

#include "SurfaceFitting.h"

#include <TopoDS_Shape.hxx>

#include <cstddef>
#include <string>
#include <vector>

namespace recad {

struct PatchReconstructionOptions {
    // Zero derives a tolerance from the mesh scale, weld tolerance, accepted
    // surface-fit residuals, and the curved-patch chordal sag represented by
    // the source tessellation.
    double sewingTolerance{};

    // Zero derives a per-patch limit from the sewing tolerance and the
    // accepted fit residual. An analytic patch whose boundary projects farther
    // than this limit is rejected.
    double maximumBoundaryProjectionDistance{};
    bool enableNurbsFitting{true};
    // Zero derives the fitting tolerance from mesh scale and weld precision.
    double nurbsFitTolerance{};
};

struct PatchReconstructionResult {
    TopoDS_Shape shape;

    bool succeeded{};
    bool valid{};
    bool closed{};
    bool inputReportedWatertight{};

    double sewingTolerance{};
    std::size_t inputTriangleCount{};
    std::size_t inputPatchCount{};
    std::size_t acceptedAnalyticPatchCount{};
    std::size_t analyticPatchFailureCount{};
    std::size_t analyticFaceCount{};
    std::size_t fittedNurbsFaceCount{};
    std::size_t skippedTriangleCount{};
    std::size_t shellCount{};
    std::size_t closedShellCount{};
    std::size_t solidCount{};
    std::size_t freeEdgeCount{};
    std::size_t multipleEdgeCount{};

    std::vector<std::string> diagnostics;
};

// Builds trimmed analytic faces for accepted plane, cylinder, sphere, and
// torus patches. Patch boundaries are inferred solely from triangle topology,
// projected to the fitted surface, unwrapped in periodic parameter directions,
// and assembled as outer wires plus holes. Unrecognized bounded surfaces use
// fitted NURBS. Every input triangle must be covered by an accepted surface;
// failures return diagnostics without substituting triangle faces.
//
// Patch triangle sets must be disjoint and all indices must be in range.
[[nodiscard]] PatchReconstructionResult buildPatchReconstruction(
    const Mesh& mesh,
    const std::vector<SurfacePatchFit>& patches,
    const PatchReconstructionOptions& options = {});

} // namespace recad
