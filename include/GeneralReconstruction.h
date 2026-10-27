#pragma once

#include "Geometry.h"

#include <TopoDS_Shape.hxx>

#include <cstddef>
#include <string>
#include <vector>

namespace recad {

// Result of the geometry-preserving fallback used when analytic primitive
// recovery cannot explain all of a mesh.  Each input triangle is represented
// by a trimmed degree-one B-spline surface, then the faces are sewn and healed
// into one or more solids.
struct GeneralReconstructionResult {
    TopoDS_Shape shape;

    bool succeeded{};
    bool valid{};
    bool closed{};
    bool inputReportedWatertight{};

    double sewingTolerance{};
    std::size_t inputTriangleCount{};
    std::size_t createdFaceCount{};
    std::size_t skippedTriangleCount{};
    std::size_t shellCount{};
    std::size_t closedShellCount{};
    std::size_t solidCount{};
    std::size_t freeEdgeCount{};
    std::size_t multipleEdgeCount{};

    std::vector<std::string> diagnostics;
};

// Builds an exact representation of the input triangulation using trimmed
// degree-one OpenCascade B-spline surfaces.  A positive sewingTolerance is
// used verbatim; zero derives a scale-aware value from the mesh.  Negative or
// non-finite tolerances and empty meshes are rejected with invalid_argument.
//
// The function does not require Mesh::features to be populated.  Triangle
// winding is preserved while faces are built; closed shells are subsequently
// oriented as finite solids by OpenCascade.  Disconnected closed components
// are returned as a compound of solids.
[[nodiscard]] GeneralReconstructionResult buildNurbsFallback(
    const Mesh& mesh,
    double sewingTolerance = 0.0);

} // namespace recad
