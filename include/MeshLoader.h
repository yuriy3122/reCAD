#pragma once

#include "Geometry.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace recad {

struct MeshLoadOptions {
    // A positive value overrides the scale-aware weld tolerance.
    double weldTolerance{};
    double weldToleranceRelativeToDiagonal{1.0e-9};
    double weldToleranceRelativeToMedianEdge{1.0e-4};
    double minimumWeldTolerance{1.0e-12};
    bool applySceneTransforms{true};
    bool discardDegenerateTriangles{true};
    bool repairTriangleWinding{true};
    // -1 selects the glTF default scene (or scene zero when unspecified).
    int sceneIndex{-1};
};

struct MeshLoadResult {
    Mesh mesh;
    std::filesystem::path sourcePath;
    std::vector<std::string> warnings;
    std::size_t visitedNodeCount{};
    std::size_t loadedPrimitiveCount{};
};

class MeshLoadError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Loads a binary glTF scene, applies accumulated node transforms, expands all
// triangle primitives, position-welds their world-space vertices, repairs
// orientation where possible, and builds mesh topology.
[[nodiscard]] MeshLoadResult loadGlbMesh(
    const std::filesystem::path& path,
    const MeshLoadOptions& options = {});

} // namespace recad
