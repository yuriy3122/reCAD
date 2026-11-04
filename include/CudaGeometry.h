#pragma once

#include "Geometry.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace recad {

// Device metadata is returned with every operation so reports cannot
// accidentally describe a different adapter from the one that did the work.
struct CudaDeviceInfo {
    int ordinal{-1};
    std::string name;
    int computeCapabilityMajor{};
    int computeCapabilityMinor{};
    std::uint64_t totalGlobalMemoryBytes{};
    int multiprocessorCount{};
    int maxThreadsPerBlock{};
    int runtimeVersion{};
    int driverVersion{};
};

// CUDA-event measurements. Host packing/validation is intentionally excluded;
// totalGpuMs covers the three recorded stream stages below.
struct CudaTimings {
    float hostToDeviceMs{};
    float kernelMs{};
    float deviceToHostMs{};
    float totalGpuMs{};
};

struct CudaTriangleFeatureResult {
    std::vector<TriangleFeatures> features;
    std::size_t degenerateCount{};
    CudaDeviceInfo device;
    CudaTimings timings;
};

// Initial, topology-aware over-segmentation options. Adjacent triangles are
// connected when their oriented normals satisfy the dihedral threshold and
// every enabled centroid safeguard passes. A zero centroid limit disables that
// safeguard. The edge-relative limit is useful when a mesh contains triangles
// at very different absolute scales.
struct CudaTriangleSegmentationOptions {
    // Triangle soups often duplicate all indices, which is not a face prior.
    bool respectSourcePatchBoundaries{true};
    double maximumDihedralAngleRadians{0.5235987755982988}; // 30 degrees.
    double maximumCentroidDistance{};
    double maximumCentroidDistanceToEdgeRatio{};
    // Principal curvature discontinuities separate tangent plane/cylinder/sphere
    // regions. Estimated on CUDA from supplied smooth normals; zero disables.
    double maximumRelativeCurvatureDifference{0.35};
    // Enable only for meshes whose winding could not be made consistent. When
    // enabled, n and -n are treated as the same orientation and the maximum
    // meaningful angle is pi/2.
    bool treatOppositeNormalsAsAligned{};
};

struct CudaTriangleRegion {
    std::vector<std::uint32_t> triangleIndices;
    double area{};
    Vec3d centroid{}; // Triangle-area-weighted centroid.
};

struct CudaTriangleSegmentationResult {
    // Degenerate triangles receive -1. All other labels are contiguous from
    // zero and deterministic: regions are ordered by their lowest triangle
    // index and every region's triangle list is ascending.
    std::vector<std::int32_t> triangleLabels;
    std::vector<CudaTriangleRegion> regions;
    std::size_t adjacencyPairCount{};
    std::size_t mergedAdjacencyPairCount{};
    // Adjacencies rejected because the GLB's pre-weld index topology placed
    // the triangles in different source patches.
    std::size_t sourceBoundaryPairCount{};
    std::size_t degenerateTriangleCount{};
    CudaDeviceInfo device;
    CudaTimings timings;
    double hostUnionFindMs{};
};

struct CudaResidualStatistics {
    std::uint64_t sampleCount{};
    std::uint64_t inlierCount{};
    double sumSquaredResidual{};
    double rmsResidual{};
    double maxResidual{};
    double inlierFraction{};

    // If no weights are supplied every point has weight one. This makes the
    // weighted fields useful to both vertex and area-weighted callers.
    double weightSum{};
    double weightedSumSquaredResidual{};
    double weightedRmsResidual{};
    double inlierWeight{};
    double inlierWeightFraction{};
};

struct CudaResidualResult {
    std::vector<double> residuals;
    std::vector<std::uint8_t> inlierMask;
    CudaResidualStatistics statistics;
    CudaDeviceInfo device;
    CudaTimings timings;
};

// Returns all CUDA compute devices. A machine with a working CUDA runtime but
// no CUDA device returns an empty vector; runtime/driver failures throw.
[[nodiscard]] std::vector<CudaDeviceInfo> enumerateCudaDevices();

// Computes geometric (not imported GLB) triangle data. Indices and finite
// coordinates are validated before launch. A triangle is degenerate when its
// area is <= degenerateAreaTolerance.
[[nodiscard]] CudaTriangleFeatureResult computeTriangleFeaturesCuda(
    const std::vector<Vec3d>& vertices,
    const std::vector<Triangle>& triangles,
    double degenerateAreaTolerance = 0.0,
    int deviceOrdinal = -1);

// Classifies unique adjacent-triangle pairs on the GPU, then performs a
// deterministic union-find pass on the CPU. mesh.features and
// mesh.triangleNeighbors must each contain one entry per triangle (normally
// produced by computeTriangleFeaturesCuda followed by rebuildTopology).
[[nodiscard]] CudaTriangleSegmentationResult segmentTriangleRegionsCuda(
    const Mesh& mesh,
    const CudaTriangleSegmentationOptions& options = {},
    int deviceOrdinal = -1);

// Residual is |n dot p + d| / |n|, so thresholds are expressed in model-space
// distance even when the supplied plane normal is not normalized.
[[nodiscard]] CudaResidualResult evaluatePlaneResidualsCuda(
    const std::vector<Vec3d>& points,
    const Vec3d& planeNormal,
    double planeOffset,
    double inlierTolerance,
    const std::vector<double>& weights = {},
    int deviceOrdinal = -1);

// Residual is abs(|p - center| - radius).
[[nodiscard]] CudaResidualResult evaluateSphereResidualsCuda(
    const std::vector<Vec3d>& points,
    const Vec3d& center,
    double radius,
    double inlierTolerance,
    const std::vector<double>& weights = {},
    int deviceOrdinal = -1);

} // namespace recad
