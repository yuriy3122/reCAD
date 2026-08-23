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
