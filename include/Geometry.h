#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace recad {

struct Vec3d {
    double x{};
    double y{};
    double z{};

    [[nodiscard]] constexpr double& operator[](std::size_t i) noexcept {
        return i == 0U ? x : (i == 1U ? y : z);
    }
    [[nodiscard]] constexpr const double& operator[](std::size_t i) const noexcept {
        return i == 0U ? x : (i == 1U ? y : z);
    }

    constexpr Vec3d& operator+=(const Vec3d& rhs) noexcept {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }
    constexpr Vec3d& operator-=(const Vec3d& rhs) noexcept {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }
    constexpr Vec3d& operator*=(double scale) noexcept {
        x *= scale;
        y *= scale;
        z *= scale;
        return *this;
    }
    constexpr Vec3d& operator/=(double scale) noexcept {
        x /= scale;
        y /= scale;
        z /= scale;
        return *this;
    }
};

[[nodiscard]] constexpr Vec3d operator+(Vec3d lhs, const Vec3d& rhs) noexcept {
    return lhs += rhs;
}
[[nodiscard]] constexpr Vec3d operator-(Vec3d lhs, const Vec3d& rhs) noexcept {
    return lhs -= rhs;
}
[[nodiscard]] constexpr Vec3d operator-(const Vec3d& value) noexcept {
    return {-value.x, -value.y, -value.z};
}
[[nodiscard]] constexpr Vec3d operator*(Vec3d value, double scale) noexcept {
    return value *= scale;
}
[[nodiscard]] constexpr Vec3d operator*(double scale, Vec3d value) noexcept {
    return value *= scale;
}
[[nodiscard]] constexpr Vec3d operator/(Vec3d value, double scale) noexcept {
    return value /= scale;
}
[[nodiscard]] constexpr bool operator==(const Vec3d& lhs, const Vec3d& rhs) noexcept {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

[[nodiscard]] constexpr double dot(const Vec3d& lhs, const Vec3d& rhs) noexcept {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}
[[nodiscard]] constexpr Vec3d cross(const Vec3d& lhs, const Vec3d& rhs) noexcept {
    return {lhs.y * rhs.z - lhs.z * rhs.y,
            lhs.z * rhs.x - lhs.x * rhs.z,
            lhs.x * rhs.y - lhs.y * rhs.x};
}
[[nodiscard]] constexpr double squaredNorm(const Vec3d& value) noexcept {
    return dot(value, value);
}
[[nodiscard]] double norm(const Vec3d& value) noexcept;
[[nodiscard]] Vec3d normalized(const Vec3d& value,
                               double minimumNorm = std::numeric_limits<double>::min()) noexcept;
[[nodiscard]] bool isFinite(const Vec3d& value) noexcept;

struct Triangle {
    std::array<std::uint32_t, 3> vertices{};

    [[nodiscard]] constexpr std::uint32_t& operator[](std::size_t i) noexcept {
        return vertices[i];
    }
    [[nodiscard]] constexpr const std::uint32_t& operator[](std::size_t i) const noexcept {
        return vertices[i];
    }
};

struct TriangleFeatures {
    Vec3d normal{};
    Vec3d centroid{};
    double area{};
    double planeOffset{};
    double maxEdgeLength{};
    bool degenerate{true};
};

struct Aabb {
    Vec3d minimum{std::numeric_limits<double>::infinity(),
                  std::numeric_limits<double>::infinity(),
                  std::numeric_limits<double>::infinity()};
    Vec3d maximum{-std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity()};

    void expand(const Vec3d& point) noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] Vec3d center() const noexcept;
    [[nodiscard]] Vec3d extent() const noexcept;
    [[nodiscard]] double diagonal() const noexcept;
};

struct TopologyStats {
    std::size_t rawVertexCount{};
    std::size_t weldedVertexCount{};
    std::size_t inputTriangleCount{};
    std::size_t triangleCount{};
    std::size_t discardedDegenerateTriangles{};
    std::size_t uniqueEdgeCount{};
    std::size_t boundaryEdgeCount{};
    std::size_t manifoldEdgeCount{};
    std::size_t nonManifoldEdgeCount{};
    std::size_t connectedComponentCount{};
    std::size_t closedComponentCount{};
    std::int64_t eulerCharacteristic{};
    bool consistentlyOriented{};
    bool watertight{};
};

struct Mesh {
    std::vector<Vec3d> vertices;
    std::vector<Triangle> triangles;
    std::vector<TriangleFeatures> features;
    std::vector<std::vector<std::uint32_t>> triangleNeighbors;
    std::vector<std::vector<std::uint32_t>> vertexTriangles;
    Aabb bounds;
    TopologyStats stats;
    double weldTolerance{};
};

// Recomputes double-precision geometric features. The threshold is twice-area.
void computeTriangleFeatures(Mesh& mesh, double degeneracyThreshold = 0.0);

// Rebuilds edge incidence, triangle/vertex adjacency and topology statistics.
// Indices and features must already be valid. This function does not weld vertices.
void rebuildTopology(Mesh& mesh);

struct Plane {
    Vec3d normal{}; // Unit length for a valid plane.
    double offset{}; // normal dot point + offset = 0.

    [[nodiscard]] double signedDistance(const Vec3d& point) const noexcept {
        return dot(normal, point) + offset;
    }
    [[nodiscard]] double distance(const Vec3d& point) const noexcept;
    [[nodiscard]] Vec3d project(const Vec3d& point) const noexcept;
};

struct PlaneSegment {
    Plane plane;
    std::vector<std::uint32_t> triangleIndices;
    double area{};
    Vec3d centroid{};
    double rmsResidual{};
    double maxResidual{};
    std::size_t connectedComponents{1};
};

struct PlaneSegmentationOptions {
    // Zero selects a tolerance from mesh scale and floating-point resolution.
    double distanceTolerance{};
    double angleToleranceRadians{0.017453292519943295}; // 1 degree.
    // A candidate must exceed both nonzero thresholds.
    double minimumSupportArea{};
    double minimumSupportAreaFraction{1.0e-4};
    std::size_t minimumTriangleCount{2};
    std::size_t maximumRefinementIterations{8};
    bool mergeDisconnectedCoplanarRegions{true};
};

struct PlaneSegmentationResult {
    std::vector<PlaneSegment> segments;
    std::vector<std::uint32_t> nonPlanarTriangles;
    // -1 denotes non-planar. Entries correspond one-for-one with mesh.triangles.
    std::vector<std::int32_t> triangleLabels;
    double distanceTolerance{};
    double angleToleranceRadians{};
    double planarArea{};
    double nonPlanarArea{};
};

// Area-integrated TLS: triangle surface moments are used rather than duplicated
// vertex counts. Throws std::invalid_argument for an empty/degenerate selection.
[[nodiscard]] PlaneSegment fitPlaneTLS(
    const Mesh& mesh,
    std::span<const std::uint32_t> triangleIndices);

[[nodiscard]] PlaneSegmentationResult segmentPlanarRegions(
    const Mesh& mesh,
    const PlaneSegmentationOptions& options = {});

struct SphereFitOptions {
    std::size_t maximumIterations{64};
    double convergenceTolerance{1.0e-12};
    // Zero derives a robust scale from the current residual distribution.
    double huberThreshold{};
    double initialDamping{1.0e-6};
};

struct SphereFitResult {
    bool valid{};
    Vec3d center{};
    double radius{};
    double rmsResidual{std::numeric_limits<double>::infinity()};
    double maxResidual{std::numeric_limits<double>::infinity()};
    double medianResidual{std::numeric_limits<double>::infinity()};
    double inlierFraction{};
    // +1 means facet normals point radially out, -1 in, and 0 means ambiguous.
    int radialOrientation{};
    std::size_t pointCount{};
    std::size_t iterations{};
    std::string diagnostic;
};

// Uses all welded vertices incident to the supplied triangles, with incident
// triangle area as the vertex weight. This intentionally includes shared
// plane/curved-surface boundary vertices.
[[nodiscard]] SphereFitResult fitSphere(
    const Mesh& mesh,
    std::span<const std::uint32_t> surfaceTriangles,
    const SphereFitOptions& options = {});

struct OrientedBoxFitOptions {
    double parallelToleranceRadians{0.017453292519943295};
    double orthogonalityToleranceRadians{0.03490658503988659};
    // Zero derives a threshold from mesh scale and plane fit errors.
    double planeDistanceTolerance{};
};

struct OrientedBoxFitResult {
    bool valid{};
    Vec3d center{};
    std::array<Vec3d, 3> axes{}; // Right-handed, orthonormal frame.
    Vec3d dimensions{};
    Vec3d lowCoordinates{};
    Vec3d highCoordinates{};
    std::array<std::pair<std::uint32_t, std::uint32_t>, 3> supportingPlanePairs{};
    double rmsResidual{std::numeric_limits<double>::infinity()};
    double maxResidual{std::numeric_limits<double>::infinity()};
    double orthogonalityError{std::numeric_limits<double>::infinity()};
    double supportedArea{};
    double score{-std::numeric_limits<double>::infinity()};
    std::string diagnostic;
};

// Searches dynamically for three disjoint opposite parallel plane pairs, then
// projects their directions onto SO(3) and robustly refits the six offsets.
[[nodiscard]] OrientedBoxFitResult inferOrientedBox(
    const Mesh& mesh,
    std::span<const PlaneSegment> planeSegments,
    const OrientedBoxFitOptions& options = {});

} // namespace recad
