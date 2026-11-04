#include "CudaGeometry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using recad::CudaTriangleSegmentationOptions;
using recad::CudaTriangleSegmentationResult;
using recad::Mesh;
using recad::Triangle;
using recad::TriangleFeatures;
using recad::Vec3d;

constexpr double kPi = 3.141592653589793238462643383279502884;

[[nodiscard]] Vec3d subtract(const Vec3d& left, const Vec3d& right) {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

[[nodiscard]] Vec3d crossProduct(const Vec3d& left, const Vec3d& right) {
    return {left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x};
}

[[nodiscard]] double dotProduct(const Vec3d& left, const Vec3d& right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

[[nodiscard]] double length(const Vec3d& value) {
    return std::sqrt(dotProduct(value, value));
}

[[nodiscard]] TriangleFeatures triangleFeatures(const Mesh& mesh,
                                                const Triangle& triangle) {
    const Vec3d& first = mesh.vertices[triangle.vertices[0]];
    const Vec3d& second = mesh.vertices[triangle.vertices[1]];
    const Vec3d& third = mesh.vertices[triangle.vertices[2]];
    const Vec3d firstEdge = subtract(second, first);
    const Vec3d secondEdge = subtract(third, first);
    const Vec3d oppositeEdge = subtract(third, second);
    const Vec3d twiceAreaVector = crossProduct(firstEdge, secondEdge);
    const double twiceArea = length(twiceAreaVector);

    TriangleFeatures result;
    result.centroid = {(first.x + second.x + third.x) / 3.0,
                       (first.y + second.y + third.y) / 3.0,
                       (first.z + second.z + third.z) / 3.0};
    result.area = 0.5 * twiceArea;
    result.maxEdgeLength = std::max(
        {length(firstEdge), length(secondEdge), length(oppositeEdge)});
    result.degenerate = !(twiceArea > 0.0);
    if (!result.degenerate) {
        result.normal = {twiceAreaVector.x / twiceArea,
                         twiceAreaVector.y / twiceArea,
                         twiceAreaVector.z / twiceArea};
        result.planeOffset = -dotProduct(result.normal, first);
    }
    return result;
}

[[nodiscard]] std::size_t sharedVertexCount(const Triangle& first,
                                            const Triangle& second) {
    std::size_t result = 0U;
    for (std::size_t i = 0; i < first.vertices.size(); ++i) {
        bool duplicate = false;
        for (std::size_t previous = 0; previous < i; ++previous) {
            duplicate = duplicate || first.vertices[previous] == first.vertices[i];
        }
        if (!duplicate &&
            std::find(second.vertices.begin(), second.vertices.end(), first.vertices[i]) !=
                second.vertices.end()) {
            ++result;
        }
    }
    return result;
}

void finishMesh(Mesh& mesh) {
    mesh.features.clear();
    mesh.features.reserve(mesh.triangles.size());
    for (const Triangle& triangle : mesh.triangles) {
        mesh.features.push_back(triangleFeatures(mesh, triangle));
    }

    mesh.triangleNeighbors.assign(mesh.triangles.size(), {});
    for (std::size_t first = 0; first < mesh.triangles.size(); ++first) {
        for (std::size_t second = first + 1U; second < mesh.triangles.size(); ++second) {
            if (sharedVertexCount(mesh.triangles[first], mesh.triangles[second]) < 2U) {
                continue;
            }
            mesh.triangleNeighbors[first].push_back(static_cast<std::uint32_t>(second));
            mesh.triangleNeighbors[second].push_back(static_cast<std::uint32_t>(first));
        }
    }
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

[[nodiscard]] Mesh twoCoplanarTriangles() {
    Mesh mesh;
    mesh.vertices = {{0.0, 0.0, 0.0},
                     {1.0, 0.0, 0.0},
                     {1.0, 1.0, 0.0},
                     {0.0, 1.0, 0.0}};
    mesh.triangles = {{{0U, 1U, 2U}}, {{0U, 2U, 3U}}};
    finishMesh(mesh);
    return mesh;
}

void testCoplanarTrianglesMerge() {
    const Mesh mesh = twoCoplanarTriangles();
    CudaTriangleSegmentationOptions options;
    options.maximumDihedralAngleRadians = kPi / 180.0;
    const CudaTriangleSegmentationResult result =
        recad::segmentTriangleRegionsCuda(mesh, options);

    require(result.adjacencyPairCount == 1U, "coplanar pair count must be one");
    require(result.mergedAdjacencyPairCount == 1U, "coplanar pair must merge");
    require(result.regions.size() == 1U, "coplanar triangles must form one region");
    require(result.triangleLabels == std::vector<std::int32_t>({0, 0}),
            "coplanar labels must both be zero");
    require(result.regions[0].triangleIndices == std::vector<std::uint32_t>({0U, 1U}),
            "coplanar region membership must be deterministic");
    require(std::abs(result.regions[0].area - 1.0) < 1.0e-12,
            "coplanar region area must equal the square area");
}

void testSourcePatchBoundaryWinsOverCoplanarity() {
    Mesh mesh = twoCoplanarTriangles();
    mesh.sourcePatchLabels = {0U, 1U};
    mesh.sourcePatchCount = 2U;
    CudaTriangleSegmentationOptions options;
    options.maximumDihedralAngleRadians = kPi / 180.0;
    const CudaTriangleSegmentationResult result =
        recad::segmentTriangleRegionsCuda(mesh, options);

    require(result.adjacencyPairCount == 1U, "source-boundary pair count must be one");
    require(result.sourceBoundaryPairCount == 1U,
            "the pre-weld source boundary must be reported");
    require(result.mergedAdjacencyPairCount == 0U,
            "coplanar triangles from different source patches must not merge");
    require(result.triangleLabels == std::vector<std::int32_t>({0, 1}),
            "source boundary must produce two deterministic regions");
    options.respectSourcePatchBoundaries = false;
    const auto geometric = recad::segmentTriangleRegionsCuda(mesh, options);
    require(geometric.regions.size() == 1U && geometric.sourceBoundaryPairCount == 0U,
            "Disabling source provenance must restore geometric segmentation");
}

void testSharpFacesSplit() {
    Mesh mesh;
    mesh.vertices = {{0.0, 0.0, 0.0},
                     {1.0, 0.0, 0.0},
                     {0.0, 1.0, 0.0},
                     {0.0, 0.0, 1.0}};
    // The triangles share edge 0-1 with opposite edge orientation, while their
    // outward normals differ by 90 degrees.
    mesh.triangles = {{{0U, 1U, 2U}}, {{1U, 0U, 3U}}};
    finishMesh(mesh);

    CudaTriangleSegmentationOptions options;
    options.maximumDihedralAngleRadians = 30.0 * kPi / 180.0;
    const CudaTriangleSegmentationResult result =
        recad::segmentTriangleRegionsCuda(mesh, options);

    require(result.adjacencyPairCount == 1U, "sharp pair count must be one");
    require(result.mergedAdjacencyPairCount == 0U, "90-degree faces must not merge");
    require(result.regions.size() == 2U, "90-degree faces must form two regions");
    require(result.triangleLabels == std::vector<std::int32_t>({0, 1}),
            "sharp-face labels must follow triangle order");
}

[[nodiscard]] Mesh cylinderLikeStrip() {
    Mesh mesh;
    constexpr std::size_t segmentCount = 3U;
    constexpr double angleStep = 15.0 * kPi / 180.0;
    for (std::size_t ringIndex = 0; ringIndex <= segmentCount; ++ringIndex) {
        const double angle = static_cast<double>(ringIndex) * angleStep;
        const double x = std::cos(angle);
        const double y = std::sin(angle);
        mesh.vertices.push_back({x, y, 0.0});
        mesh.vertices.push_back({x, y, 1.0});
    }
    for (std::size_t segment = 0; segment < segmentCount; ++segment) {
        const std::uint32_t lowerFirst = static_cast<std::uint32_t>(2U * segment);
        const std::uint32_t upperFirst = lowerFirst + 1U;
        const std::uint32_t lowerSecond = lowerFirst + 2U;
        const std::uint32_t upperSecond = lowerFirst + 3U;
        mesh.triangles.push_back({{lowerFirst, lowerSecond, upperSecond}});
        mesh.triangles.push_back({{lowerFirst, upperSecond, upperFirst}});
    }
    finishMesh(mesh);
    return mesh;
}

void testSmoothCylinderLikeStripMerges() {
    const Mesh mesh = cylinderLikeStrip();
    CudaTriangleSegmentationOptions options;
    options.maximumDihedralAngleRadians = 20.0 * kPi / 180.0;
    options.maximumCentroidDistanceToEdgeRatio = 1.0;
    const CudaTriangleSegmentationResult result =
        recad::segmentTriangleRegionsCuda(mesh, options);

    require(result.regions.size() == 1U,
            "15-degree cylinder-like facets must form one smooth region");
    require(result.mergedAdjacencyPairCount == result.adjacencyPairCount,
            "every adjacency in the smooth strip must merge");
    require(result.regions[0].triangleIndices.size() == mesh.triangles.size(),
            "the smooth region must contain every strip triangle");
    require(std::all_of(result.triangleLabels.begin(), result.triangleLabels.end(),
                        [](std::int32_t label) { return label == 0; }),
            "all smooth-strip labels must be zero");
}

[[nodiscard]] Mesh interleavedDisconnectedSquares() {
    Mesh mesh;
    mesh.vertices = {{0.0, 0.0, 0.0},
                     {1.0, 0.0, 0.0},
                     {1.0, 1.0, 0.0},
                     {0.0, 1.0, 0.0},
                     {3.0, 0.0, 0.0},
                     {4.0, 0.0, 0.0},
                     {4.0, 1.0, 0.0},
                     {3.0, 1.0, 0.0}};
    // Component A uses triangle indices 0 and 3; component B uses 1 and 2.
    mesh.triangles = {{{0U, 1U, 2U}},
                      {{4U, 5U, 6U}},
                      {{4U, 6U, 7U}},
                      {{0U, 2U, 3U}}};
    finishMesh(mesh);
    return mesh;
}

void testLabelsAreDeterministic() {
    Mesh mesh = interleavedDisconnectedSquares();
    CudaTriangleSegmentationOptions options;
    options.maximumDihedralAngleRadians = kPi / 180.0;
    const CudaTriangleSegmentationResult first =
        recad::segmentTriangleRegionsCuda(mesh, options);

    // Duplicate and reorder adjacency entries. Canonical pair construction must
    // produce exactly the same regions and labels.
    for (std::vector<std::uint32_t>& neighbors : mesh.triangleNeighbors) {
        if (!neighbors.empty()) neighbors.push_back(neighbors.front());
        std::reverse(neighbors.begin(), neighbors.end());
    }
    const CudaTriangleSegmentationResult second =
        recad::segmentTriangleRegionsCuda(mesh, options);

    const std::vector<std::int32_t> expected{0, 1, 1, 0};
    require(first.triangleLabels == expected, "component labels must use lowest triangle order");
    require(second.triangleLabels == expected,
            "labels must not depend on adjacency entry order or duplication");
    require(first.adjacencyPairCount == 2U && second.adjacencyPairCount == 2U,
            "canonical adjacency pairs must be deduplicated");
    require(first.regions.size() == second.regions.size(),
            "deterministic runs must produce the same region count");
    for (std::size_t region = 0; region < first.regions.size(); ++region) {
        require(first.regions[region].triangleIndices == second.regions[region].triangleIndices,
                "deterministic runs must preserve region membership order");
    }
}

void testAsymmetricAdjacencyIsRejected() {
    Mesh mesh = twoCoplanarTriangles();
    mesh.triangleNeighbors[1].clear();
    bool rejected = false;
    try {
        static_cast<void>(recad::segmentTriangleRegionsCuda(mesh));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "asymmetric adjacency must throw std::invalid_argument");
}

void testSmoothCurvatureBoundary() {
    Mesh mesh = twoCoplanarTriangles();
    for (const auto& point : mesh.vertices) mesh.bounds.expand(point);
    mesh.cornerNormals.resize(2);
    for (auto& normals : mesh.cornerNormals) normals.fill({0.0, 0.0, 1.0});
    // Normals agree on the shared tangent edge. Only the second patch bends
    // away from that edge; its average normal is still within the angle guard.
    mesh.cornerNormals[1][2] = recad::normalized(Vec3d{-0.05, 0.05, 1.0});
    CudaTriangleSegmentationOptions options;
    options.maximumDihedralAngleRadians = 10.0 * kPi / 180.0;
    options.maximumRelativeCurvatureDifference = 0.0;
    const auto angleOnly = recad::segmentTriangleRegionsCuda(mesh, options);
    require(angleOnly.regions.size() == 1U, "Tangent patches pass the angle-only guard");
    options.maximumRelativeCurvatureDifference = 0.35;
    const auto curvature = recad::segmentTriangleRegionsCuda(mesh, options);
    require(curvature.regions.size() == 2U, "CUDA curvature must separate distinct tangent surfaces");
}

} // namespace

int runCudaSegmentationTests() {
    try {
        testCoplanarTrianglesMerge();
        testSourcePatchBoundaryWinsOverCoplanarity();
        testSharpFacesSplit();
        testSmoothCylinderLikeStripMerges();
        testLabelsAreDeterministic();
        testAsymmetricAdjacencyIsRejected();
        testSmoothCurvatureBoundary();
        std::cout << "CUDA triangle segmentation tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "CUDA triangle segmentation test failure: " << exception.what() << '\n';
        return 1;
    }
}

#ifndef RECAD_COMBINED_TESTS
int main() { return runCudaSegmentationTests(); }
#endif
