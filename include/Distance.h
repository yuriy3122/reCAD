#pragma once

#include "Geometry.h"

#include <TopoDS_Shape.hxx>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace recad {

struct DistanceStatistics {
    double rms = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double weight = 0.0;
    std::size_t samples = 0;
};

struct BidirectionalDistance {
    DistanceStatistics meshToBrep;
    DistanceStatistics brepToMesh;
    double symmetricRms = 0.0;
};

class TriangleBvh {
public:
    explicit TriangleBvh(const Mesh& mesh);
    [[nodiscard]] double distance(const Vec3d& point) const;

private:
    struct Node {
        Aabb bounds;
        std::uint32_t begin = 0;
        std::uint32_t count = 0;
        std::uint32_t left = 0;
        std::uint32_t right = 0;
        bool leaf = false;
    };

    const Mesh* mesh_ = nullptr;
    std::vector<std::uint32_t> indices_;
    std::vector<Node> nodes_;

    std::uint32_t build(std::uint32_t begin, std::uint32_t end);
    double distanceSquared(std::uint32_t nodeIndex, const Vec3d& point, double best) const;
};

[[nodiscard]] DistanceStatistics meshToBrepDistance(
    const Mesh& mesh, const TopoDS_Shape& shape, bool includeVertexMaximum = true,
    std::size_t maximumSamples = 0U);

[[nodiscard]] DistanceStatistics brepToMeshDistance(
    const TopoDS_Shape& shape, const Mesh& mesh, double linearDeflection,
    double angularDeflection = 0.08);

[[nodiscard]] BidirectionalDistance compareMeshAndBrep(
    const Mesh& mesh, const TopoDS_Shape& shape, double linearDeflection,
    std::size_t maximumMeshSamples = 0U);

[[nodiscard]] Aabb brepBounds(const TopoDS_Shape& shape);

} // namespace recad

