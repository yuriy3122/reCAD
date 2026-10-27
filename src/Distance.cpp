#include "Distance.h"

#include <BRepBndLib.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRep_Builder.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <Poly_Triangulation.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Compound.hxx>
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Trsf.hxx>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace recad {
namespace {

double component(const Vec3d& p, int axis) {
    if (axis == 0) return p.x;
    if (axis == 1) return p.y;
    return p.z;
}

Vec3d minimum(const Vec3d& a, const Vec3d& b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}

Vec3d maximum(const Vec3d& a, const Vec3d& b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

double pointAabbDistanceSquared(const Vec3d& p, const Aabb& box) {
    double result = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        const double value = component(p, axis);
        const double low = component(box.minimum, axis);
        const double high = component(box.maximum, axis);
        const double delta = value < low ? low - value : (value > high ? value - high : 0.0);
        result += delta * delta;
    }
    return result;
}

double pointTriangleDistanceSquared(
    const Vec3d& p, const Vec3d& a, const Vec3d& b, const Vec3d& c) {
    // Closest-point region tests from Real-Time Collision Detection (Ericson).
    const Vec3d ab = b - a;
    const Vec3d ac = c - a;
    const Vec3d ap = p - a;
    const double d1 = dot(ab, ap);
    const double d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return squaredNorm(ap);

    const Vec3d bp = p - b;
    const double d3 = dot(ab, bp);
    const double d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return squaredNorm(bp);

    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double v = d1 / (d1 - d3);
        return squaredNorm(p - (a + ab * v));
    }

    const Vec3d cp = p - c;
    const double d5 = dot(ab, cp);
    const double d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return squaredNorm(cp);

    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w = d2 / (d2 - d6);
        return squaredNorm(p - (a + ac * w));
    }

    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const Vec3d bc = c - b;
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return squaredNorm(p - (b + bc * w));
    }

    const double denominator = 1.0 / (va + vb + vc);
    const double v = vb * denominator;
    const double w = vc * denominator;
    return squaredNorm(p - (a + ab * v + ac * w));
}

double exactPointShapeDistance(const Vec3d& p, const TopoDS_Shape& shape) {
    const TopoDS_Vertex vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(p.x, p.y, p.z));
    BRepExtrema_DistShapeShape query(vertex, shape);
    query.Perform();
    if (!query.IsDone() || query.NbSolution() == 0) {
        throw std::runtime_error("OpenCascade point-to-shape distance failed");
    }
    return query.Value();
}

void addSample(DistanceStatistics& result, double distance, double weight) {
    result.samples += 1;
    result.weight += weight;
    result.mean += weight * distance;
    result.rms += weight * distance * distance;
    result.maximum = std::max(result.maximum, distance);
}

void finish(DistanceStatistics& result) {
    if (result.weight <= 0.0) return;
    result.mean /= result.weight;
    result.rms = std::sqrt(result.rms / result.weight);
}

Vec3d toVec(const gp_Pnt& point) {
    return {point.X(), point.Y(), point.Z()};
}

TopoDS_Compound surfaceCompound(const TopoDS_Shape& shape) {
    TopoDS_Compound compound;
    BRep_Builder builder;
    builder.MakeCompound(compound);
    for (TopExp_Explorer explorer(shape, TopAbs_FACE); explorer.More(); explorer.Next()) {
        builder.Add(compound, explorer.Current());
    }
    return compound;
}

} // namespace

TriangleBvh::TriangleBvh(const Mesh& mesh) : mesh_(&mesh) {
    indices_.resize(mesh.triangles.size());
    std::iota(indices_.begin(), indices_.end(), 0u);
    nodes_.reserve(mesh.triangles.size() * 2);
    if (!indices_.empty()) build(0, static_cast<std::uint32_t>(indices_.size()));
}

std::uint32_t TriangleBvh::build(std::uint32_t begin, std::uint32_t end) {
    const std::uint32_t nodeIndex = static_cast<std::uint32_t>(nodes_.size());
    nodes_.push_back({});
    Aabb bounds;
    bounds.minimum = {std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity()};
    bounds.maximum = {-std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity()};
    Aabb centroidBounds = bounds;
    for (std::uint32_t i = begin; i < end; ++i) {
        const Triangle& triangle = mesh_->triangles[indices_[i]];
        const Vec3d& a = mesh_->vertices[triangle.vertices[0]];
        const Vec3d& b = mesh_->vertices[triangle.vertices[1]];
        const Vec3d& c = mesh_->vertices[triangle.vertices[2]];
        bounds.minimum = minimum(bounds.minimum, minimum(a, minimum(b, c)));
        bounds.maximum = maximum(bounds.maximum, maximum(a, maximum(b, c)));
        const Vec3d center = (a + b + c) / 3.0;
        centroidBounds.minimum = minimum(centroidBounds.minimum, center);
        centroidBounds.maximum = maximum(centroidBounds.maximum, center);
    }

    Node& node = nodes_[nodeIndex];
    node.bounds = bounds;
    node.begin = begin;
    node.count = end - begin;
    if (node.count <= 8) {
        node.leaf = true;
        return nodeIndex;
    }

    const Vec3d extent = centroidBounds.maximum - centroidBounds.minimum;
    const int axis = extent.y > extent.x ? (extent.z > extent.y ? 2 : 1)
                                          : (extent.z > extent.x ? 2 : 0);
    const std::uint32_t middle = begin + (end - begin) / 2;
    std::nth_element(indices_.begin() + begin, indices_.begin() + middle,
                     indices_.begin() + end, [&](std::uint32_t lhs, std::uint32_t rhs) {
        return component(mesh_->features[lhs].centroid, axis) <
               component(mesh_->features[rhs].centroid, axis);
    });
    const std::uint32_t left = build(begin, middle);
    const std::uint32_t right = build(middle, end);
    nodes_[nodeIndex].left = left;
    nodes_[nodeIndex].right = right;
    return nodeIndex;
}

double TriangleBvh::distanceSquared(
    std::uint32_t nodeIndex, const Vec3d& point, double best) const {
    const Node& node = nodes_[nodeIndex];
    if (pointAabbDistanceSquared(point, node.bounds) >= best) return best;
    if (node.leaf) {
        for (std::uint32_t i = node.begin; i < node.begin + node.count; ++i) {
            const Triangle& triangle = mesh_->triangles[indices_[i]];
            best = std::min(best, pointTriangleDistanceSquared(
                point, mesh_->vertices[triangle.vertices[0]],
                mesh_->vertices[triangle.vertices[1]],
                mesh_->vertices[triangle.vertices[2]]));
        }
        return best;
    }
    const double leftDistance = pointAabbDistanceSquared(point, nodes_[node.left].bounds);
    const double rightDistance = pointAabbDistanceSquared(point, nodes_[node.right].bounds);
    if (leftDistance < rightDistance) {
        best = distanceSquared(node.left, point, best);
        return distanceSquared(node.right, point, best);
    }
    best = distanceSquared(node.right, point, best);
    return distanceSquared(node.left, point, best);
}

double TriangleBvh::distance(const Vec3d& point) const {
    if (nodes_.empty()) return std::numeric_limits<double>::infinity();
    return std::sqrt(distanceSquared(0, point, std::numeric_limits<double>::infinity()));
}

DistanceStatistics meshToBrepDistance(
    const Mesh& mesh, const TopoDS_Shape& shape, bool includeVertexMaximum,
    std::size_t maximumSamples) {
    DistanceStatistics result;
    // Query a face-only compound. DistShapeShape reports zero for points in a
    // solid's interior, which is containment distance rather than the required
    // distance to the analytic B-Rep boundary.
    const TopoDS_Compound boundary = surfaceCompound(shape);
    const std::size_t triangleStride = maximumSamples > 0U
        ? std::max<std::size_t>(1U, (mesh.triangles.size() + maximumSamples - 1U) / maximumSamples)
        : 1U;
    for (std::size_t i = 0; i < mesh.triangles.size(); i += triangleStride) {
        if (mesh.features[i].degenerate || mesh.features[i].area <= 0.0) continue;
        const double distance = exactPointShapeDistance(mesh.features[i].centroid, boundary);
        addSample(result, distance, mesh.features[i].area);
    }
    if (includeVertexMaximum) {
        const std::size_t vertexStride = maximumSamples > 0U
            ? std::max<std::size_t>(1U, (mesh.vertices.size() + maximumSamples - 1U) / maximumSamples)
            : 1U;
        for (std::size_t i = 0; i < mesh.vertices.size(); i += vertexStride) {
            const Vec3d& vertex = mesh.vertices[i];
            result.maximum = std::max(result.maximum, exactPointShapeDistance(vertex, boundary));
        }
    }
    finish(result);
    return result;
}

DistanceStatistics brepToMeshDistance(
    const TopoDS_Shape& shape, const Mesh& mesh, double linearDeflection,
    double angularDeflection) {
    BRepMesh_IncrementalMesh mesher(shape, linearDeflection, false, angularDeflection, true);
    mesher.Perform();
    if (!mesher.IsDone()) throw std::runtime_error("OpenCascade B-Rep meshing failed");

    TriangleBvh bvh(mesh);
    DistanceStatistics result;
    for (TopExp_Explorer explorer(shape, TopAbs_FACE); explorer.More(); explorer.Next()) {
        const TopoDS_Face& face = TopoDS::Face(explorer.Current());
        TopLoc_Location location;
        const Handle(Poly_Triangulation) triangulation = BRep_Tool::Triangulation(face, location);
        if (triangulation.IsNull()) continue;
        const gp_Trsf transform = location.Transformation();
        const BRepAdaptor_Surface exactSurface(face, true);
        const bool hasUvNodes = triangulation->HasUVNodes();
        for (int i = 1; i <= triangulation->NbTriangles(); ++i) {
            int ia = 0, ib = 0, ic = 0;
            triangulation->Triangle(i).Get(ia, ib, ic);
            const Vec3d a = toVec(triangulation->Node(ia).Transformed(transform));
            const Vec3d b = toVec(triangulation->Node(ib).Transformed(transform));
            const Vec3d c = toVec(triangulation->Node(ic).Transformed(transform));
            const double area = 0.5 * norm(cross(b - a, c - a));
            if (area <= 0.0) continue;
            const double nodeWeight = area / (hasUvNodes ? 4.0 : 3.0);
            addSample(result, bvh.distance(a), nodeWeight);
            addSample(result, bvh.distance(b), nodeWeight);
            addSample(result, bvh.distance(c), nodeWeight);
            if (hasUvNodes) {
                const gp_Pnt2d uva = triangulation->UVNode(ia);
                const gp_Pnt2d uvb = triangulation->UVNode(ib);
                const gp_Pnt2d uvc = triangulation->UVNode(ic);
                const double u = (uva.X() + uvb.X() + uvc.X()) / 3.0;
                const double v = (uva.Y() + uvb.Y() + uvc.Y()) / 3.0;
                const gp_Pnt exactPoint = exactSurface.Value(u, v);
                addSample(result, bvh.distance(toVec(exactPoint)), area / 4.0);
            }
        }
    }
    finish(result);
    return result;
}

BidirectionalDistance compareMeshAndBrep(
    const Mesh& mesh, const TopoDS_Shape& shape, double linearDeflection,
    std::size_t maximumMeshSamples) {
    BidirectionalDistance result;
    result.meshToBrep = meshToBrepDistance(mesh, shape, true, maximumMeshSamples);
    result.brepToMesh = brepToMeshDistance(shape, mesh, linearDeflection);
    result.symmetricRms = std::sqrt(0.5 *
        (result.meshToBrep.rms * result.meshToBrep.rms +
         result.brepToMesh.rms * result.brepToMesh.rms));
    return result;
}

Aabb brepBounds(const TopoDS_Shape& shape) {
    Bnd_Box bounds;
    BRepBndLib::AddOptimal(shape, bounds, false, false);
    double xmin = 0.0, ymin = 0.0, zmin = 0.0;
    double xmax = 0.0, ymax = 0.0, zmax = 0.0;
    bounds.Get(xmin, ymin, zmin, xmax, ymax, zmax);
    return {{xmin, ymin, zmin}, {xmax, ymax, zmax}};
}

} // namespace recad
