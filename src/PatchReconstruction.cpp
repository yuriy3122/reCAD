#include "PatchReconstruction.h"

#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_Sewing.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <BRepLib.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <GC_MakeSegment2d.hxx>
#include <GC_MakeSegment.hxx>
#include <GeomAdaptor_Curve.hxx>
#include <GeomPlate_BuildPlateSurface.hxx>
#include <GeomPlate_CurveConstraint.hxx>
#include <GeomPlate_MakeApprox.hxx>
#include <GeomPlate_PointConstraint.hxx>
#include <GeomPlate_Surface.hxx>
#include <Geom2d_TrimmedCurve.hxx>
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <Geom_BSplineSurface.hxx>
#include <Geom_CylindricalSurface.hxx>
#include <Geom_ConicalSurface.hxx>
#include <Geom_Plane.hxx>
#include <Geom_SphericalSurface.hxx>
#include <Geom_Surface.hxx>
#include <Geom_ToroidalSurface.hxx>
#include <NCollection_Array1.hxx>
#include <NCollection_Array2.hxx>
#include <NCollection_IndexedMap.hxx>
#include <Precision.hxx>
#include <ShapeFix_Shape.hxx>
#include <ShapeFix_Solid.hxx>
#include <ShapeUpgrade_UnifySameDomain.hxx>
#include <Standard_Failure.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopTools_ShapeMapHasher.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shell.hxx>
#include <TopoDS_Solid.hxx>
#include <TopoDS_Vertex.hxx>
#include <TopoDS_Wire.hxx>
#include <gp_Ax3.hxx>
#include <gp_Dir.hxx>
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Vec.hxx>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace recad {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

using ShapeMap = NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher>;

struct EdgeKey {
    std::uint32_t low{};
    std::uint32_t high{};

    [[nodiscard]] bool operator<(const EdgeKey& other) const noexcept {
        return low < other.low || (low == other.low && high < other.high);
    }
};

struct DirectedEdge {
    std::uint32_t from{};
    std::uint32_t to{};
};

struct PatchMoments {
    Vec3d centroid{};
    Vec3d normal{};
    double area{};
};

struct ProjectedLoop {
    std::vector<std::uint32_t> vertexIndices;
    std::vector<gp_Pnt2d> parameters;
    double signedArea{};
    double meanU{};
    double meanV{};
    double minimumU{};
    double maximumU{};
    double minimumV{};
    double maximumV{};
    long long uWinding{};
    long long vWinding{};
};

[[nodiscard]] bool finitePoint(const Vec3d& point) noexcept {
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

[[nodiscard]] double dotProduct(const Vec3d& left, const Vec3d& right) noexcept {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

[[nodiscard]] Vec3d crossProduct(const Vec3d& left, const Vec3d& right) noexcept {
    return {left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x};
}

[[nodiscard]] double vectorLength(const Vec3d& value) noexcept {
    return std::sqrt(dotProduct(value, value));
}

[[nodiscard]] Vec3d normalizedVector(const Vec3d& value) noexcept {
    const double length = vectorLength(value);
    return std::isfinite(length) && length > std::numeric_limits<double>::min()
        ? value / length
        : Vec3d{};
}

[[nodiscard]] Vec3d stablePerpendicular(const Vec3d& direction) {
    const Vec3d unit = normalizedVector(direction);
    const std::array<Vec3d, 3> axes{{{1.0, 0.0, 0.0},
                                     {0.0, 1.0, 0.0},
                                     {0.0, 0.0, 1.0}}};
    std::size_t best = 0U;
    for (std::size_t i = 1U; i < axes.size(); ++i) {
        if (std::abs(dotProduct(unit, axes[i])) <
            std::abs(dotProduct(unit, axes[best]))) {
            best = i;
        }
    }
    return normalizedVector(crossProduct(unit, axes[best]));
}

[[nodiscard]] gp_Pnt toPoint(const Vec3d& point) {
    return {point.x, point.y, point.z};
}

[[nodiscard]] gp_Dir toDirection(const Vec3d& direction) {
    const Vec3d unit = normalizedVector(direction);
    if (!(vectorLength(unit) > 0.0)) {
        throw std::invalid_argument("An analytic surface direction has zero length");
    }
    return {unit.x, unit.y, unit.z};
}

[[nodiscard]] double meshScale(const Mesh& mesh) {
    if (mesh.bounds.valid()) {
        const double diagonal = mesh.bounds.diagonal();
        if (std::isfinite(diagonal) && diagonal > 0.0) return diagonal;
    }
    Aabb bounds;
    for (const Vec3d& point : mesh.vertices) {
        if (finitePoint(point)) bounds.expand(point);
    }
    return bounds.valid() ? bounds.diagonal() : 0.0;
}

[[nodiscard]] double estimatedBoundaryChordSag(
    const Mesh& mesh,
    const std::vector<SurfacePatchFit>& patches) {
    double maximumSag = 0.0;
    for (const SurfacePatchFit& patch : patches) {
        if (!patch.accepted) continue;
        std::map<EdgeKey, std::size_t> incidence;
        for (const std::uint32_t triangleIndex : patch.triangleIndices) {
            if (triangleIndex >= mesh.triangles.size()) continue;
            const Triangle& triangle = mesh.triangles[triangleIndex];
            for (std::size_t edge = 0; edge < 3U; ++edge) {
                const auto first = triangle[edge];
                const auto second = triangle[(edge + 1U) % 3U];
                ++incidence[{std::min(first, second), std::max(first, second)}];
            }
        }
        // Only shared patch boundaries need a sewing allowance. Internal
        // diagonals and a torus's small tube radius can wildly overestimate it.
        for (const auto& [edge, count] : incidence) {
            if (count != 1U) continue;
            const Vec3d midpoint = (mesh.vertices.at(edge.low) +
                                   mesh.vertices.at(edge.high)) * 0.5;
            double sag = 0.0;
            if (const auto* sphere = std::get_if<SphereSurface>(&patch.parameters)) {
                sag = std::abs(norm(midpoint - sphere->center) - sphere->radius);
            } else if (const auto* cylinder = std::get_if<CylinderSurface>(&patch.parameters)) {
                const Vec3d d = midpoint - cylinder->axisPoint;
                sag = std::abs(norm(d - cylinder->axisDirection *
                    dot(d, cylinder->axisDirection)) - cylinder->radius);
            } else if (const auto* torus = std::get_if<TorusSurface>(&patch.parameters)) {
                const Vec3d d = midpoint - torus->center;
                const double axial = dot(d, torus->axisDirection);
                const double rho = norm(d - torus->axisDirection * axial);
                sag = std::abs(std::hypot(rho - torus->majorRadius, axial) -
                               torus->minorRadius);
            }
            if (std::isfinite(sag)) maximumSag = std::max(maximumSag, sag);
        }
    }
    return maximumSag;
}

[[nodiscard]] double resolveSewingTolerance(
    const Mesh& mesh,
    const std::vector<SurfacePatchFit>& patches,
    double requested) {
    if (!std::isfinite(requested) || requested < 0.0) {
        throw std::invalid_argument(
            "Patch reconstruction sewing tolerance must be finite and non-negative");
    }
    if (requested > 0.0) return requested;

    double fitResidual = 0.0;
    for (const SurfacePatchFit& patch : patches) {
        if (patch.accepted && std::isfinite(patch.residuals.positionMax)) {
            fitResidual = std::max(fitResidual, patch.residuals.positionMax);
        }
    }
    const double scale = meshScale(mesh);
    return std::max({Precision::Confusion(),
                     mesh.weldTolerance * 4.0,
                     scale > 0.0 ? scale * 1.0e-7 : 0.0,
                     fitResidual * 2.0,
                     estimatedBoundaryChordSag(mesh, patches) * 4.0});
}

[[nodiscard]] std::size_t countSubshapes(
    const TopoDS_Shape& shape,
    TopAbs_ShapeEnum kind) {
    if (shape.IsNull()) return 0U;
    ShapeMap mapped;
    TopExp::MapShapes(shape, kind, mapped);
    return static_cast<std::size_t>(mapped.Extent());
}

[[nodiscard]] std::size_t countClosedShells(const TopoDS_Shape& shape) {
    if (shape.IsNull()) return 0U;
    ShapeMap shells;
    TopExp::MapShapes(shape, TopAbs_SHELL, shells);
    std::size_t result = 0U;
    for (int index = 1; index <= shells.Extent(); ++index) {
        if (BRep_Tool::IsClosed(shells(index))) ++result;
    }
    return result;
}

[[nodiscard]] TopoDS_Shape assembleShapes(const std::vector<TopoDS_Shape>& shapes) {
    if (shapes.empty()) return {};
    if (shapes.size() == 1U) return shapes.front();
    BRep_Builder builder;
    TopoDS_Compound compound;
    builder.MakeCompound(compound);
    for (const TopoDS_Shape& shape : shapes) builder.Add(compound, shape);
    return compound;
}

[[nodiscard]] EdgeKey edgeKey(std::uint32_t first, std::uint32_t second) {
    return {std::min(first, second), std::max(first, second)};
}

[[nodiscard]] PatchMoments patchMoments(
    const Mesh& mesh,
    const SurfacePatchFit& patch) {
    PatchMoments result;
    for (const std::uint32_t triangleIndex : patch.triangleIndices) {
        if (triangleIndex >= mesh.triangles.size()) continue;
        const Triangle& triangle = mesh.triangles[triangleIndex];
        if (triangle[0] >= mesh.vertices.size() || triangle[1] >= mesh.vertices.size() ||
            triangle[2] >= mesh.vertices.size()) {
            continue;
        }
        const Vec3d& first = mesh.vertices[triangle[0]];
        const Vec3d& second = mesh.vertices[triangle[1]];
        const Vec3d& third = mesh.vertices[triangle[2]];
        const Vec3d twiceAreaVector = crossProduct(second - first, third - first);
        const double twiceArea = vectorLength(twiceAreaVector);
        if (!std::isfinite(twiceArea) || !(twiceArea > 0.0)) continue;
        const double area = 0.5 * twiceArea;
        result.area += area;
        result.centroid += (first + second + third) * (area / 3.0);
        result.normal += twiceAreaVector * 0.5;
    }
    if (result.area > 0.0) result.centroid /= result.area;
    result.normal = normalizedVector(result.normal);
    return result;
}

[[nodiscard]] bool extractBoundaryLoops(
    const Mesh& mesh,
    const SurfacePatchFit& patch,
    std::vector<std::vector<std::uint32_t>>& loops,
    std::string& error) {
    std::map<EdgeKey, std::vector<DirectedEdge>> edgeUses;
    std::set<std::uint32_t> uniqueTriangles;
    for (const std::uint32_t triangleIndex : patch.triangleIndices) {
        if (!uniqueTriangles.insert(triangleIndex).second) {
            error = "the patch repeats a triangle index";
            return false;
        }
        if (triangleIndex >= mesh.triangles.size()) {
            error = "the patch contains an out-of-range triangle index";
            return false;
        }
        const Triangle& triangle = mesh.triangles[triangleIndex];
        for (const std::uint32_t vertex : triangle.vertices) {
            if (vertex >= mesh.vertices.size()) {
                error = "a patch triangle contains an out-of-range vertex index";
                return false;
            }
        }
        for (std::size_t edge = 0; edge < 3U; ++edge) {
            const std::uint32_t from = triangle[edge];
            const std::uint32_t to = triangle[(edge + 1U) % 3U];
            if (from == to) {
                error = "a patch triangle contains a zero-length topological edge";
                return false;
            }
            edgeUses[edgeKey(from, to)].push_back({from, to});
        }
    }

    std::vector<DirectedEdge> boundaryEdges;
    for (const auto& [key, uses] : edgeUses) {
        static_cast<void>(key);
        if (uses.size() == 1U) {
            boundaryEdges.push_back(uses.front());
        } else if (uses.size() == 2U) {
            if (uses[0].from != uses[1].to || uses[0].to != uses[1].from) {
                error = "the patch has inconsistent triangle winding";
                return false;
            }
        } else {
            error = "the patch has a non-manifold edge";
            return false;
        }
    }
    if (boundaryEdges.empty()) return true;

    std::sort(boundaryEdges.begin(), boundaryEdges.end(),
              [](const DirectedEdge& left, const DirectedEdge& right) {
                  return left.from < right.from ||
                      (left.from == right.from && left.to < right.to);
              });
    std::map<std::uint32_t, std::uint32_t> successor;
    std::map<std::uint32_t, std::size_t> predecessorCount;
    for (const DirectedEdge edge : boundaryEdges) {
        if (!successor.emplace(edge.from, edge.to).second) {
            error = "a patch boundary branches at a vertex";
            return false;
        }
        ++predecessorCount[edge.to];
    }
    for (const auto& [from, to] : successor) {
        static_cast<void>(to);
        if (predecessorCount[from] != 1U) {
            error = "a patch boundary is open or branches at a vertex";
            return false;
        }
    }

    std::set<std::uint32_t> visited;
    for (const auto& [start, ignored] : successor) {
        static_cast<void>(ignored);
        if (visited.contains(start)) continue;
        std::vector<std::uint32_t> loop;
        std::uint32_t current = start;
        while (true) {
            if (current == start && !loop.empty()) break;
            if (!visited.insert(current).second) {
                error = "two patch boundary loops touch or merge";
                return false;
            }
            loop.push_back(current);
            const auto next = successor.find(current);
            if (next == successor.end()) {
                error = "a patch boundary loop is open";
                return false;
            }
            current = next->second;
            if (loop.size() > boundaryEdges.size()) {
                error = "a patch boundary loop did not close";
                return false;
            }
        }
        if (loop.size() < 3U) {
            error = "a patch boundary loop has fewer than three vertices";
            return false;
        }
        loops.push_back(std::move(loop));
    }
    if (visited.size() != boundaryEdges.size()) {
        error = "not every patch boundary edge belongs to a closed loop";
        return false;
    }
    return true;
}

[[nodiscard]] Handle(Geom_Surface) makeAnalyticSurface(
    const SurfacePatchFit& patch,
    const PatchMoments& moments) {
    switch (patch.kind) {
    case SurfaceKind::Plane: {
        const Plane* plane = std::get_if<Plane>(&patch.parameters);
        if (plane == nullptr || !finitePoint(plane->normal) ||
            !std::isfinite(plane->offset)) {
            throw std::invalid_argument("plane parameters are invalid or mismatched");
        }
        const double normalSquared = dotProduct(plane->normal, plane->normal);
        if (!(normalSquared > 0.0)) throw std::invalid_argument("plane normal is zero");
        const Vec3d normal = normalizedVector(plane->normal);
        const Vec3d origin = plane->normal * (-plane->offset / normalSquared);
        return new Geom_Plane(gp_Ax3(toPoint(origin),
                                     toDirection(normal),
                                     toDirection(stablePerpendicular(normal))));
    }
    case SurfaceKind::Cylinder: {
        const CylinderSurface* cylinder = std::get_if<CylinderSurface>(&patch.parameters);
        if (cylinder == nullptr || !finitePoint(cylinder->axisPoint) ||
            !finitePoint(cylinder->axisDirection) ||
            !std::isfinite(cylinder->radius) || !(cylinder->radius > 0.0)) {
            throw std::invalid_argument("cylinder parameters are invalid or mismatched");
        }
        const Vec3d axis = normalizedVector(cylinder->axisDirection);
        Vec3d radial = moments.centroid - cylinder->axisPoint;
        radial -= axis * dotProduct(radial, axis);
        radial = normalizedVector(radial);
        if (!(vectorLength(radial) > 0.0)) radial = stablePerpendicular(axis);
        // Put the U seam opposite the patch's weighted center.
        radial *= -1.0;
        return new Geom_CylindricalSurface(
            gp_Ax3(toPoint(cylinder->axisPoint), toDirection(axis), toDirection(radial)),
            cylinder->radius);
    }
    case SurfaceKind::Cone: {
        const auto* cone = std::get_if<ConeSurface>(&patch.parameters);
        if (cone == nullptr || !finitePoint(cone->apex) || !finitePoint(cone->axisDirection) ||
            !(cone->semiAngle > 0.0 && cone->semiAngle < 1.5707963267948966)) {
            throw std::invalid_argument("cone parameters are invalid or mismatched");
        }
        const Vec3d axis = normalizedVector(cone->axisDirection);
        Vec3d radial = moments.centroid - cone->apex;
        radial -= axis * dotProduct(radial, axis);
        if (vectorLength(radial) == 0.0) radial = stablePerpendicular(axis);
        return new Geom_ConicalSurface(gp_Ax3(toPoint(cone->apex), toDirection(axis),
                                             toDirection(radial * -1.0)), cone->semiAngle, 0.0);
    }
    case SurfaceKind::Sphere: {
        const SphereSurface* sphere = std::get_if<SphereSurface>(&patch.parameters);
        if (sphere == nullptr || !finitePoint(sphere->center) ||
            !std::isfinite(sphere->radius) || !(sphere->radius > 0.0)) {
            throw std::invalid_argument("sphere parameters are invalid or mismatched");
        }
        Vec3d radial = normalizedVector(moments.centroid - sphere->center);
        if (!(vectorLength(radial) > 0.0)) radial = {1.0, 0.0, 0.0};
        // Put chart geometry near the equator, with the seam on the opposite
        // side. Wide spherical regions are divided into bounded charts below.
        const Vec3d zDirection = stablePerpendicular(radial);
        const Vec3d xDirection = radial * -1.0;
        return new Geom_SphericalSurface(
            gp_Ax3(toPoint(sphere->center),
                   toDirection(zDirection),
                   toDirection(xDirection)),
            sphere->radius);
    }
    case SurfaceKind::Torus: {
        const TorusSurface* torus = std::get_if<TorusSurface>(&patch.parameters);
        if (torus == nullptr || !finitePoint(torus->center) ||
            !finitePoint(torus->axisDirection) ||
            !std::isfinite(torus->majorRadius) || !(torus->majorRadius > 0.0) ||
            !std::isfinite(torus->minorRadius) || !(torus->minorRadius > 0.0)) {
            throw std::invalid_argument("torus parameters are invalid or mismatched");
        }
        const Vec3d axis = normalizedVector(torus->axisDirection);
        Vec3d radial = moments.centroid - torus->center;
        radial -= axis * dotProduct(radial, axis);
        radial = normalizedVector(radial);
        if (!(vectorLength(radial) > 0.0)) radial = stablePerpendicular(axis);
        radial *= -1.0;
        return new Geom_ToroidalSurface(
            gp_Ax3(toPoint(torus->center), toDirection(axis), toDirection(radial)),
            torus->majorRadius,
            torus->minorRadius);
    }
    default:
        throw std::invalid_argument("the patch has no supported analytic surface kind");
    }
}

[[nodiscard]] double unwrapNear(double value, double reference, double period) {
    if (!(period > 0.0) || !std::isfinite(period)) return value;
    return value + std::round((reference - value) / period) * period;
}

void updateLoopStatistics(ProjectedLoop& loop,
                          const Handle(Geom_Surface)& surface) {
    loop.minimumU = std::numeric_limits<double>::infinity();
    loop.maximumU = -std::numeric_limits<double>::infinity();
    loop.minimumV = std::numeric_limits<double>::infinity();
    loop.maximumV = -std::numeric_limits<double>::infinity();
    loop.meanU = 0.0;
    loop.meanV = 0.0;
    for (const gp_Pnt2d& parameter : loop.parameters) {
        loop.minimumU = std::min(loop.minimumU, parameter.X());
        loop.maximumU = std::max(loop.maximumU, parameter.X());
        loop.minimumV = std::min(loop.minimumV, parameter.Y());
        loop.maximumV = std::max(loop.maximumV, parameter.Y());
        loop.meanU += parameter.X();
        loop.meanV += parameter.Y();
    }
    loop.meanU /= static_cast<double>(loop.parameters.size());
    loop.meanV /= static_cast<double>(loop.parameters.size());

    gp_Pnt2d closing = loop.parameters.front();
    if (surface->IsUPeriodic()) {
        closing.SetX(unwrapNear(closing.X(),
                                loop.parameters.back().X(),
                                surface->UPeriod()));
        loop.uWinding = std::llround(
            (closing.X() - loop.parameters.front().X()) / surface->UPeriod());
    }
    if (surface->IsVPeriodic()) {
        closing.SetY(unwrapNear(closing.Y(),
                                loop.parameters.back().Y(),
                                surface->VPeriod()));
        loop.vWinding = std::llround(
            (closing.Y() - loop.parameters.front().Y()) / surface->VPeriod());
    }
    loop.signedArea = 0.0;
    for (std::size_t index = 0; index < loop.parameters.size(); ++index) {
        const gp_Pnt2d& first = loop.parameters[index];
        const gp_Pnt2d& second = index + 1U < loop.parameters.size()
            ? loop.parameters[index + 1U]
            : closing;
        loop.signedArea += first.X() * second.Y() - second.X() * first.Y();
    }
    loop.signedArea *= 0.5;
}

[[nodiscard]] ProjectedLoop projectBoundaryLoop(
    const Mesh& mesh,
    const std::vector<std::uint32_t>& vertexIndices,
    const Handle(Geom_Surface)& surface,
    double maximumDistance,
    double& observedMaximumDistance) {
    ProjectedLoop result;
    result.vertexIndices = vertexIndices;
    result.parameters.reserve(vertexIndices.size());
    for (const std::uint32_t vertexIndex : vertexIndices) {
        const gp_Pnt point = toPoint(mesh.vertices[vertexIndex]);
        GeomAPI_ProjectPointOnSurf projector(point, surface, Precision::Confusion());
        if (!projector.IsDone() || projector.NbPoints() == 0) {
            throw std::runtime_error("OpenCascade could not project a boundary vertex");
        }
        const double distance = projector.LowerDistance();
        if (!std::isfinite(distance) || distance > maximumDistance) {
            std::ostringstream message;
            message << "boundary projection distance " << distance
                    << " exceeds limit " << maximumDistance;
            throw std::runtime_error(message.str());
        }
        observedMaximumDistance = std::max(observedMaximumDistance, distance);
        double u{};
        double v{};
        projector.LowerDistanceParameters(u, v);
        if (!result.parameters.empty()) {
            if (surface->IsUPeriodic()) {
                u = unwrapNear(u, result.parameters.back().X(), surface->UPeriod());
            }
            if (surface->IsVPeriodic()) {
                v = unwrapNear(v, result.parameters.back().Y(), surface->VPeriod());
            }
        }
        result.parameters.emplace_back(u, v);
    }
    updateLoopStatistics(result, surface);
    return result;
}

void alignLoopToReference(ProjectedLoop& loop,
                          double referenceU,
                          double referenceV,
                          const Handle(Geom_Surface)& surface) {
    double shiftU = 0.0;
    double shiftV = 0.0;
    if (surface->IsUPeriodic()) {
        shiftU = std::round((referenceU - loop.meanU) / surface->UPeriod()) *
            surface->UPeriod();
    }
    if (surface->IsVPeriodic()) {
        shiftV = std::round((referenceV - loop.meanV) / surface->VPeriod()) *
            surface->VPeriod();
    }
    if (shiftU == 0.0 && shiftV == 0.0) return;
    for (gp_Pnt2d& parameter : loop.parameters) {
        parameter.SetX(parameter.X() + shiftU);
        parameter.SetY(parameter.Y() + shiftV);
    }
    updateLoopStatistics(loop, surface);
}

[[nodiscard]] Handle(Geom2d_TrimmedCurve) parameterSegment(
    const gp_Pnt2d& first,
    const gp_Pnt2d& last) {
    const GC_MakeSegment2d builder(first, last);
    if (!builder.IsDone()) {
        throw std::runtime_error("OpenCascade could not construct a boundary pcurve");
    }
    return builder.Value();
}

[[nodiscard]] TopoDS_Edge makeSurfaceEdge(
    const Handle(Geom_Surface)& surface,
    const gp_Pnt2d& firstParameter,
    const gp_Pnt2d& lastParameter,
    const TopoDS_Vertex& firstVertex,
    const TopoDS_Vertex& lastVertex) {
    const Handle(Geom2d_TrimmedCurve) curve =
        parameterSegment(firstParameter, lastParameter);
    BRepBuilderAPI_MakeEdge builder(curve,
                                   surface,
                                   firstVertex,
                                   lastVertex,
                                   curve->FirstParameter(),
                                   curve->LastParameter());
    if (!builder.IsDone()) {
        throw std::runtime_error("OpenCascade could not construct an analytic boundary edge");
    }
    return builder.Edge();
}

[[nodiscard]] gp_Pnt2d closingParameter(
    const ProjectedLoop& loop,
    const Handle(Geom_Surface)& surface) {
    gp_Pnt2d result = loop.parameters.front();
    if (surface->IsUPeriodic()) {
        result.SetX(unwrapNear(result.X(),
                               loop.parameters.back().X(),
                               surface->UPeriod()));
    }
    if (surface->IsVPeriodic()) {
        result.SetY(unwrapNear(result.Y(),
                               loop.parameters.back().Y(),
                               surface->VPeriod()));
    }
    return result;
}

[[nodiscard]] TopoDS_Wire makeAnalyticWire(
    const ProjectedLoop& loop,
    const Handle(Geom_Surface)& surface,
    double tolerance) {
    std::vector<TopoDS_Vertex> vertices;
    vertices.reserve(loop.parameters.size());
    BRep_Builder topologyBuilder;
    for (const gp_Pnt2d& parameter : loop.parameters) {
        const TopoDS_Vertex vertex =
            BRepBuilderAPI_MakeVertex(surface->Value(parameter.X(), parameter.Y())).Vertex();
        topologyBuilder.UpdateVertex(vertex, tolerance);
        vertices.push_back(vertex);
    }

    BRepBuilderAPI_MakeWire wireBuilder;
    for (std::size_t index = 0; index < loop.parameters.size(); ++index) {
        const std::size_t next = (index + 1U) % loop.parameters.size();
        const gp_Pnt2d lastParameter = next == 0U
            ? closingParameter(loop, surface)
            : loop.parameters[next];
        wireBuilder.Add(makeSurfaceEdge(surface,
                                        loop.parameters[index],
                                        lastParameter,
                                        vertices[index],
                                        vertices[next]));
    }
    if (!wireBuilder.IsDone()) {
        throw std::runtime_error("OpenCascade could not close an analytic boundary wire");
    }
    return wireBuilder.Wire();
}

[[nodiscard]] bool pointInPolygon(const gp_Pnt2d& point,
                                  const ProjectedLoop& polygon) {
    bool inside = false;
    const std::size_t count = polygon.parameters.size();
    for (std::size_t i = 0U, j = count - 1U; i < count; j = i++) {
        const gp_Pnt2d& first = polygon.parameters[i];
        const gp_Pnt2d& second = polygon.parameters[j];
        const bool crosses = (first.Y() > point.Y()) != (second.Y() > point.Y());
        if (!crosses) continue;
        const double intersectionX = (second.X() - first.X()) *
            (point.Y() - first.Y()) / (second.Y() - first.Y()) + first.X();
        if (point.X() < intersectionX) inside = !inside;
    }
    return inside;
}

void orientFaceToPatch(TopoDS_Face& face,
                       const Handle(Geom_Surface)& surface,
                       const gp_Pnt2d& parameter,
                       const Vec3d& patchNormal) {
    if (!(vectorLength(patchNormal) > 0.0)) return;
    gp_Pnt point;
    gp_Vec derivativeU;
    gp_Vec derivativeV;
    surface->D1(parameter.X(), parameter.Y(), point, derivativeU, derivativeV);
    const gp_Vec normal = derivativeU.Crossed(derivativeV);
    const double alignment = normal.X() * patchNormal.x +
        normal.Y() * patchNormal.y + normal.Z() * patchNormal.z;
    if (alignment < 0.0) face.Reverse();
}

// Cut a periodic annulus into two ordinary UV charts. Both boundary curves
// retain their measured height variation; they need not be coaxial circles.
[[nodiscard]] bool makePeriodicBandFaces(
    const Handle(Geom_Surface)& surface,
    const std::vector<ProjectedLoop>& loops,
    double tolerance,
    std::vector<TopoDS_Face>& faces) {
    if (loops.size() != 2U) return false;
    const bool alongU = surface->IsUPeriodic() &&
        std::abs(loops[0].uWinding) == 1 && std::abs(loops[1].uWinding) == 1;
    const bool alongV = surface->IsVPeriodic() &&
        std::abs(loops[0].vWinding) == 1 && std::abs(loops[1].vWinding) == 1;
    if (!alongU && !alongV) return false;
    const double period = alongU ? surface->UPeriod() : surface->VPeriod();
    std::array<std::vector<gp_Pnt2d>, 2> rings;
    for (unsigned ring = 0; ring < 2; ++ring) {
        auto parameters = loops[ring].parameters;
        const auto winding = alongU ? loops[ring].uWinding : loops[ring].vWinding;
        if (winding < 0) std::reverse(parameters.begin(), parameters.end());
        double previous = -std::numeric_limits<double>::infinity();
        for (const auto& parameter : parameters) {
            double u = alongU ? parameter.X() : parameter.Y();
            const double v = alongU ? parameter.Y() : parameter.X();
            if (std::isfinite(previous)) u = unwrapNear(u, previous, period);
            if (u < previous - 1.0e-8) {
                throw std::runtime_error("Periodic boundary doubles back across a chart meridian");
            }
            previous = u;
            u -= std::floor(u / period) * period;
            rings[ring].emplace_back(u, v);
        }
        std::sort(rings[ring].begin(), rings[ring].end(),
            [](const gp_Pnt2d& a, const gp_Pnt2d& b) { return a.X() < b.X(); });
    }
    const auto heightAt = [&](const std::vector<gp_Pnt2d>& ring, double u) {
        for (std::size_t i = 0; i < ring.size(); ++i) {
            const auto& a = ring[i];
            gp_Pnt2d b = ring[(i + 1) % ring.size()];
            if (i + 1 == ring.size()) b.SetX(b.X() + period);
            double query = u;
            if (query < ring.front().X()) query += period;
            if (query >= a.X() - 1.0e-10 && query <= b.X() + 1.0e-10) {
                const double width = b.X() - a.X();
                if (width <= 1.0e-12) return 0.5 * (a.Y() + b.Y());
                return a.Y() + (b.Y() - a.Y()) * (query - a.X()) / width;
            }
        }
        throw std::runtime_error("Periodic boundary does not cover the chart meridian");
    };
    if (heightAt(rings[0], 0.0) > heightAt(rings[1], 0.0)) std::swap(rings[0], rings[1]);
    for (unsigned chart = 0; chart < 2; ++chart) {
        const double start = period * chart / 2.0;
        const double end = period * (chart + 1) / 2.0;
        ProjectedLoop polygon;
        for (unsigned ring = 0; ring < 2; ++ring) {
            std::vector<gp_Pnt2d> arc;
            arc.emplace_back(start, heightAt(rings[ring], start));
            for (const auto& p : rings[ring]) {
                if (p.X() > start + 1.0e-10 && p.X() < end - 1.0e-10) arc.push_back(p);
            }
            arc.emplace_back(end, heightAt(rings[ring], end == period ? 0.0 : end));
            if (ring == 1) std::reverse(arc.begin(), arc.end());
            for (const auto& p : arc) {
                polygon.parameters.push_back(alongU ? p : gp_Pnt2d(p.Y(), p.X()));
            }
        }
        const TopoDS_Wire wire = makeAnalyticWire(polygon, surface, tolerance);
        BRepBuilderAPI_MakeFace builder(surface, wire, true);
        if (!builder.IsDone()) throw std::runtime_error("Could not trim periodic surface chart");
        const TopoDS_Face face = builder.Face();
        if (!BRepLib::BuildCurves3d(face, tolerance) || !BRepCheck_Analyzer(face, true).IsValid()) {
            throw std::runtime_error("Invalid periodic surface chart");
        }
        faces.push_back(face);
    }
    return true;
}

[[nodiscard]] std::vector<TopoDS_Face> makeAnalyticPatchFaces(
    const Mesh& mesh,
    const SurfacePatchFit& patch,
    double sewingTolerance,
    double maximumProjectionDistance,
    double& observedProjectionDistance,
    const Handle(Geom_Surface)& fittedSurface = {}) {
    const PatchMoments moments = patchMoments(mesh, patch);
    if (!(moments.area > 0.0)) {
        throw std::runtime_error("patch has no non-degenerate surface area");
    }
    const Handle(Geom_Surface) surface = fittedSurface.IsNull()
        ? makeAnalyticSurface(patch, moments) : fittedSurface;
    std::vector<std::vector<std::uint32_t>> boundaryLoops;
    std::string boundaryError;
    if (!extractBoundaryLoops(mesh, patch, boundaryLoops, boundaryError)) {
        throw std::runtime_error(boundaryError);
    }

    std::vector<TopoDS_Face> faces;
    if (boundaryLoops.empty()) {
        if (patch.kind != SurfaceKind::Sphere && patch.kind != SurfaceKind::Torus) {
            throw std::runtime_error(
                "an unbounded analytic surface patch has no boundary loop");
        }
        BRepBuilderAPI_MakeFace faceBuilder(surface, sewingTolerance);
        if (!faceBuilder.IsDone()) {
            throw std::runtime_error("OpenCascade could not build the closed analytic face");
        }
        TopoDS_Face face = faceBuilder.Face();
        double uMinimum{};
        double uMaximum{};
        double vMinimum{};
        double vMaximum{};
        surface->Bounds(uMinimum, uMaximum, vMinimum, vMaximum);
        orientFaceToPatch(face,
                          surface,
                          gp_Pnt2d(0.5 * (uMinimum + uMaximum),
                                   0.5 * (vMinimum + vMaximum)),
                          moments.normal);
        faces.push_back(face);
        return faces;
    }

    std::vector<ProjectedLoop> projected;
    projected.reserve(boundaryLoops.size());
    for (const std::vector<std::uint32_t>& loop : boundaryLoops) {
        projected.push_back(projectBoundaryLoop(mesh,
                                                loop,
                                                surface,
                                                maximumProjectionDistance,
                                                observedProjectionDistance));
    }
    const auto largestLoop = std::max_element(
        projected.begin(), projected.end(),
        [](const ProjectedLoop& left, const ProjectedLoop& right) {
            return std::abs(left.signedArea) < std::abs(right.signedArea);
        });
    const double referenceU = largestLoop->meanU;
    const double referenceV = largestLoop->meanV;
    for (ProjectedLoop& loop : projected) {
        alignLoopToReference(loop, referenceU, referenceV, surface);
    }

    // A sphere with one latitude boundary is a cap, including the common
    // complementary cap larger than a hemisphere. Select the pole from the
    // observed patch, rather than letting a UV polygon select the complement.
    if (patch.kind == SurfaceKind::Sphere && projected.size() == 1U &&
        std::abs(projected.front().uWinding) == 1) {
        const ProjectedLoop& loop = projected.front();
        const auto spherical = Handle(Geom_SphericalSurface)::DownCast(surface);
        const double latitudeSpread = loop.maximumV - loop.minimumV;
        const double latitudeTolerance = std::max(
            1.0e-6, maximumProjectionDistance / spherical->Radius());
        if (latitudeSpread <= latitudeTolerance * 4.0) {
            GeomAPI_ProjectPointOnSurf interior(toPoint(moments.centroid), surface);
            double interiorU = 0.0;
            double interiorV = 0.0;
            interior.LowerDistanceParameters(interiorU, interiorV);
            const double lowerV = interiorV < loop.meanV ? -kPi * 0.5 : loop.meanV;
            const double upperV = interiorV < loop.meanV ? loop.meanV : kPi * 0.5;
            BRepBuilderAPI_MakeFace cap(surface, 0.0, 2.0 * kPi,
                                         lowerV, upperV, sewingTolerance);
            if (cap.IsDone()) {
                TopoDS_Face face = cap.Face();
                orientFaceToPatch(face, surface, loop.parameters.front(), moments.normal);
                if (BRepCheck_Analyzer(face, true).IsValid()) {
                    faces.push_back(face);
                    return faces;
                }
            }
        }
    }
    if (makePeriodicBandFaces(surface, projected, sewingTolerance, faces)) {
        return faces;
    }

    for (const ProjectedLoop& loop : projected) {
        if (std::abs(loop.signedArea) <= std::numeric_limits<double>::epsilon()) {
            throw std::runtime_error(
                "a projected boundary loop has zero parameter-space area");
        }
    }

    std::vector<TopoDS_Wire> wires;
    wires.reserve(projected.size());
    for (const ProjectedLoop& loop : projected) {
        wires.push_back(makeAnalyticWire(loop, surface, sewingTolerance));
    }

    // A source face normally has one outer loop and zero or more holes. The
    // containment tree also handles the uncommon case of disconnected islands
    // or nested loops without relying on loop winding conventions.
    std::vector<int> parent(projected.size(), -1);
    for (std::size_t child = 0; child < projected.size(); ++child) {
        const gp_Pnt2d representative(projected[child].meanU, projected[child].meanV);
        double smallestContainerArea = std::numeric_limits<double>::infinity();
        for (std::size_t candidate = 0; candidate < projected.size(); ++candidate) {
            if (candidate == child ||
                std::abs(projected[candidate].signedArea) <=
                    std::abs(projected[child].signedArea)) {
                continue;
            }
            if (pointInPolygon(representative, projected[candidate]) &&
                std::abs(projected[candidate].signedArea) < smallestContainerArea) {
                parent[child] = static_cast<int>(candidate);
                smallestContainerArea = std::abs(projected[candidate].signedArea);
            }
        }
    }
    std::vector<int> depth(projected.size(), 0);
    for (std::size_t index = 0; index < projected.size(); ++index) {
        int current = parent[index];
        while (current >= 0) {
            ++depth[index];
            if (depth[index] > static_cast<int>(projected.size())) {
                throw std::runtime_error("projected boundary containment is cyclic");
            }
            current = parent[static_cast<std::size_t>(current)];
        }
    }

    BRep_Builder topologyBuilder;
    for (std::size_t outer = 0; outer < projected.size(); ++outer) {
        if (depth[outer] % 2 != 0) continue;
        BRepBuilderAPI_MakeFace faceBuilder(surface, wires[outer], true);
        if (!faceBuilder.IsDone()) {
            throw std::runtime_error("OpenCascade could not trim an analytic outer wire");
        }
        for (std::size_t hole = 0; hole < projected.size(); ++hole) {
            if (parent[hole] != static_cast<int>(outer) || depth[hole] != depth[outer] + 1) {
                continue;
            }
            TopoDS_Wire holeWire = wires[hole];
            if (std::signbit(projected[hole].signedArea) ==
                std::signbit(projected[outer].signedArea)) {
                holeWire.Reverse();
            }
            faceBuilder.Add(holeWire);
        }
        if (!faceBuilder.IsDone()) {
            throw std::runtime_error("OpenCascade could not add analytic hole wires");
        }
        TopoDS_Face face = faceBuilder.Face();
        orientFaceToPatch(face,
                          surface,
                          projected[outer].parameters.front(),
                          moments.normal);
        topologyBuilder.UpdateFace(face, sewingTolerance);
        if (!BRepLib::BuildCurves3d(face, sewingTolerance)) {
            throw std::runtime_error("OpenCascade could not build analytic edge curves in 3D");
        }
        if (!BRepCheck_Analyzer(face, true).IsValid()) {
            throw std::runtime_error("OpenCascade produced an invalid analytic face");
        }
        faces.push_back(face);
    }
    if (faces.empty()) throw std::runtime_error("no analytic face was constructed");
    return faces;
}

// GeomPlate accepts scattered constraints, unlike grid-based B-spline fitting.
// A chart must project consistently to an initial TLS plane; folded/closed
// patches require another valid parameter chart or reconstruction fails.
[[nodiscard]] Handle(Geom_Surface) fitNurbsPatch(
    const Mesh& mesh, const SurfacePatchFit& patch, double tolerance) {
    std::vector<std::vector<std::uint32_t>> loops;
    std::string error;
    if (!extractBoundaryLoops(mesh, patch, loops, error) || loops.empty()) {
        throw std::runtime_error("NURBS fitting requires a bounded manifold chart");
    }
    const PlaneSegment plane = fitPlaneTLS(mesh, patch.triangleIndices);
    for (const auto triangle : patch.triangleIndices) {
        if (dot(mesh.features[triangle].normal, plane.plane.normal) < 0.25) {
            throw std::runtime_error("NURBS chart folds across its initial plane");
        }
    }
    const Handle(Geom_Surface) initial = new Geom_Plane(
        gp_Ax3(toPoint(plane.centroid), toDirection(plane.plane.normal)));
    GeomPlate_BuildPlateSurface plate(initial, 3, 8, 3,
                                      Precision::PConfusion(), tolerance);
    std::set<std::uint32_t> boundaryVertices;
    for (const auto& loop : loops) {
        for (std::size_t i = 0; i < loop.size(); ++i) {
            const auto first = loop[i];
            const auto last = loop[(i + 1U) % loop.size()];
            boundaryVertices.insert(first);
            const Handle(Geom_Curve) segment = GC_MakeSegment(
                toPoint(mesh.vertices[first]), toPoint(mesh.vertices[last])).Value();
            const Handle(Adaptor3d_Curve) curve = new GeomAdaptor_Curve(segment);
            plate.Add(new GeomPlate_CurveConstraint(curve, 0, 3, tolerance));
        }
    }
    std::set<std::uint32_t> interiorVertices;
    for (const auto triangle : patch.triangleIndices) {
        for (const auto vertex : mesh.triangles[triangle].vertices) {
            if (!boundaryVertices.contains(vertex)) interiorVertices.insert(vertex);
        }
    }
    const std::size_t stride = std::max<std::size_t>(1U,
        (interiorVertices.size() + 255U) / 256U);
    std::size_t index = 0;
    for (const auto vertex : interiorVertices) {
        if (index++ % stride == 0U) {
            plate.Add(new GeomPlate_PointConstraint(toPoint(mesh.vertices[vertex]),
                                                     0, tolerance));
        }
    }
    plate.Perform();
    if (!plate.IsDone()) throw std::runtime_error("OpenCascade plate fitting failed");
    GeomPlate_MakeApprox approximation(plate.Surface(), tolerance, 16, 8,
                                        std::max(tolerance, plate.G0Error() * 10.0), 0);
    const Handle(Geom_Surface) surface = approximation.Surface();
    if (surface.IsNull() || approximation.ApproxError() > tolerance * 4.0) {
        throw std::runtime_error("OpenCascade NURBS approximation exceeded tolerance");
    }
    // Check every input vertex, including those omitted from the bounded-size
    // fitting system. A fitted chart cannot discard unsupported detail.
    for (const auto triangle : patch.triangleIndices) {
        for (const auto vertex : mesh.triangles[triangle].vertices) {
            GeomAPI_ProjectPointOnSurf projection(toPoint(mesh.vertices[vertex]), surface);
            if (!projection.IsDone() || projection.NbPoints() == 0 ||
                projection.LowerDistance() > tolerance * 4.0) {
                throw std::runtime_error("Fitted NURBS failed full-vertex validation");
            }
        }
    }
    return surface;
}

[[nodiscard]] std::string patchDiagnostic(
    std::size_t patchIndex,
    const std::string& reason) {
    std::ostringstream message;
    message << "Patch " << patchIndex << " reconstruction failed: " << reason;
    return message.str();
}

// Spherical coordinates have two poles and a seam. A broad patch with holes
// cannot be represented as one ordinary UV polygon. Six directional charts
// bound angular extent; connectivity separates islands within each chart.
std::vector<SurfacePatchFit> parameterCharts(const Mesh& mesh, const SurfacePatchFit& patch) {
    if (patch.kind != SurfaceKind::Sphere) return {patch};
    const auto& sphere = std::get<SphereSurface>(patch.parameters);
    const auto moments = patchMoments(mesh, patch);
    const Vec3d direction = normalizedVector(moments.centroid - sphere.center);
    bool bounded = vectorLength(direction) > 0.0;
    for (const auto triangle : patch.triangleIndices) {
        for (const auto vertex : mesh.triangles[triangle].vertices) {
            if (dotProduct(direction, normalizedVector(mesh.vertices[vertex] - sphere.center)) < 0.25) bounded = false;
        }
    }
    if (bounded) return {patch};
    std::vector<int> chart(mesh.triangles.size(), -1);
    for (const auto triangle : patch.triangleIndices) {
        const Vec3d radial = mesh.features[triangle].centroid - sphere.center;
        unsigned axis = 0;
        if (std::abs(radial.y) > std::abs(radial.x)) axis = 1;
        if (std::abs(radial.z) > std::abs(radial[axis])) axis = 2;
        chart[triangle] = static_cast<int>(2 * axis + (radial[axis] < 0.0 ? 1 : 0));
    }
    std::vector<unsigned char> visited(mesh.triangles.size(), 0);
    std::vector<SurfacePatchFit> charts;
    for (const auto seed : patch.triangleIndices) {
        if (visited[seed]) continue;
        SurfacePatchFit part = patch;
        part.triangleIndices = {seed};
        visited[seed] = 1;
        for (std::size_t head = 0; head < part.triangleIndices.size(); ++head) {
            for (const auto neighbor : mesh.triangleNeighbors[part.triangleIndices[head]]) {
                if (!visited[neighbor] && chart[neighbor] == chart[seed]) {
                    visited[neighbor] = 1;
                    part.triangleIndices.push_back(neighbor);
                }
            }
        }
        charts.push_back(std::move(part));
    }
    return charts;
}

} // namespace

PatchReconstructionResult buildPatchReconstruction(
    const Mesh& mesh,
    const std::vector<SurfacePatchFit>& patches,
    const PatchReconstructionOptions& options) {
    if (mesh.vertices.empty() || mesh.triangles.empty()) {
        throw std::invalid_argument("Patch reconstruction requires a non-empty triangle mesh");
    }
    if (!std::isfinite(options.maximumBoundaryProjectionDistance) ||
        options.maximumBoundaryProjectionDistance < 0.0) {
        throw std::invalid_argument(
            "Maximum boundary projection distance must be finite and non-negative");
    }
    if (!std::isfinite(options.nurbsFitTolerance) || options.nurbsFitTolerance < 0.0) {
        throw std::invalid_argument("NURBS fit tolerance must be finite and non-negative");
    }

    PatchReconstructionResult result;
    result.inputTriangleCount = mesh.triangles.size();
    result.inputPatchCount = patches.size();
    result.inputReportedWatertight = mesh.stats.watertight;
    result.sewingTolerance = resolveSewingTolerance(mesh, patches, options.sewingTolerance);
    if (options.sewingTolerance == 0.0 && options.enableNurbsFitting) {
        const bool needsNurbs = std::any_of(patches.begin(), patches.end(),
            [](const SurfacePatchFit& patch) { return !patch.accepted; });
        if (needsNurbs) {
            const double fitTolerance = options.nurbsFitTolerance > 0.0
                ? options.nurbsFitTolerance
                : std::max(mesh.weldTolerance * 8.0, meshScale(mesh) * 2.0e-5);
            result.sewingTolerance = std::max(result.sewingTolerance, fitTolerance * 4.0);
        }
    }
    // Use numeric precision while constructing faces. The larger chord-sag
    // allowance is for joining different patches, not for collapsing their
    // short boundary edges or vertices during construction.
    const double constructionTolerance = std::max(Precision::Confusion(),
        meshScale(mesh) * 1.0e-8);
    if (!mesh.stats.watertight) {
        result.diagnostics.emplace_back(
            "Input topology is not reported watertight; closed-solid recovery may be incomplete");
    }

    std::vector<std::int32_t> triangleOwner(mesh.triangles.size(), -1);
    for (std::size_t patchIndex = 0; patchIndex < patches.size(); ++patchIndex) {
        for (const std::uint32_t triangleIndex : patches[patchIndex].triangleIndices) {
            if (triangleIndex >= mesh.triangles.size()) {
                throw std::out_of_range("A surface patch contains an out-of-range triangle index");
            }
            if (triangleOwner[triangleIndex] >= 0) {
                throw std::invalid_argument("Surface patch triangle sets must be disjoint");
            }
            triangleOwner[triangleIndex] = static_cast<std::int32_t>(patchIndex);
        }
    }

    BRepBuilderAPI_Sewing sewing(result.sewingTolerance, true, true, true, false);
    if (std::any_of(triangleOwner.begin(), triangleOwner.end(),
                    [](std::int32_t owner) { return owner < 0; })) {
        result.diagnostics.emplace_back("Every triangle must belong to a fitted surface patch");
        return result;
    }

    for (std::size_t patchIndex = 0; patchIndex < patches.size(); ++patchIndex) {
        const SurfacePatchFit& patch = patches[patchIndex];
        const bool supportedAnalytic = patch.accepted &&
            (patch.kind == SurfaceKind::Plane || patch.kind == SurfaceKind::Cylinder ||
             patch.kind == SurfaceKind::Cone ||
             patch.kind == SurfaceKind::Sphere || patch.kind == SurfaceKind::Torus);
        if (!supportedAnalytic || patch.triangleIndices.empty()) {
            if (options.enableNurbsFitting && patch.triangleIndices.size() >= 4U) {
                try {
                    const double tolerance = options.nurbsFitTolerance > 0.0
                        ? options.nurbsFitTolerance
                        : std::max({Precision::Confusion(), mesh.weldTolerance * 8.0,
                                    meshScale(mesh) * 2.0e-5});
                    const Handle(Geom_Surface) surface = fitNurbsPatch(mesh, patch, tolerance);
                    double projectionError = 0.0;
                    const auto faces = makeAnalyticPatchFaces(mesh, patch,
                        constructionTolerance, tolerance * 4.0,
                        projectionError, surface);
                    for (const auto& face : faces) sewing.Add(face);
                    result.fittedNurbsFaceCount += faces.size();
                    continue;
                } catch (const Standard_Failure& failure) {
                    result.diagnostics.push_back(std::string("NURBS fitting: ") + failure.what());
                } catch (const std::exception& failure) {
                    result.diagnostics.push_back(std::string("NURBS fitting: ") + failure.what());
                }
            }
            if (!patch.triangleIndices.empty()) {
                result.diagnostics.push_back(patchDiagnostic(
                    patchIndex,
                    patch.diagnostic.empty() ? "no analytic model was accepted"
                                             : patch.diagnostic));
            }
            return result;
        }

        const double projectionLimit = options.maximumBoundaryProjectionDistance > 0.0
            ? options.maximumBoundaryProjectionDistance
            : std::max(result.sewingTolerance * 4.0,
                       std::isfinite(patch.residuals.positionMax)
                           ? patch.residuals.positionMax * 2.0
                           : 0.0);
        try {
            double observedProjectionDistance = 0.0;
            std::size_t count = 0;
            for (const auto& chart : parameterCharts(mesh, patch)) {
                const auto faces = makeAnalyticPatchFaces(mesh, chart, constructionTolerance,
                    projectionLimit, observedProjectionDistance);
                for (const auto& face : faces) sewing.Add(face);
                count += faces.size();
            }
            ++result.acceptedAnalyticPatchCount;
            result.analyticFaceCount += count;
        } catch (const Standard_Failure& error) {
            ++result.analyticPatchFailureCount;
            result.diagnostics.push_back(patchDiagnostic(
                patchIndex,
                error.what() != nullptr ? error.what()
                                        : "OpenCascade analytic construction failure"));
            return result;
        } catch (const std::exception& error) {
            ++result.analyticPatchFailureCount;
            result.diagnostics.push_back(patchDiagnostic(patchIndex, error.what()));
            return result;
        }
    }

    try {
        sewing.Perform();
    } catch (const Standard_Failure& error) {
        result.diagnostics.emplace_back(
            error.what() != nullptr ? error.what() : "OpenCascade sewing failed");
        return result;
    }
    result.freeEdgeCount = static_cast<std::size_t>(std::max(0, sewing.NbFreeEdges()));
    if (result.freeEdgeCount > 0U) {
        std::ostringstream details;
        details << "Sewing tolerance " << result.sewingTolerance;
        result.diagnostics.push_back(details.str());
        for (int edgeIndex = 1; edgeIndex <= sewing.NbFreeEdges(); ++edgeIndex) {
            const BRepAdaptor_Curve curve(sewing.FreeEdge(edgeIndex));
            const gp_Pnt first = curve.Value(curve.FirstParameter());
            const gp_Pnt last = curve.Value(curve.LastParameter());
            const gp_Pnt middle = curve.Value(
                0.5 * (curve.FirstParameter() + curve.LastParameter()));
            std::ostringstream edgeDetails;
            edgeDetails << "Free edge " << edgeIndex << ": ("
                        << first.X() << ',' << first.Y() << ',' << first.Z()
                        << ") -> (" << last.X() << ',' << last.Y() << ',' << last.Z()
                        << "), midpoint (" << middle.X() << ',' << middle.Y() << ','
                        << middle.Z() << ')';
            result.diagnostics.push_back(edgeDetails.str());
        }
    }
    result.multipleEdgeCount =
        static_cast<std::size_t>(std::max(0, sewing.NbMultipleEdges()));
    const TopoDS_Shape sewedShape = sewing.SewedShape();
    if (sewedShape.IsNull()) {
        result.diagnostics.emplace_back("OpenCascade sewing returned a null shape");
        return result;
    }

    ShapeMap shells;
    TopExp::MapShapes(sewedShape, TopAbs_SHELL, shells);
    std::vector<TopoDS_Shape> assembled;
    assembled.reserve(static_cast<std::size_t>(std::max(1, shells.Extent())));
    const double minimumTolerance = std::min(result.sewingTolerance, Precision::Confusion());
    const double maximumTolerance =
        std::max(result.sewingTolerance * 10.0, Precision::Confusion());
    for (int shellIndex = 1; shellIndex <= shells.Extent(); ++shellIndex) {
        const TopoDS_Shell shell = TopoDS::Shell(shells(shellIndex));
        if (!BRep_Tool::IsClosed(shell)) {
            assembled.emplace_back(shell);
            continue;
        }
        try {
            ShapeFix_Solid solidBuilder;
            solidBuilder.SetPrecision(result.sewingTolerance);
            solidBuilder.SetMinTolerance(minimumTolerance);
            solidBuilder.SetMaxTolerance(maximumTolerance);
            TopoDS_Solid solid = solidBuilder.SolidFromShell(shell);
            if (solid.IsNull()) {
                result.diagnostics.emplace_back(
                    "OpenCascade could not create a solid from a closed sewn shell");
                assembled.emplace_back(shell);
                continue;
            }
            if (!BRepLib::OrientClosedSolid(solid)) {
                result.diagnostics.emplace_back(
                    "OpenCascade could not determine a finite solid orientation");
            }
            assembled.emplace_back(solid);
        } catch (const Standard_Failure& error) {
            result.diagnostics.emplace_back(
                error.what() != nullptr ? error.what()
                                        : "OpenCascade solid construction failed");
            assembled.emplace_back(shell);
        }
    }
    if (assembled.empty()) assembled.emplace_back(sewedShape);

    TopoDS_Shape combined = assembleShapes(assembled);
    try {
        ShapeFix_Shape healer(combined);
        healer.SetPrecision(result.sewingTolerance);
        healer.SetMinTolerance(minimumTolerance);
        healer.SetMaxTolerance(maximumTolerance);
        healer.Perform();
        const TopoDS_Shape healed = healer.Shape();
        if (!healed.IsNull()) combined = healed;
    } catch (const Standard_Failure& error) {
        result.diagnostics.emplace_back(
            error.what() != nullptr ? error.what() : "OpenCascade shape healing failed");
    }

    ShapeUpgrade_UnifySameDomain unifier(combined, true, true, false);
    unifier.SetLinearTolerance(constructionTolerance);
    unifier.Build();
    result.shape = unifier.Shape();
    result.shellCount = countSubshapes(result.shape, TopAbs_SHELL);
    result.closedShellCount = countClosedShells(result.shape);
    result.solidCount = countSubshapes(result.shape, TopAbs_SOLID);
    result.valid = !result.shape.IsNull() && BRepCheck_Analyzer(result.shape, true).IsValid();
    result.closed = result.solidCount > 0U && result.shellCount > 0U &&
                    result.closedShellCount == result.shellCount &&
                    result.freeEdgeCount == 0U && result.multipleEdgeCount == 0U;
    result.succeeded = result.valid && result.closed && result.skippedTriangleCount == 0U;

    if (result.freeEdgeCount > 0U) {
        std::ostringstream message;
        message << "Sewing left " << result.freeEdgeCount << " free edge(s)";
        result.diagnostics.push_back(message.str());
    }
    if (result.multipleEdgeCount > 0U) {
        std::ostringstream message;
        message << "Sewing found " << result.multipleEdgeCount
                << " non-manifold multiple edge(s)";
        result.diagnostics.push_back(message.str());
    }
    if (!result.valid) {
        result.diagnostics.emplace_back("The final patch shape failed BRepCheck_Analyzer");
    }
    if (result.solidCount == 0U) {
        result.diagnostics.emplace_back("No closed solid could be built from the sewn faces");
    }
    return result;
}

} // namespace recad
