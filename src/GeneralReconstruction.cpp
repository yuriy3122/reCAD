#include "GeneralReconstruction.h"

#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_Sewing.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepLib.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <GC_MakeSegment2d.hxx>
#include <Geom2d_TrimmedCurve.hxx>
#include <Geom_BSplineSurface.hxx>
#include <NCollection_Array1.hxx>
#include <NCollection_Array2.hxx>
#include <NCollection_IndexedMap.hxx>
#include <Precision.hxx>
#include <ShapeFix_Shape.hxx>
#include <ShapeFix_Solid.hxx>
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
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace recad {
namespace {

using ShapeMap = NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher>;

gp_Pnt toPoint(const Vec3d& point) {
    return {point.x, point.y, point.z};
}

bool finitePoint(const Vec3d& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

double meshScale(const Mesh& mesh) {
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

double resolveTolerance(const Mesh& mesh, double requested) {
    if (!std::isfinite(requested) || requested < 0.0) {
        throw std::invalid_argument("NURBS fallback sewing tolerance must be finite and non-negative");
    }
    if (requested > 0.0) return requested;

    const double scale = meshScale(mesh);
    const double scaleAware = scale > 0.0 ? scale * 1.0e-8 : 0.0;
    return std::max({Precision::Confusion(), mesh.weldTolerance, scaleAware});
}

std::size_t countSubshapes(const TopoDS_Shape& shape, TopAbs_ShapeEnum kind) {
    if (shape.IsNull()) return 0;
    ShapeMap mapped;
    TopExp::MapShapes(shape, kind, mapped);
    return static_cast<std::size_t>(mapped.Extent());
}

std::size_t countClosedShells(const TopoDS_Shape& shape) {
    if (shape.IsNull()) return 0;
    ShapeMap shells;
    TopExp::MapShapes(shape, TopAbs_SHELL, shells);

    std::size_t closed = 0;
    for (int index = 1; index <= shells.Extent(); ++index) {
        if (BRep_Tool::IsClosed(shells(index))) ++closed;
    }
    return closed;
}

Handle(Geom2d_TrimmedCurve) parameterSegment(const gp_Pnt2d& first,
                                              const gp_Pnt2d& last) {
    const GC_MakeSegment2d builder(first, last);
    if (!builder.IsDone()) {
        throw std::runtime_error("OpenCascade could not construct a triangle parameter edge");
    }
    return builder.Value();
}

TopoDS_Edge makeSurfaceEdge(const Handle(Geom_BSplineSurface)& surface,
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
        throw std::runtime_error("OpenCascade could not construct an edge on a NURBS patch");
    }
    return builder.Edge();
}

TopoDS_Face makeTriangleNurbsFace(const Vec3d& p0,
                                  const Vec3d& p1,
                                  const Vec3d& p2,
                                  const TopoDS_Vertex& v0,
                                  const TopoDS_Vertex& v1,
                                  const TopoDS_Vertex& v2,
                                  double tolerance) {
    // A bilinear 2x2 patch is affine when its fourth pole completes the
    // parallelogram.  The triangle occupies the (0,0)-(1,0)-(0,1) half of it.
    // Its surface normal therefore follows the input triangle winding.
    const Vec3d p11 = p1 + p2 - p0;
    NCollection_Array2<gp_Pnt> poles(1, 2, 1, 2);
    poles.SetValue(1, 1, toPoint(p0));
    poles.SetValue(2, 1, toPoint(p1));
    poles.SetValue(1, 2, toPoint(p2));
    poles.SetValue(2, 2, toPoint(p11));

    NCollection_Array1<double> uKnots(1, 2);
    NCollection_Array1<double> vKnots(1, 2);
    NCollection_Array1<int> uMultiplicities(1, 2);
    NCollection_Array1<int> vMultiplicities(1, 2);
    uKnots.SetValue(1, 0.0);
    uKnots.SetValue(2, 1.0);
    vKnots.SetValue(1, 0.0);
    vKnots.SetValue(2, 1.0);
    uMultiplicities.SetValue(1, 2);
    uMultiplicities.SetValue(2, 2);
    vMultiplicities.SetValue(1, 2);
    vMultiplicities.SetValue(2, 2);

    const Handle(Geom_BSplineSurface) surface = new Geom_BSplineSurface(
        poles,
        uKnots,
        vKnots,
        uMultiplicities,
        vMultiplicities,
        1,
        1,
        false,
        false);

    const gp_Pnt2d uv0(0.0, 0.0);
    const gp_Pnt2d uv1(1.0, 0.0);
    const gp_Pnt2d uv2(0.0, 1.0);
    const TopoDS_Edge e01 = makeSurfaceEdge(surface, uv0, uv1, v0, v1);
    const TopoDS_Edge e12 = makeSurfaceEdge(surface, uv1, uv2, v1, v2);
    const TopoDS_Edge e20 = makeSurfaceEdge(surface, uv2, uv0, v2, v0);

    BRepBuilderAPI_MakeWire wireBuilder;
    wireBuilder.Add(e01);
    wireBuilder.Add(e12);
    wireBuilder.Add(e20);
    if (!wireBuilder.IsDone()) {
        throw std::runtime_error("OpenCascade could not close a triangle NURBS wire");
    }

    BRepBuilderAPI_MakeFace faceBuilder(surface, wireBuilder.Wire(), true);
    if (!faceBuilder.IsDone()) {
        throw std::runtime_error("OpenCascade could not trim a triangle NURBS face");
    }
    TopoDS_Face face = faceBuilder.Face();

    BRep_Builder topologyBuilder;
    topologyBuilder.UpdateFace(face, tolerance);
    if (!BRepLib::BuildCurves3d(face, tolerance)) {
        throw std::runtime_error("OpenCascade could not build triangle edge curves in 3D");
    }
    return face;
}

TopoDS_Shape assembleShapes(const std::vector<TopoDS_Shape>& shapes) {
    if (shapes.empty()) return {};
    if (shapes.size() == 1) return shapes.front();

    BRep_Builder builder;
    TopoDS_Compound compound;
    builder.MakeCompound(compound);
    for (const TopoDS_Shape& shape : shapes) builder.Add(compound, shape);
    return compound;
}

std::string triangleDiagnostic(std::size_t triangleIndex, const std::string& reason) {
    std::ostringstream stream;
    stream << "Triangle " << triangleIndex << " was skipped: " << reason;
    return stream.str();
}

} // namespace

GeneralReconstructionResult buildNurbsFallback(const Mesh& mesh, double sewingTolerance) {
    if (mesh.vertices.empty() || mesh.triangles.empty()) {
        throw std::invalid_argument("NURBS fallback requires a non-empty triangle mesh");
    }

    GeneralReconstructionResult result;
    result.inputTriangleCount = mesh.triangles.size();
    result.inputReportedWatertight = mesh.stats.watertight;
    result.sewingTolerance = resolveTolerance(mesh, sewingTolerance);
    if (!mesh.stats.watertight) {
        result.diagnostics.emplace_back(
            "Input topology is not reported watertight; closed-solid recovery may be incomplete");
    }

    BRep_Builder topologyBuilder;
    std::vector<TopoDS_Vertex> vertices(mesh.vertices.size());
    std::vector<bool> usableVertex(mesh.vertices.size(), false);
    for (std::size_t index = 0; index < mesh.vertices.size(); ++index) {
        if (!finitePoint(mesh.vertices[index])) continue;
        vertices[index] = BRepBuilderAPI_MakeVertex(toPoint(mesh.vertices[index])).Vertex();
        topologyBuilder.UpdateVertex(vertices[index], result.sewingTolerance);
        usableVertex[index] = true;
    }

    const double scale = std::max(meshScale(mesh), 1.0);
    const double twiceAreaThreshold =
        std::numeric_limits<double>::epsilon() * scale * scale * 16.0;
    BRepBuilderAPI_Sewing sewing(result.sewingTolerance, true, true, true, false);

    for (std::size_t triangleIndex = 0;
         triangleIndex < mesh.triangles.size();
         ++triangleIndex) {
        const Triangle& triangle = mesh.triangles[triangleIndex];
        const std::size_t i0 = triangle[0];
        const std::size_t i1 = triangle[1];
        const std::size_t i2 = triangle[2];
        if (i0 >= mesh.vertices.size() || i1 >= mesh.vertices.size() ||
            i2 >= mesh.vertices.size()) {
            ++result.skippedTriangleCount;
            result.diagnostics.push_back(
                triangleDiagnostic(triangleIndex, "a vertex index is out of range"));
            continue;
        }
        if (i0 == i1 || i1 == i2 || i2 == i0 ||
            !usableVertex[i0] || !usableVertex[i1] || !usableVertex[i2]) {
            ++result.skippedTriangleCount;
            result.diagnostics.push_back(
                triangleDiagnostic(triangleIndex, "its vertices are invalid or repeated"));
            continue;
        }

        const Vec3d& p0 = mesh.vertices[i0];
        const Vec3d& p1 = mesh.vertices[i1];
        const Vec3d& p2 = mesh.vertices[i2];
        const double twiceArea = norm(cross(p1 - p0, p2 - p0));
        if (!std::isfinite(twiceArea) || twiceArea <= twiceAreaThreshold) {
            ++result.skippedTriangleCount;
            result.diagnostics.push_back(
                triangleDiagnostic(triangleIndex, "it is geometrically degenerate"));
            continue;
        }

        try {
            TopoDS_Face face = makeTriangleNurbsFace(
                p0,
                p1,
                p2,
                vertices[i0],
                vertices[i1],
                vertices[i2],
                result.sewingTolerance);
            if (!BRepCheck_Analyzer(face, true).IsValid()) {
                ++result.skippedTriangleCount;
                result.diagnostics.push_back(
                    triangleDiagnostic(triangleIndex, "OpenCascade produced an invalid face"));
                continue;
            }
            sewing.Add(face);
            ++result.createdFaceCount;
        } catch (const Standard_Failure& error) {
            ++result.skippedTriangleCount;
            result.diagnostics.push_back(triangleDiagnostic(
                triangleIndex,
                error.what() != nullptr ? error.what() : "OpenCascade construction failure"));
        } catch (const std::exception& error) {
            ++result.skippedTriangleCount;
            result.diagnostics.push_back(triangleDiagnostic(triangleIndex, error.what()));
        }
    }

    if (result.createdFaceCount == 0) {
        result.diagnostics.emplace_back("No valid NURBS triangle faces were constructed");
        return result;
    }

    try {
        sewing.Perform();
    } catch (const Standard_Failure& error) {
        result.diagnostics.emplace_back(
            error.what() != nullptr ? error.what() : "OpenCascade sewing failed");
        return result;
    }

    result.freeEdgeCount = static_cast<std::size_t>(std::max(0, sewing.NbFreeEdges()));
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
                    "OpenCascade could not determine a finite orientation for a closed shell");
            }
            assembled.emplace_back(solid);
        } catch (const Standard_Failure& error) {
            result.diagnostics.emplace_back(
                error.what() != nullptr ? error.what()
                                        : "OpenCascade solid construction failed");
            assembled.emplace_back(shell);
        }
    }

    if (assembled.empty()) {
        // A single face (or an unexpected OCCT aggregate) has no shell.  Keep
        // it as a diagnostic shape rather than discarding valid geometry.
        assembled.emplace_back(sewedShape);
    }

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

    result.shape = combined;
    result.shellCount = countSubshapes(result.shape, TopAbs_SHELL);
    result.closedShellCount = countClosedShells(result.shape);
    result.solidCount = countSubshapes(result.shape, TopAbs_SOLID);
    result.valid = !result.shape.IsNull() && BRepCheck_Analyzer(result.shape, true).IsValid();
    result.closed = result.solidCount > 0 && result.shellCount > 0 &&
                    result.closedShellCount == result.shellCount &&
                    result.freeEdgeCount == 0 && result.multipleEdgeCount == 0;
    result.succeeded = result.valid && result.closed && result.skippedTriangleCount == 0;

    if (result.freeEdgeCount > 0) {
        std::ostringstream stream;
        stream << "Sewing left " << result.freeEdgeCount << " free edge(s)";
        result.diagnostics.push_back(stream.str());
    }
    if (result.multipleEdgeCount > 0) {
        std::ostringstream stream;
        stream << "Sewing found " << result.multipleEdgeCount
               << " non-manifold multiple edge(s)";
        result.diagnostics.push_back(stream.str());
    }
    if (!result.valid) {
        result.diagnostics.emplace_back("The final fallback shape failed BRepCheck_Analyzer");
    }
    if (result.solidCount == 0) {
        result.diagnostics.emplace_back("No closed solid could be built from the sewn faces");
    }
    return result;
}

} // namespace recad
