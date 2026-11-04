#include "ReferenceValidation.h"

#include <BRepAdaptor_Surface.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_Copy.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepGProp.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepTools.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <GProp_GProps.hxx>
#include <GeomAbs_SurfaceType.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <NCollection_IndexedMap.hxx>
#include <Poly_Triangulation.hxx>
#include <Precision.hxx>
#include <STEPControl_Reader.hxx>
#include <STEPConstruct_UnitContext.hxx>
#include <StepData_StepModel.hxx>
#include <StepData_UndefinedEntity.hxx>
#include <Interface_ReportEntity.hxx>
#include <Interface_UndefinedContent.hxx>
#include <StepData_Factors.hxx>
#include <StepRepr_GlobalUnitAssignedContext.hxx>
#include <StepGeom_GeomRepContextAndGlobUnitAssCtxAndGlobUncertaintyAssCtx.hxx>
#include <StepGeom_CartesianPoint.hxx>
#include <StepShape_CsgSolid.hxx>
#include <StepShape_CsgShapeRepresentation.hxx>
#include <StepShape_BooleanResult.hxx>
#include <StepShape_Block.hxx>
#include <StepShape_Sphere.hxx>
#include <StepShape_Torus.hxx>
#include <StepShape_RightCircularCylinder.hxx>
#include <StepShape_RightCircularCone.hxx>
#include <StepShape_RightAngularWedge.hxx>
#include <StepToGeom.hxx>
#include <Geom_Axis1Placement.hxx>
#include <Geom_Axis2Placement.hxx>
#include <Geom_CartesianPoint.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakeCone.hxx>
#include <BRepPrimAPI_MakeTorus.hxx>
#include <BRepPrimAPI_MakeWedge.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Common.hxx>
#include <Standard_Failure.hxx>
#include <TopAbs_Orientation.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopLoc_Location.hxx>
#include <TopTools_ShapeMapHasher.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Vertex.hxx>
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Trsf.hxx>
#include <gp_Ax1.hxx>
#include <gp_Dir.hxx>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace recad {
namespace {

using ShapeMap = NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher>;

// STEP's CSG representation is an explicit expression graph. OCCT parses its
// entities but does not transfer these roots to solids. Evaluate that graph
// with OCCT's primitive and Boolean builders for independent reference tests.
// This adapter never receives a mesh, fit, filename rule, or expected dimensions.
TopoDS_Shape transferCsgReference(const Handle(StepData_StepModel)& model, int& rootCount) {
    std::vector<Handle(StepShape_CsgSolid)> roots;
    std::vector<Handle(StepShape_BooleanResult)> operations;
    StepData_Factors factors;
    bool unitsFound = false;
    for (int i = 1; i <= model->NbEntities(); ++i) {
        const auto entity = model->Value(i);
        if (const auto root = Handle(StepShape_CsgSolid)::DownCast(entity); !root.IsNull()) roots.push_back(root);
        if (const auto context = Handle(StepGeom_GeomRepContextAndGlobUnitAssCtxAndGlobUncertaintyAssCtx)::DownCast(entity);
            !context.IsNull()) {
            STEPConstruct_UnitContext units;
            if (units.ComputeFactors(context->GlobalUnitAssignedContext()) != 0 || !units.LengthDone()) {
                throw std::runtime_error("CSG reference has unsupported units");
            }
            if (unitsFound && (factors.LengthFactor() != units.LengthFactor() ||
                               factors.PlaneAngleFactor() != units.PlaneAngleFactor())) {
                throw std::runtime_error("CSG reference has multiple distinct unit contexts");
            }
            factors.InitializeFactors(units.LengthFactor(), units.PlaneAngleFactor(), units.SolidAngleFactor());
            unitsFound = true;
        }
    }
    rootCount = static_cast<int>(roots.size());
    if (roots.empty()) return {};
    if (!unitsFound) throw std::runtime_error("CSG reference has no supported unit context");
    const double scale = factors.LengthFactor();
    std::unordered_map<const Standard_Transient*, TopoDS_Shape> shapes;
    const auto placement = [&](const auto& position) {
        const auto axis = StepToGeom::MakeAxis1Placement(position, factors);
        if (axis.IsNull()) throw std::runtime_error("Invalid CSG axis placement");
        return gp_Ax2(axis->Location(), axis->Direction());
    };
    for (int i = 1; i <= model->NbEntities(); ++i) {
        const auto entity = model->Value(i);
        TopoDS_Shape shape;
        if (const auto block = Handle(StepShape_Block)::DownCast(entity); !block.IsNull()) {
            const auto axis = StepToGeom::MakeAxis2Placement(block->Position(), factors);
            if (axis.IsNull()) throw std::runtime_error("Invalid CSG block placement");
            shape = BRepPrimAPI_MakeBox(axis->Ax2(), block->X() * scale, block->Y() * scale, block->Z() * scale).Shape();
        } else if (const auto sphere = Handle(StepShape_Sphere)::DownCast(entity); !sphere.IsNull()) {
            const auto point = StepToGeom::MakeCartesianPoint(Handle(StepGeom_CartesianPoint)::DownCast(sphere->Centre()), factors);
            if (point.IsNull()) throw std::runtime_error("Invalid CSG sphere centre");
            shape = BRepPrimAPI_MakeSphere(point->Pnt(), sphere->Radius() * scale).Shape();
        } else if (const auto cylinder = Handle(StepShape_RightCircularCylinder)::DownCast(entity); !cylinder.IsNull()) {
            shape = BRepPrimAPI_MakeCylinder(placement(cylinder->Position()), cylinder->Radius() * scale,
                                             cylinder->Height() * scale).Shape();
        } else if (const auto cone = Handle(StepShape_RightCircularCone)::DownCast(entity); !cone.IsNull()) {
            const double radius = cone->Radius() * scale;
            const double height = cone->Height() * scale;
            // ISO 10303-42 locates the base in the NEGATIVE axis direction
            // from position.location, which is the truncated top (or apex).
            gp_Ax2 axis = placement(cone->Position());
            axis.SetLocation(axis.Location().Translated(gp_Vec(axis.Direction()) * -height));
            const double baseRadius = radius + height * std::tan(cone->SemiAngle() * factors.PlaneAngleFactor());
            shape = BRepPrimAPI_MakeCone(axis, baseRadius, radius, height).Shape();
        } else if (const auto torus = Handle(StepShape_Torus)::DownCast(entity); !torus.IsNull()) {
            shape = BRepPrimAPI_MakeTorus(placement(torus->Position()), torus->MajorRadius() * scale,
                                          torus->MinorRadius() * scale).Shape();
        } else if (const auto wedge = Handle(StepShape_RightAngularWedge)::DownCast(entity); !wedge.IsNull()) {
            const auto axis = StepToGeom::MakeAxis2Placement(wedge->Position(), factors);
            if (axis.IsNull()) throw std::runtime_error("Invalid CSG wedge placement");
            shape = BRepPrimAPI_MakeWedge(axis->Ax2(), wedge->X() * scale, wedge->Y() * scale,
                                          wedge->Z() * scale, wedge->Ltx() * scale).Shape();
        } else if (const auto operation = Handle(StepShape_BooleanResult)::DownCast(entity); !operation.IsNull()) {
            operations.push_back(operation);
        }
        if (!shape.IsNull()) shapes.emplace(entity.get(), shape);
    }
    const auto operandEntity = [&](const Handle(StepShape_BooleanResult)& operation, int parameter) -> Handle(Standard_Transient) {
        // OCCT's Boolean reader only types SolidModel operands; the native
        // parser retains primitive/Boolean references in the literal record.
        const auto report = model->ReportEntity(model->Number(operation));
        if (!report.IsNull() && report->HasContent()) {
            const auto record = Handle(StepData_UndefinedEntity)::DownCast(report->Content());
            if (!record.IsNull() && record->UndefinedContent()->IsParamEntity(parameter)) {
                return record->UndefinedContent()->ParamEntity(parameter);
            }
        }
        const auto operand = parameter == 3 ? operation->FirstOperand() : operation->SecondOperand();
        if (!operand.BooleanResult().IsNull()) return operand.BooleanResult();
        if (!operand.CsgPrimitive().Value().IsNull()) return operand.CsgPrimitive().Value();
        if (!operand.SolidModel().IsNull()) return operand.SolidModel();
        throw std::runtime_error("Unsupported STEP CSG operand");
    };
    // A bounded topological evaluation also handles forward references. A pass
    // without progress is an invalid cycle or unsupported dependency.
    while (!operations.empty()) {
        bool progressed = false;
        for (auto iterator = operations.begin(); iterator != operations.end();) {
            const auto& operation = *iterator;
            const auto left = shapes.find(operandEntity(operation, 3).get());
            const auto right = shapes.find(operandEntity(operation, 4).get());
            if (left == shapes.end() || right == shapes.end()) { ++iterator; continue; }
            TopoDS_Shape shape;
            switch (operation->Operator()) {
            case StepShape_boUnion: {
                BRepAlgoAPI_Fuse builder(left->second, right->second);
                if (!builder.IsDone()) throw std::runtime_error("CSG union failed");
                builder.SimplifyResult();
                shape = builder.Shape();
                break;
            }
            case StepShape_boDifference: {
                BRepAlgoAPI_Cut builder(left->second, right->second);
                if (!builder.IsDone()) throw std::runtime_error("CSG difference failed");
                builder.SimplifyResult();
                shape = builder.Shape();
                break;
            }
            case StepShape_boIntersection: {
                BRepAlgoAPI_Common builder(left->second, right->second);
                if (!builder.IsDone()) throw std::runtime_error("CSG intersection failed");
                builder.SimplifyResult();
                shape = builder.Shape();
                break;
            }
            default: throw std::runtime_error("Unsupported CSG Boolean operator");
            }
            shapes.emplace(operation.get(), shape);
            iterator = operations.erase(iterator);
            progressed = true;
        }
        if (!progressed) throw std::runtime_error("Unresolved or cyclic CSG expression graph");
    }
    BRep_Builder builder;
    TopoDS_Compound result;
    builder.MakeCompound(result);
    for (const auto& root : roots) {
        const auto expression = root->TreeRootExpression();
        const Handle(Standard_Transient) entity = !expression.BooleanResult().IsNull()
            ? Handle(Standard_Transient)(expression.BooleanResult()) : expression.CsgPrimitive().Value();
        const auto found = shapes.find(entity.get());
        if (found == shapes.end()) throw std::runtime_error("Unsupported CSG root expression");
        if (roots.size() == 1) return found->second;
        builder.Add(result, found->second);
    }
    return result;
}

struct SurfaceTriangle {
    Vec3d a;
    Vec3d b;
    Vec3d c;
    TopoDS_Face face;
    std::array<gp_Pnt2d, 3> uv;
    bool hasUv{};
    double area{};
};

struct SurfaceSample {
    Vec3d point;
    double weight{};
};

struct SampleSet {
    std::vector<SurfaceSample> samples;
    std::size_t candidateTriangleCount{};
    bool capped{};
    double area{};
};

std::size_t countSubshapes(const TopoDS_Shape& shape, TopAbs_ShapeEnum kind) {
    ShapeMap mapped;
    TopExp::MapShapes(shape, kind, mapped);
    return static_cast<std::size_t>(mapped.Extent());
}

std::int64_t countDelta(std::size_t reconstructed, std::size_t reference) {
    constexpr std::size_t limit =
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
    const std::int64_t lhs = static_cast<std::int64_t>(std::min(reconstructed, limit));
    const std::int64_t rhs = static_cast<std::int64_t>(std::min(reference, limit));
    return lhs - rhs;
}

Vec3d toVec(const gp_Pnt& point) {
    return {point.X(), point.Y(), point.Z()};
}

gp_Pnt toPoint(const Vec3d& point) {
    return {point.x, point.y, point.z};
}

double maximumCoordinateMagnitude(const Vec3d& value) {
    return std::max({std::abs(value.x), std::abs(value.y), std::abs(value.z)});
}

double safeRelativeError(double measured, double reference, double scaleFloor) {
    const double denominator = std::max(std::abs(reference), scaleFloor);
    return std::abs(measured - reference) / denominator;
}

ValidationShapeMetrics inspectShape(const TopoDS_Shape& shape,
                                    std::vector<std::string>& diagnostics,
                                    const char* description) {
    ValidationShapeMetrics result;
    if (shape.IsNull()) {
        diagnostics.emplace_back(std::string(description) + " shape is null");
        return result;
    }

    result.valid = BRepCheck_Analyzer(shape, true).IsValid();
    result.solidCount = countSubshapes(shape, TopAbs_SOLID);
    result.shellCount = countSubshapes(shape, TopAbs_SHELL);
    result.faceCount = countSubshapes(shape, TopAbs_FACE);
    result.edgeCount = countSubshapes(shape, TopAbs_EDGE);
    result.vertexCount = countSubshapes(shape, TopAbs_VERTEX);

    ShapeMap shells;
    TopExp::MapShapes(shape, TopAbs_SHELL, shells);
    for (int index = 1; index <= shells.Extent(); ++index) {
        if (BRep_Tool::IsClosed(shells(index))) ++result.closedShellCount;
    }
    result.closed = result.solidCount > 0 && result.shellCount > 0 &&
                    result.closedShellCount == result.shellCount;

    for (TopExp_Explorer explorer(shape, TopAbs_FACE); explorer.More(); explorer.Next()) {
        try {
            const BRepAdaptor_Surface surface(TopoDS::Face(explorer.Current()), true);
            switch (surface.GetType()) {
            case GeomAbs_Plane: ++result.surfaces.planes; break;
            case GeomAbs_Cylinder: ++result.surfaces.cylinders; break;
            case GeomAbs_Cone: ++result.surfaces.cones; break;
            case GeomAbs_Sphere: ++result.surfaces.spheres; break;
            case GeomAbs_Torus: ++result.surfaces.tori; break;
            case GeomAbs_BezierSurface: ++result.surfaces.bezier; break;
            case GeomAbs_BSplineSurface: ++result.surfaces.bspline; break;
            case GeomAbs_SurfaceOfRevolution: ++result.surfaces.revolutions; break;
            case GeomAbs_SurfaceOfExtrusion: ++result.surfaces.extrusions; break;
            case GeomAbs_OffsetSurface: ++result.surfaces.offsets; break;
            default: ++result.surfaces.other; break;
            }
        } catch (const Standard_Failure&) {
            ++result.surfaces.other;
        }
    }

    try {
        GProp_GProps volumeProperties;
        GProp_GProps surfaceProperties;
        BRepGProp::VolumeProperties(shape, volumeProperties, 1.0e-9);
        BRepGProp::SurfaceProperties(shape, surfaceProperties, 1.0e-9);
        result.volume = std::abs(volumeProperties.Mass());
        result.surfaceArea = std::abs(surfaceProperties.Mass());
    } catch (const Standard_Failure& error) {
        diagnostics.emplace_back(std::string(description) +
                                 " mass-property calculation failed: " + error.what());
    }

    try {
        Bnd_Box box;
        BRepBndLib::AddOptimal(shape, box, false, false);
        if (!box.IsVoid()) {
            double xmin = 0.0;
            double ymin = 0.0;
            double zmin = 0.0;
            double xmax = 0.0;
            double ymax = 0.0;
            double zmax = 0.0;
            box.Get(xmin, ymin, zmin, xmax, ymax, zmax);
            result.bounds = {{xmin, ymin, zmin}, {xmax, ymax, zmax}};
        }
    } catch (const Standard_Failure& error) {
        diagnostics.emplace_back(std::string(description) +
                                 " bounding-box calculation failed: " + error.what());
    }

    if (!result.valid) {
        diagnostics.emplace_back(std::string(description) +
                                 " shape failed BRepCheck_Analyzer");
    }
    return result;
}

TopologyCountComparison compareTopology(const ValidationShapeMetrics& reconstructed,
                                        const ValidationShapeMetrics& reference) {
    TopologyCountComparison result;
    result.solidDelta = countDelta(reconstructed.solidCount, reference.solidCount);
    result.shellDelta = countDelta(reconstructed.shellCount, reference.shellCount);
    result.faceDelta = countDelta(reconstructed.faceCount, reference.faceCount);
    result.edgeDelta = countDelta(reconstructed.edgeCount, reference.edgeCount);
    result.vertexDelta = countDelta(reconstructed.vertexCount, reference.vertexCount);
    result.planeDelta = countDelta(reconstructed.surfaces.planes, reference.surfaces.planes);
    result.cylinderDelta = countDelta(
        reconstructed.surfaces.cylinders, reference.surfaces.cylinders);
    result.coneDelta = countDelta(reconstructed.surfaces.cones, reference.surfaces.cones);
    result.sphereDelta = countDelta(reconstructed.surfaces.spheres, reference.surfaces.spheres);
    result.torusDelta = countDelta(reconstructed.surfaces.tori, reference.surfaces.tori);
    result.bezierDelta = countDelta(reconstructed.surfaces.bezier, reference.surfaces.bezier);
    result.bsplineDelta = countDelta(reconstructed.surfaces.bspline, reference.surfaces.bspline);
    result.revolutionDelta = countDelta(
        reconstructed.surfaces.revolutions, reference.surfaces.revolutions);
    result.extrusionDelta = countDelta(
        reconstructed.surfaces.extrusions, reference.surfaces.extrusions);
    result.offsetDelta = countDelta(reconstructed.surfaces.offsets, reference.surfaces.offsets);
    result.otherSurfaceDelta = countDelta(
        reconstructed.surfaces.other, reference.surfaces.other);

    result.exactMatch = result.solidDelta == 0 && result.shellDelta == 0 &&
                        result.faceDelta == 0 && result.edgeDelta == 0 &&
                        result.vertexDelta == 0 && result.planeDelta == 0 &&
                        result.cylinderDelta == 0 && result.coneDelta == 0 &&
                        result.sphereDelta == 0 && result.torusDelta == 0 &&
                        result.bezierDelta == 0 && result.bsplineDelta == 0 &&
                        result.revolutionDelta == 0 && result.extrusionDelta == 0 &&
                        result.offsetDelta == 0 && result.otherSurfaceDelta == 0;
    return result;
}

PropertyComparison compareProperties(const ValidationShapeMetrics& reconstructed,
                                     const ValidationShapeMetrics& reference,
                                     double comparisonScale) {
    PropertyComparison result;
    if (reconstructed.bounds.valid() && reference.bounds.valid()) {
        const Vec3d minimumDelta = reconstructed.bounds.minimum - reference.bounds.minimum;
        const Vec3d maximumDelta = reconstructed.bounds.maximum - reference.bounds.maximum;
        result.boundsMinimumCornerDistance = norm(minimumDelta);
        result.boundsMaximumCornerDistance = norm(maximumDelta);
        result.boundsCenterDistance = norm(
            reconstructed.bounds.center() - reference.bounds.center());
        result.boundsMaximumCoordinateError = std::max(
            maximumCoordinateMagnitude(minimumDelta), maximumCoordinateMagnitude(maximumDelta));
        result.boundsDiagonalRelativeError = safeRelativeError(
            reconstructed.bounds.diagonal(),
            reference.bounds.diagonal(),
            comparisonScale * std::numeric_limits<double>::epsilon());
    }

    result.volumeAbsoluteError = std::abs(reconstructed.volume - reference.volume);
    result.volumeRelativeError = safeRelativeError(
        reconstructed.volume,
        reference.volume,
        std::pow(comparisonScale, 3.0) * std::numeric_limits<double>::epsilon());
    result.surfaceAreaAbsoluteError =
        std::abs(reconstructed.surfaceArea - reference.surfaceArea);
    result.surfaceAreaRelativeError = safeRelativeError(
        reconstructed.surfaceArea,
        reference.surfaceArea,
        comparisonScale * comparisonScale * std::numeric_limits<double>::epsilon());
    return result;
}

double radicalInverse(std::size_t index, std::size_t base) {
    double result = 0.0;
    double factor = 1.0 / static_cast<double>(base);
    while (index > 0) {
        result += factor * static_cast<double>(index % base);
        index /= base;
        factor /= static_cast<double>(base);
    }
    return result;
}

Vec3d surfacePoint(const SurfaceTriangle& triangle,
                   double wa,
                   double wb,
                   double wc) {
    // Interpolating UV coordinates can cross a periodic seam or a trimmed-out
    // hole. Project the chordal sample onto its own trimmed face instead.
    const Vec3d chord = triangle.a * wa + triangle.b * wb + triangle.c * wc;
    const auto vertex = BRepBuilderAPI_MakeVertex(toPoint(chord)).Vertex();
    BRepExtrema_DistShapeShape projection(vertex, triangle.face);
    if (!projection.IsDone() || projection.NbSolution() == 0) {
        throw std::runtime_error("Cannot project a validation sample onto its source face");
    }
    return toVec(projection.PointOnShape2(1));
}

Vec3d areaUniformPoint(const SurfaceTriangle& triangle, std::size_t sequenceIndex) {
    const double first = radicalInverse(sequenceIndex + 1U, 2U);
    const double second = radicalInverse(sequenceIndex + 1U, 3U);
    const double root = std::sqrt(first);
    const double wa = 1.0 - root;
    const double wb = root * (1.0 - second);
    const double wc = root * second;
    return surfacePoint(triangle, wa, wb, wc);
}

SampleSet sampleShape(const TopoDS_Shape& shape,
                      double linearDeflection,
                      double angularDeflection,
                      std::size_t maximumSamples) {
    // Mesh a topology/geometry copy so validation cannot replace or clear the
    // reconstruction's triangulation cache.
    BRepBuilderAPI_Copy copy(shape, true, false);
    TopoDS_Shape meshedShape = copy.Shape();
    if (meshedShape.IsNull()) {
        throw std::runtime_error("OpenCascade could not copy a shape for validation meshing");
    }
    BRepTools::Clean(meshedShape, true);
    BRepMesh_IncrementalMesh mesher(
        meshedShape, linearDeflection, false, angularDeflection, false);
    if (!mesher.IsDone()) {
        throw std::runtime_error("OpenCascade validation meshing failed");
    }

    std::vector<SurfaceTriangle> triangles;
    for (TopExp_Explorer explorer(meshedShape, TopAbs_FACE);
         explorer.More();
         explorer.Next()) {
        const TopoDS_Face face = TopoDS::Face(explorer.Current());
        TopLoc_Location location;
        const Handle(Poly_Triangulation) triangulation =
            BRep_Tool::Triangulation(face, location);
        if (triangulation.IsNull()) continue;
        const gp_Trsf transform = location.Transformation();
        const bool hasUv = triangulation->HasUVNodes();
        for (int triangleIndex = 1;
             triangleIndex <= triangulation->NbTriangles();
             ++triangleIndex) {
            int ia = 0;
            int ib = 0;
            int ic = 0;
            triangulation->Triangle(triangleIndex).Get(ia, ib, ic);
            const Vec3d a = toVec(triangulation->Node(ia).Transformed(transform));
            const Vec3d b = toVec(triangulation->Node(ib).Transformed(transform));
            const Vec3d c = toVec(triangulation->Node(ic).Transformed(transform));
            const double area = 0.5 * norm(cross(b - a, c - a));
            if (std::isfinite(area) && area > 0.0) {
                std::array<gp_Pnt2d, 3> uv{};
                if (hasUv) {
                    uv = {triangulation->UVNode(ia),
                          triangulation->UVNode(ib),
                          triangulation->UVNode(ic)};
                }
                triangles.push_back({a, b, c, face, uv, hasUv, area});
            }
        }
    }
    if (triangles.empty()) {
        throw std::runtime_error("Validation meshing produced no usable surface triangles");
    }

    SampleSet result;
    result.candidateTriangleCount = triangles.size();
    for (const SurfaceTriangle& triangle : triangles) result.area += triangle.area;
    if (!(result.area > 0.0) || !std::isfinite(result.area)) {
        throw std::runtime_error("Validation surface area is invalid");
    }

    if (triangles.size() <= maximumSamples) {
        result.samples.reserve(triangles.size());
        for (const SurfaceTriangle& triangle : triangles) {
            result.samples.push_back({
                surfacePoint(triangle, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
                triangle.area});
        }
        return result;
    }

    result.capped = true;
    result.samples.reserve(maximumSamples);
    const double stratumArea = result.area / static_cast<double>(maximumSamples);
    std::size_t triangleIndex = 0;
    double cumulativeArea = triangles.front().area;
    for (std::size_t sampleIndex = 0; sampleIndex < maximumSamples; ++sampleIndex) {
        const double targetArea =
            (static_cast<double>(sampleIndex) + 0.5) * stratumArea;
        while (triangleIndex + 1U < triangles.size() && cumulativeArea < targetArea) {
            ++triangleIndex;
            cumulativeArea += triangles[triangleIndex].area;
        }
        result.samples.push_back({
            areaUniformPoint(triangles[triangleIndex], sampleIndex),
            stratumArea});
    }
    return result;
}

TopoDS_Compound faceCompound(const TopoDS_Shape& shape) {
    BRep_Builder builder;
    TopoDS_Compound compound;
    builder.MakeCompound(compound);
    for (TopExp_Explorer explorer(shape, TopAbs_FACE); explorer.More(); explorer.Next()) {
        builder.Add(compound, explorer.Current());
    }
    return compound;
}

void appendBoundarySamples(SampleSet& samples, const TopoDS_Shape& shape,
                           std::size_t limit) {
    ShapeMap edges;
    TopExp::MapShapes(shape, TopAbs_EDGE, edges);
    std::vector<Vec3d> candidates;
    for (int i = 1; i <= edges.Extent(); ++i) {
        const auto edge = TopoDS::Edge(edges(i));
        if (BRep_Tool::Degenerated(edge)) continue;
        const BRepAdaptor_Curve curve(edge);
        const double first = curve.FirstParameter();
        const double last = curve.LastParameter();
        if (!std::isfinite(first) || !std::isfinite(last)) continue;
        for (const double fraction : {0.0, 0.25, 0.5, 0.75, 1.0}) {
            candidates.push_back(toVec(curve.Value(first + (last - first) * fraction)));
        }
    }
    const std::size_t count = std::min(limit, candidates.size());
    for (std::size_t i = 0; i < count; ++i) {
        const auto index = std::min(candidates.size() - 1U,
            static_cast<std::size_t>((static_cast<double>(i) + 0.5) *
                                    candidates.size() / count));
        // Boundary samples contribute to maximum and sample count. Surface
        // RMS remains area weighted; an edge has zero surface-area measure.
        samples.samples.push_back({candidates[index], 0.0});
    }
}

SampledSurfaceDistance measureSamples(const SampleSet& samples,
                                      const TopoDS_Shape& target,
                                      double comparisonScale,
                                      std::size_t requestedLimit) {
    SampledSurfaceDistance result;
    result.candidateTriangleCount = samples.candidateTriangleCount;
    result.requestedSampleLimit = requestedLimit;
    result.sampleCapApplied = samples.capped;
    result.sampledArea = samples.area;

    const TopoDS_Compound boundary = faceCompound(target);
    if (countSubshapes(boundary, TopAbs_FACE) == 0) {
        throw std::runtime_error("Distance target has no boundary faces");
    }

    double weightedMean = 0.0;
    double weightedSquares = 0.0;
    double successfulWeight = 0.0;
    BRepExtrema_DistShapeShape query;
    query.LoadS2(boundary);
    for (const SurfaceSample& sample : samples.samples) {
        try {
            const TopoDS_Vertex vertex = BRepBuilderAPI_MakeVertex(toPoint(sample.point));
            query.LoadS1(vertex);
            query.Perform();
            if (!query.IsDone() || query.NbSolution() == 0) {
                ++result.failedSampleCount;
                continue;
            }
            const double distance = query.Value();
            if (!std::isfinite(distance)) {
                ++result.failedSampleCount;
                continue;
            }
            ++result.sampleCount;
            successfulWeight += sample.weight;
            weightedMean += sample.weight * distance;
            weightedSquares += sample.weight * distance * distance;
            if (distance > result.maximum) {
                result.maximum = distance;
                result.maximumPoint = sample.point;
                result.maximumOnBoundary = sample.weight == 0.0;
            }
        } catch (const Standard_Failure&) {
            ++result.failedSampleCount;
        }
    }
    if (result.sampleCount == 0 || !(successfulWeight > 0.0)) {
        throw std::runtime_error("All point-to-shape validation queries failed");
    }

    result.mean = weightedMean / successfulWeight;
    result.rms = std::sqrt(weightedSquares / successfulWeight);
    result.normalizedMean = result.mean / comparisonScale;
    result.normalizedRms = result.rms / comparisonScale;
    result.normalizedMaximum = result.maximum / comparisonScale;
    return result;
}

bool validOptions(const ReferenceValidationOptions& options, std::string& reason) {
    if (!(options.referenceLinearScale > 0.0) ||
        !std::isfinite(options.referenceLinearScale)) {
        reason = "referenceLinearScale must be finite and positive";
        return false;
    }
    if (options.meshingLinearDeflection < 0.0 ||
        !std::isfinite(options.meshingLinearDeflection)) {
        reason = "meshingLinearDeflection must be finite and non-negative";
        return false;
    }
    if (!(options.meshingLinearDeflectionRelative > 0.0) ||
        !std::isfinite(options.meshingLinearDeflectionRelative)) {
        reason = "meshingLinearDeflectionRelative must be finite and positive";
        return false;
    }
    if (!(options.meshingAngularDeflectionRadians > 0.0) ||
        !std::isfinite(options.meshingAngularDeflectionRadians)) {
        reason = "meshingAngularDeflectionRadians must be finite and positive";
        return false;
    }
    if (options.maximumSamplesPerDirection == 0) {
        reason = "maximumSamplesPerDirection must be positive";
        return false;
    }
    return true;
}

const char* readStatusName(IFSelect_ReturnStatus status) {
    switch (status) {
    case IFSelect_RetVoid: return "void";
    case IFSelect_RetDone: return "done";
    case IFSelect_RetError: return "error";
    case IFSelect_RetFail: return "fail";
    case IFSelect_RetStop: return "stop";
    }
    return "unknown";
}

} // namespace

ReferenceValidationResult compareShapesForValidation(
    const TopoDS_Shape& reconstructed,
    const TopoDS_Shape& reference,
    const ReferenceValidationOptions& options) {
    ReferenceValidationResult result;
    std::string optionError;
    if (!validOptions(options, optionError)) {
        result.diagnostics.push_back("Invalid reference validation options: " + optionError);
        return result;
    }
    if (reconstructed.IsNull() || reference.IsNull()) {
        result.diagnostics.emplace_back(
            "Reference validation requires non-null reconstructed and reference shapes");
        return result;
    }

    result.referenceLoaded = true;
    try {
        result.reconstructed = inspectShape(
            reconstructed, result.diagnostics, "Reconstructed");
        result.reference = inspectShape(reference, result.diagnostics, "Reference");
    } catch (const Standard_Failure& error) {
        result.diagnostics.push_back(
            std::string("OpenCascade shape inspection failed: ") + error.what());
        return result;
    } catch (const std::exception& error) {
        result.diagnostics.push_back(
            std::string("Shape inspection failed: ") + error.what());
        return result;
    }
    const double reconstructedScale = result.reconstructed.bounds.valid()
        ? result.reconstructed.bounds.diagonal()
        : 0.0;
    const double referenceScale = result.reference.bounds.valid()
        ? result.reference.bounds.diagonal()
        : 0.0;
    result.comparisonScale = std::max({
        reconstructedScale,
        referenceScale,
        Precision::Confusion()});
    result.meshingLinearDeflection = options.meshingLinearDeflection > 0.0
        ? options.meshingLinearDeflection
        : std::max(Precision::Confusion(),
                   result.comparisonScale * options.meshingLinearDeflectionRelative);

    result.topology = compareTopology(result.reconstructed, result.reference);
    result.properties = compareProperties(
        result.reconstructed, result.reference, result.comparisonScale);
    result.propertiesCompared = true;
    if (!result.topology.exactMatch) {
        result.diagnostics.emplace_back(
            "Reconstructed and reference topology/surface counts differ");
    }

    try {
        SampleSet reconstructedSamples = sampleShape(
            reconstructed,
            result.meshingLinearDeflection,
            options.meshingAngularDeflectionRadians,
            options.maximumSamplesPerDirection);
        SampleSet referenceSamples = sampleShape(
            reference,
            result.meshingLinearDeflection,
            options.meshingAngularDeflectionRadians,
            options.maximumSamplesPerDirection);
        appendBoundarySamples(reconstructedSamples, reconstructed,
                              options.maximumSamplesPerDirection);
        appendBoundarySamples(referenceSamples, reference,
                              options.maximumSamplesPerDirection);
        result.distances.reconstructedToReference = measureSamples(
            reconstructedSamples,
            reference,
            result.comparisonScale,
            options.maximumSamplesPerDirection);
        result.distances.referenceToReconstructed = measureSamples(
            referenceSamples,
            reconstructed,
            result.comparisonScale,
            options.maximumSamplesPerDirection);

        const double forwardRms = result.distances.reconstructedToReference.rms;
        const double reverseRms = result.distances.referenceToReconstructed.rms;
        result.distances.symmetricRms = std::sqrt(
            0.5 * (forwardRms * forwardRms + reverseRms * reverseRms));
        result.distances.symmetricMaximum = std::max(
            result.distances.reconstructedToReference.maximum,
            result.distances.referenceToReconstructed.maximum);
        result.distances.normalizedSymmetricRms =
            result.distances.symmetricRms / result.comparisonScale;
        result.distances.normalizedSymmetricMaximum =
            result.distances.symmetricMaximum / result.comparisonScale;
        result.distancesCompared = true;

        if (reconstructedSamples.capped || referenceSamples.capped) {
            result.diagnostics.emplace_back(
                "Surface-distance sampling was deterministically capped for runtime");
        }
        const std::size_t failed =
            result.distances.reconstructedToReference.failedSampleCount +
            result.distances.referenceToReconstructed.failedSampleCount;
        if (failed > 0) {
            result.diagnostics.push_back(
                std::to_string(failed) + " point-to-shape distance sample(s) failed");
        }
    } catch (const Standard_Failure& error) {
        result.diagnostics.push_back(
            std::string("OpenCascade surface-distance validation failed: ") + error.what());
    } catch (const std::exception& error) {
        result.diagnostics.push_back(
            std::string("Surface-distance validation failed: ") + error.what());
    }
    return result;
}

ReferenceValidationResult validateAgainstStepReference(
    const TopoDS_Shape& reconstructed,
    const std::filesystem::path& stepPath,
    const ReferenceValidationOptions& options) {
    ReferenceValidationResult failure;
    failure.referencePath = stepPath;

    std::string optionError;
    if (!validOptions(options, optionError)) {
        failure.diagnostics.push_back("Invalid reference validation options: " + optionError);
        return failure;
    }
    std::error_code filesystemError;
    const bool exists = std::filesystem::exists(stepPath, filesystemError);
    const bool regular = exists &&
        std::filesystem::is_regular_file(stepPath, filesystemError);
    if (stepPath.empty() || filesystemError || !regular) {
        failure.diagnostics.push_back(
            "STEP reference does not exist or is not a regular file: " + stepPath.string());
        return failure;
    }

    try {
        STEPControl_Reader reader;
        const std::string nativePath = stepPath.string();
        const IFSelect_ReturnStatus status = reader.ReadFile(nativePath.c_str());
        if (status != IFSelect_RetDone) {
            failure.diagnostics.push_back(
                std::string("OpenCascade could not read the STEP reference (status: ") +
                readStatusName(status) + ")");
            return failure;
        }
        int transferredRoots = 0;
        TopoDS_Shape reference = transferCsgReference(reader.StepModel(), transferredRoots);
        if (transferredRoots == 0) {
            transferredRoots = reader.TransferRoots();
            reference = reader.OneShape();
        }
        if (transferredRoots <= 0) {
            failure.diagnostics.emplace_back(
                "OpenCascade read the STEP file but transferred no roots");
            return failure;
        }
        if (reference.IsNull()) {
            failure.diagnostics.emplace_back(
                "OpenCascade STEP transfer returned a null shape");
            return failure;
        }

        if (options.referenceLinearScale != 1.0) {
            gp_Trsf scale;
            scale.SetScale(gp_Pnt(0.0, 0.0, 0.0), options.referenceLinearScale);
            BRepBuilderAPI_Transform transform(reference, scale, true, false);
            if (!transform.IsDone() || transform.Shape().IsNull()) {
                failure.diagnostics.emplace_back(
                    "OpenCascade could not apply the requested reference scale");
                return failure;
            }
            reference = transform.Shape();
        }
        if (options.referenceZUpToGltf) {
            gp_Trsf rotation;
            rotation.SetRotation(gp_Ax1(gp_Pnt(0.0, 0.0, 0.0),
                                        gp_Dir(1.0, 0.0, 0.0)),
                                 -1.57079632679489661923);
            reference = BRepBuilderAPI_Transform(reference, rotation, true).Shape();
        }

        ReferenceValidationResult result = compareShapesForValidation(
            reconstructed, reference, options);
        result.referencePath = stepPath;
        result.referenceLoaded = true;
        result.diagnostics.insert(
            result.diagnostics.begin(),
            "Loaded and transferred " + std::to_string(transferredRoots) +
                " STEP reference root(s)");
        return result;
    } catch (const Standard_Failure& error) {
        failure.diagnostics.push_back(
            std::string("OpenCascade STEP validation failed: ") + error.what());
    } catch (const std::exception& error) {
        failure.diagnostics.push_back(
            std::string("STEP reference validation failed: ") + error.what());
    }
    return failure;
}

} // namespace recad
