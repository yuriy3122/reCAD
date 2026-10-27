#include "CadReconstruction.h"

#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepGProp.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRep_Tool.hxx>
#include <BRepTools.hxx>
#include <DESTEP_Parameters.hxx>
#include <GProp_GProps.hxx>
#include <GeomAbs_CurveType.hxx>
#include <GeomAbs_SurfaceType.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <NCollection_IndexedMap.hxx>
#include <Standard_Failure.hxx>
#include <STEPControl_StepModelType.hxx>
#include <STEPControl_Writer.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_ShapeMapHasher.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <gp_Ax2.hxx>
#include <gp_Dir.hxx>
#include <gp_Pnt.hxx>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <system_error>

namespace recad {
namespace {

using ShapeMap = NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher>;

std::size_t countSubshapes(const TopoDS_Shape& shape, TopAbs_ShapeEnum kind) {
    ShapeMap shapes;
    TopExp::MapShapes(shape, kind, shapes);
    return static_cast<std::size_t>(shapes.Extent());
}

double boundsError(const Aabb& lhs, const Aabb& rhs) {
    return norm(lhs.minimum - rhs.minimum) + norm(lhs.maximum - rhs.maximum);
}

template <typename Algorithm>
TopoDS_Shape runBoolean(Algorithm& algorithm, double tolerance, std::string& diagnostic) {
    algorithm.SetNonDestructive(true);
    algorithm.SetRunParallel(true);
    algorithm.SetFuzzyValue(tolerance);
    algorithm.Build();
    if (!algorithm.IsDone()) {
        diagnostic = "OpenCascade Boolean builder did not complete";
        return {};
    }
    TopoDS_Shape result = algorithm.Shape();
    if (result.IsNull()) diagnostic = "OpenCascade Boolean returned a null shape";
    return result;
}

} // namespace

const char* toString(BooleanOperation operation) noexcept {
    switch (operation) {
    case BooleanOperation::Cut: return "CUT";
    case BooleanOperation::Fuse: return "FUSE";
    case BooleanOperation::Common: return "COMMON";
    }
    return "UNKNOWN";
}

ShapeProperties inspectShape(const TopoDS_Shape& shape) {
    ShapeProperties result;
    if (shape.IsNull()) return result;

    result.valid = BRepCheck_Analyzer(shape, true).IsValid();
    result.solidCount = countSubshapes(shape, TopAbs_SOLID);
    result.shellCount = countSubshapes(shape, TopAbs_SHELL);
    result.faceCount = countSubshapes(shape, TopAbs_FACE);
    result.edgeCount = countSubshapes(shape, TopAbs_EDGE);
    result.vertexCount = countSubshapes(shape, TopAbs_VERTEX);
    result.closedSolid = result.valid && result.solidCount > 0 && result.shellCount > 0;
    ShapeMap shells;
    TopExp::MapShapes(shape, TopAbs_SHELL, shells);
    for (int index = 1; index <= shells.Extent(); ++index) {
        if (!BRep_Tool::IsClosed(shells(index))) result.closedSolid = false;
    }

    ShapeMap faces;
    TopExp::MapShapes(shape, TopAbs_FACE, faces);
    for (int index = 1; index <= faces.Extent(); ++index) {
        const BRepAdaptor_Surface surface(TopoDS::Face(faces(index)), true);
        switch (surface.GetType()) {
        case GeomAbs_Plane: ++result.planarFaceCount; break;
        case GeomAbs_Cylinder: ++result.cylindricalFaceCount; break;
        case GeomAbs_Cone: ++result.conicalFaceCount; break;
        case GeomAbs_Sphere: ++result.sphericalFaceCount; break;
        case GeomAbs_Torus: ++result.toroidalFaceCount; break;
        case GeomAbs_BSplineSurface:
        case GeomAbs_BezierSurface: ++result.bsplineFaceCount; break;
        default: ++result.otherSurfaceCount; break;
        }
    }
    ShapeMap edges;
    TopExp::MapShapes(shape, TopAbs_EDGE, edges);
    for (int index = 1; index <= edges.Extent(); ++index) {
        const BRepAdaptor_Curve curve(TopoDS::Edge(edges(index)));
        switch (curve.GetType()) {
        case GeomAbs_Line: ++result.lineEdgeCount; break;
        case GeomAbs_Circle: ++result.circleEdgeCount; break;
        case GeomAbs_BSplineCurve:
        case GeomAbs_BezierCurve: ++result.bsplineEdgeCount; break;
        default: ++result.otherCurveCount; break;
        }
    }

    GProp_GProps volumeProperties;
    GProp_GProps surfaceProperties;
    BRepGProp::VolumeProperties(shape, volumeProperties);
    BRepGProp::SurfaceProperties(shape, surfaceProperties);
    result.volume = volumeProperties.Mass();
    result.surfaceArea = surfaceProperties.Mass();
    result.bounds = brepBounds(shape);
    return result;
}

CadReconstructionResult buildAndSelectBoolean(
    const OrientedBoxFitResult& box,
    const SphereFitResult& sphere,
    const Mesh& inputMesh,
    double modelingTolerance,
    double comparisonDeflection) {
    if (!box.valid) throw std::invalid_argument("Cannot construct CAD from an invalid box fit");
    if (!sphere.valid || !(sphere.radius > 0.0)) {
        throw std::invalid_argument("Cannot construct CAD from an invalid sphere fit");
    }

    Vec3d xAxis = normalized(box.axes[0]);
    Vec3d yAxis = normalized(box.axes[1] - xAxis * dot(box.axes[1], xAxis));
    Vec3d zAxis = normalized(cross(xAxis, yAxis));
    if (dot(zAxis, box.axes[2]) < 0.0) {
        yAxis = -yAxis;
        zAxis = -zAxis;
    }

    const Vec3d corner = box.center - xAxis * (box.dimensions.x * 0.5)
                                      - yAxis * (box.dimensions.y * 0.5)
                                      - zAxis * (box.dimensions.z * 0.5);
    const gp_Ax2 frame(gp_Pnt(corner.x, corner.y, corner.z),
                       gp_Dir(zAxis.x, zAxis.y, zAxis.z),
                       gp_Dir(xAxis.x, xAxis.y, xAxis.z));

    CadReconstructionResult result;
    result.boxShape = BRepPrimAPI_MakeBox(
        frame, box.dimensions.x, box.dimensions.y, box.dimensions.z).Shape();
    result.sphereShape = BRepPrimAPI_MakeSphere(
        gp_Pnt(sphere.center.x, sphere.center.y, sphere.center.z), sphere.radius).Shape();

    result.candidates.reserve(3);
    for (const BooleanOperation operation : {
             BooleanOperation::Cut, BooleanOperation::Fuse, BooleanOperation::Common}) {
        BooleanCandidate candidate;
        candidate.operation = operation;
        candidate.score = std::numeric_limits<double>::infinity();
        try {
            if (operation == BooleanOperation::Cut) {
                BRepAlgoAPI_Cut algorithm(result.boxShape, result.sphereShape);
                candidate.shape = runBoolean(algorithm, modelingTolerance, candidate.diagnostic);
            } else if (operation == BooleanOperation::Fuse) {
                BRepAlgoAPI_Fuse algorithm(result.boxShape, result.sphereShape);
                candidate.shape = runBoolean(algorithm, modelingTolerance, candidate.diagnostic);
            } else {
                BRepAlgoAPI_Common algorithm(result.boxShape, result.sphereShape);
                candidate.shape = runBoolean(algorithm, modelingTolerance, candidate.diagnostic);
            }

            candidate.buildSucceeded = !candidate.shape.IsNull();
            if (!candidate.buildSucceeded) {
                result.candidates.push_back(std::move(candidate));
                continue;
            }
            candidate.properties = inspectShape(candidate.shape);
            if (!candidate.properties.valid || !candidate.properties.closedSolid ||
                candidate.properties.solidCount != 1 ||
                candidate.properties.otherSurfaceCount != 0) {
                candidate.diagnostic =
                    "Candidate is not a closed, valid, single analytic solid";
                result.candidates.push_back(std::move(candidate));
                continue;
            }
            candidate.distances = compareMeshAndBrep(
                inputMesh, candidate.shape, comparisonDeflection);
            candidate.boundingBoxError = boundsError(candidate.properties.bounds, inputMesh.bounds);
            const double scale = std::max(
                inputMesh.bounds.diagonal(), std::numeric_limits<double>::min());
            const double symmetricMaximum = std::max(
                candidate.distances.meshToBrep.maximum,
                candidate.distances.brepToMesh.maximum);
            candidate.score = candidate.distances.symmetricRms / scale
                            + 0.10 * symmetricMaximum / scale
                            + 0.25 * candidate.boundingBoxError / scale;
        } catch (const Standard_Failure& failure) {
            const char* message = failure.what();
            candidate.diagnostic = message != nullptr ? message : "OpenCascade exception";
            candidate.score = std::numeric_limits<double>::infinity();
        } catch (const std::exception& exception) {
            candidate.diagnostic = exception.what();
            candidate.score = std::numeric_limits<double>::infinity();
        }
        result.candidates.push_back(std::move(candidate));
    }

    const auto selected = std::min_element(
        result.candidates.begin(), result.candidates.end(),
        [](const BooleanCandidate& lhs, const BooleanCandidate& rhs) {
            return lhs.score < rhs.score;
        });
    if (selected == result.candidates.end() || !std::isfinite(selected->score)) {
        throw std::runtime_error("All OpenCascade Boolean candidates failed");
    }
    result.selectedCandidate = static_cast<std::size_t>(selected - result.candidates.begin());
    result.selectedShape = selected->shape;
    return result;
}

void writeBrepAndStep(
    const TopoDS_Shape& shape,
    const std::filesystem::path& brepPath,
    const std::filesystem::path& stepPath) {
    std::error_code error;
    std::filesystem::create_directories(brepPath.parent_path(), error);
    if (error) throw std::runtime_error("Could not create BREP output directory: " + error.message());
    std::filesystem::create_directories(stepPath.parent_path(), error);
    if (error) throw std::runtime_error("Could not create STEP output directory: " + error.message());

    if (!BRepTools::Write(shape, brepPath.string().c_str())) {
        throw std::runtime_error("OpenCascade failed to write " + brepPath.string());
    }

    STEPControl_Writer writer;
    DESTEP_Parameters parameters;
    parameters.WriteUnit = UnitsMethods_LengthUnit_Millimeter;
    if (writer.Transfer(shape, STEPControl_AsIs, parameters) != IFSelect_RetDone) {
        throw std::runtime_error("OpenCascade failed to transfer shape to STEP");
    }
    if (writer.Write(stepPath.string().c_str()) != IFSelect_RetDone) {
        throw std::runtime_error("OpenCascade failed to write " + stepPath.string());
    }
}

} // namespace recad
