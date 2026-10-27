#pragma once

#include "Distance.h"
#include "Geometry.h"

#include <TopoDS_Shape.hxx>

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace recad {

enum class BooleanOperation { Cut, Fuse, Common };

[[nodiscard]] const char* toString(BooleanOperation operation) noexcept;

struct ShapeProperties {
    bool valid{};
    bool closedSolid{};
    std::size_t solidCount{};
    std::size_t shellCount{};
    std::size_t faceCount{};
    std::size_t edgeCount{};
    std::size_t vertexCount{};
    std::size_t planarFaceCount{};
    std::size_t cylindricalFaceCount{};
    std::size_t conicalFaceCount{};
    std::size_t sphericalFaceCount{};
    std::size_t toroidalFaceCount{};
    std::size_t bsplineFaceCount{};
    std::size_t otherSurfaceCount{};
    std::size_t lineEdgeCount{};
    std::size_t circleEdgeCount{};
    std::size_t bsplineEdgeCount{};
    std::size_t otherCurveCount{};
    double volume{};
    double surfaceArea{};
    Aabb bounds;
};

struct BooleanCandidate {
    BooleanOperation operation{BooleanOperation::Cut};
    TopoDS_Shape shape;
    bool buildSucceeded{};
    std::string diagnostic;
    ShapeProperties properties;
    BidirectionalDistance distances;
    double boundingBoxError{};
    double score{}; // Lower is better.
};

struct CadReconstructionResult {
    TopoDS_Shape boxShape;
    TopoDS_Shape sphereShape;
    std::vector<BooleanCandidate> candidates;
    std::size_t selectedCandidate{};
    TopoDS_Shape selectedShape;
};

[[nodiscard]] ShapeProperties inspectShape(const TopoDS_Shape& shape);

[[nodiscard]] CadReconstructionResult buildAndSelectBoolean(
    const OrientedBoxFitResult& box,
    const SphereFitResult& sphere,
    const Mesh& inputMesh,
    double modelingTolerance,
    double comparisonDeflection);

void writeBrepAndStep(
    const TopoDS_Shape& shape,
    const std::filesystem::path& brepPath,
    const std::filesystem::path& stepPath);

} // namespace recad

