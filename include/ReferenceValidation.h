#pragma once

#include "Geometry.h"

#include <TopoDS_Shape.hxx>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace recad {

struct SurfaceTypeCounts {
    std::size_t planes{};
    std::size_t cylinders{};
    std::size_t cones{};
    std::size_t spheres{};
    std::size_t tori{};
    std::size_t bezier{};
    std::size_t bspline{};
    std::size_t revolutions{};
    std::size_t extrusions{};
    std::size_t offsets{};
    std::size_t other{};
};

struct ValidationShapeMetrics {
    bool valid{};
    bool closed{};
    std::size_t solidCount{};
    std::size_t shellCount{};
    std::size_t closedShellCount{};
    std::size_t faceCount{};
    std::size_t edgeCount{};
    std::size_t vertexCount{};
    SurfaceTypeCounts surfaces;
    double volume{};
    double surfaceArea{};
    Aabb bounds;
};

struct TopologyCountComparison {
    bool exactMatch{};
    std::int64_t solidDelta{};
    std::int64_t shellDelta{};
    std::int64_t faceDelta{};
    std::int64_t edgeDelta{};
    std::int64_t vertexDelta{};
    std::int64_t planeDelta{};
    std::int64_t cylinderDelta{};
    std::int64_t coneDelta{};
    std::int64_t sphereDelta{};
    std::int64_t torusDelta{};
    std::int64_t bezierDelta{};
    std::int64_t bsplineDelta{};
    std::int64_t revolutionDelta{};
    std::int64_t extrusionDelta{};
    std::int64_t offsetDelta{};
    std::int64_t otherSurfaceDelta{};
};

struct PropertyComparison {
    double boundsMinimumCornerDistance{};
    double boundsMaximumCornerDistance{};
    double boundsCenterDistance{};
    double boundsMaximumCoordinateError{};
    double boundsDiagonalRelativeError{};
    double volumeAbsoluteError{};
    double volumeRelativeError{};
    double surfaceAreaAbsoluteError{};
    double surfaceAreaRelativeError{};
};

struct SampledSurfaceDistance {
    std::size_t candidateTriangleCount{};
    std::size_t requestedSampleLimit{};
    std::size_t sampleCount{};
    std::size_t failedSampleCount{};
    bool sampleCapApplied{};
    double sampledArea{};
    double mean{};
    double rms{};
    double maximum{};
    double normalizedMean{};
    double normalizedRms{};
    double normalizedMaximum{};
};

struct BidirectionalSurfaceDistance {
    SampledSurfaceDistance reconstructedToReference;
    SampledSurfaceDistance referenceToReconstructed;
    double symmetricRms{};
    double symmetricMaximum{};
    double normalizedSymmetricRms{};
    double normalizedSymmetricMaximum{};
};

struct ReferenceValidationOptions {
    // Optional explicit scaling about the global origin after STEP transfer.
    // OpenCascade normally transfers STEP into its configured system units, so
    // this should remain 1 unless the caller intentionally uses other units.
    double referenceLinearScale{1.0};
    // Standard CAD Z-up to glTF Y-up conversion: (x,y,z) -> (x,z,-y).
    bool referenceZUpToGltf{};

    // Zero derives an absolute deflection from the compared shapes' scale.
    double meshingLinearDeflection{};
    double meshingLinearDeflectionRelative{5.0e-4};
    double meshingAngularDeflectionRadians{0.10};

    // Exact OCCT point-to-boundary queries dominate validation time.  Samples
    // are selected deterministically and area-stratified when this cap applies.
    // Up to this many additional edge/vertex samples verify trimmed boundaries;
    // they contribute to the maximum but not area-weighted surface RMS.
    std::size_t maximumSamplesPerDirection{512};
};

struct ReferenceValidationResult {
    std::filesystem::path referencePath;
    bool referenceLoaded{};
    bool propertiesCompared{};
    bool distancesCompared{};
    double comparisonScale{};
    double meshingLinearDeflection{};

    ValidationShapeMetrics reconstructed;
    ValidationShapeMetrics reference;
    TopologyCountComparison topology;
    PropertyComparison properties;
    BidirectionalSurfaceDistance distances;
    std::vector<std::string> diagnostics;
};

// Post-hoc comparison of two already-built shapes.  This function only
// observes copies of the geometry for meshing and cannot affect reconstruction
// selection or fitting.
[[nodiscard]] ReferenceValidationResult compareShapesForValidation(
    const TopoDS_Shape& reconstructed,
    const TopoDS_Shape& reference,
    const ReferenceValidationOptions& options = {});

// Loads a STEP reference and performs the same post-hoc comparison.  Failure
// to read/transfer the reference is reported through the result diagnostics.
[[nodiscard]] ReferenceValidationResult validateAgainstStepReference(
    const TopoDS_Shape& reconstructed,
    const std::filesystem::path& stepPath,
    const ReferenceValidationOptions& options = {});

} // namespace recad
