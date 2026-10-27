#pragma once

#include "Geometry.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace recad {

// BSpline is reserved for the OpenCascade fallback stage. This module returns
// Unknown when no analytic candidate explains a patch within the requested
// tolerances; the caller can then replace it with a fitted BSpline surface.
enum class SurfaceKind {
    Unknown,
    Plane,
    Cylinder,
    Sphere,
    Torus,
    BSpline
};

[[nodiscard]] const char* toString(SurfaceKind kind) noexcept;

struct CylinderSurface {
    // axisPoint is the point on the axis in the plane through the patch's
    // weighted centroid. axisDirection has unit length for a valid model.
    Vec3d axisPoint{};
    Vec3d axisDirection{};
    double radius{};
};

struct SphereSurface {
    Vec3d center{};
    double radius{};
};

struct TorusSurface {
    Vec3d center{};
    Vec3d axisDirection{};
    double majorRadius{};
    double minorRadius{};
};

using SurfaceParameters = std::variant<std::monostate,
                                       Plane,
                                       CylinderSurface,
                                       SphereSurface,
                                       TorusSurface>;

struct SurfaceResidualSummary {
    // Position residuals are evaluated at triangle vertices with each
    // triangle's area divided equally among its three samples.
    double positionRms{std::numeric_limits<double>::infinity()};
    double positionMax{std::numeric_limits<double>::infinity()};

    // Normal residuals are unsigned angles in radians. Opposite normals are
    // equivalent so inner and outer faces use the same analytic model.
    double normalRmsRadians{std::numeric_limits<double>::infinity()};
    double normalMaxRadians{std::numeric_limits<double>::infinity()};
    // Excess beyond the normal variation across a facet. Coarse or sliver
    // tessellation can have accurate vertices but a biased facet normal.
    double normalExcessRmsRadians{std::numeric_limits<double>::infinity()};
    double normalExcessMaxRadians{std::numeric_limits<double>::infinity()};

    // Inliers require all three vertices within twice the RMS position
    // tolerance, and angular excess within the normal tolerance.
    double inlierAreaFraction{};
    double area{};
    std::size_t positionSampleCount{};
    std::size_t normalSampleCount{};
};

struct SurfaceFitOptions {
    // Zero derives a tolerance from mesh scale and the weld tolerance.
    double positionTolerance{};
    double normalToleranceRadians{0.17453292519943295}; // 10 degrees.
    double maximumPositionResidualFactor{8.0};
    // Raw RMS guard; isolated facet-normal outliers are handled by support area.
    double maximumNormalResidualRadians{0.4363323129985824}; // 25 degrees.
    double minimumInlierAreaFraction{0.90};

    std::size_t minimumTriangleCount{2};
    double minimumPatchArea{};
    double minimumPatchAreaFraction{};

    std::size_t maximumRefinementIterations{64};
    double convergenceTolerance{1.0e-10};
    double huberThresholdFactor{1.5};

    // Model score is the log of normalized positional/normal mean-square
    // error plus this BIC-style parameter-count penalty. Lower is better.
    double normalScoreWeight{0.25};
    double complexityPenaltyWeight{1.0};

    // Partial tori do not always expose their revolution axis through local
    // point/normal covariance. These deterministic hemisphere samples augment
    // data-derived axes; only the best coarse hypotheses receive a full solve.
    std::size_t torusFibonacciAxisHypotheses{64};
    std::size_t torusRefinedAxisHypotheses{16};

    // fitSurfacePatches propagates axes/axis lines from accepted adjacent
    // cylinders and tori, and consensus directions from the same model family,
    // into unresolved patches for this many passes.
    std::size_t maximumAxisConsensusPasses{3};
    double axisConsensusToleranceRadians{0.08726646259971647}; // 5 degrees.

    // Higher-complexity families can be skipped when a simpler fit is already
    // far inside both normalized tolerances. Zero disables this pruning.
    double excellentFitPositionRatio{0.02};
    double excellentFitNormalRatio{0.15};

    // A smooth CUDA component can contain tangent but analytically distinct
    // surfaces. When enabled, unresolved regions are searched with local,
    // deterministic hypotheses; connected inliers are extracted and refit.
    bool splitMixedRegions{true};
    std::size_t maximumExtractedPatchesPerRegion{128};
    std::size_t maximumHypothesisSeeds{64};
    std::size_t hypothesisNeighborhoodTriangles{24};
    double hypothesisInlierToleranceFactor{2.0};
    double minimumExtractedAreaFraction{1.0e-5};

    bool enablePlanes{true};
    bool enableCylinders{true};
    bool enableSpheres{true};
    bool enableTori{true};
};

struct SurfacePatchFit {
    SurfaceKind kind{SurfaceKind::Unknown};
    SurfaceParameters parameters{};
    std::vector<std::uint32_t> triangleIndices;
    SurfaceResidualSummary residuals;
    double score{std::numeric_limits<double>::infinity()};
    bool accepted{};
    std::string diagnostic;
};

// Fits and selects an analytic model for one connected (or otherwise
// pre-segmented) collection of triangles. Selection is independent of file
// names and expected sample structure.
[[nodiscard]] SurfacePatchFit fitSurfacePatch(
    const Mesh& mesh,
    std::span<const std::uint32_t> triangleIndices,
    const SurfaceFitOptions& options = {});

// Convenience entry point for CUDA/graph segmentation output represented as
// one triangle-index vector per region. A supplied region can produce multiple
// returned patches when splitMixedRegions is enabled.
[[nodiscard]] std::vector<SurfacePatchFit> fitSurfacePatches(
    const Mesh& mesh,
    const std::vector<std::vector<std::uint32_t>>& regions,
    const SurfaceFitOptions& options = {});

// Convenience entry point for a dense label array. Negative labels are
// ignored. Empty label IDs are skipped, and returned patches are ordered by
// increasing label.
[[nodiscard]] std::vector<SurfacePatchFit> fitSurfacePatches(
    const Mesh& mesh,
    std::span<const std::int32_t> triangleLabels,
    const SurfaceFitOptions& options = {});

} // namespace recad
