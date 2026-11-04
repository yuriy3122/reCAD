#include "SurfaceFitting.h"
#include "GraphCut.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace recad {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct WeightedPoint {
    Vec3d point{};
    double weight{};
    Vec3d normal{};
};

struct PatchSamples {
    std::vector<WeightedPoint> points;
    std::vector<std::uint32_t> triangles;
    Vec3d centroid{};
    double area{};
};

struct Candidate {
    SurfaceKind kind{SurfaceKind::Unknown};
    SurfaceParameters parameters{};
    SurfaceResidualSummary residuals;
    double score{std::numeric_limits<double>::infinity()};
    bool accepted{};
    std::string diagnostic;
};

[[nodiscard]] Eigen::Vector3d eigenVector(const Vec3d& value) noexcept {
    return {value.x, value.y, value.z};
}

[[nodiscard]] Vec3d vec3(const Eigen::Vector3d& value) noexcept {
    return {value.x(), value.y(), value.z()};
}

[[nodiscard]] double clampUnit(double value) noexcept {
    return std::clamp(value, -1.0, 1.0);
}

[[nodiscard]] Vec3d canonicalDirection(Vec3d value) noexcept {
    value = normalized(value);
    const std::array<double, 3> absolute{
        std::abs(value.x), std::abs(value.y), std::abs(value.z)};
    std::size_t dominant = 0;
    if (absolute[1] > absolute[dominant]) dominant = 1;
    if (absolute[2] > absolute[dominant]) dominant = 2;
    if (value[dominant] < 0.0) value = -value;
    return value;
}

[[nodiscard]] std::pair<Vec3d, Vec3d> perpendicularFrame(const Vec3d& direction) {
    const Vec3d axis = normalized(direction);
    const std::array<Vec3d, 3> coordinates{
        Vec3d{1.0, 0.0, 0.0}, Vec3d{0.0, 1.0, 0.0}, Vec3d{0.0, 0.0, 1.0}};
    std::size_t leastAligned = 0;
    for (std::size_t i = 1; i < coordinates.size(); ++i) {
        if (std::abs(dot(axis, coordinates[i])) <
            std::abs(dot(axis, coordinates[leastAligned]))) {
            leastAligned = i;
        }
    }
    const Vec3d first = normalized(cross(axis, coordinates[leastAligned]));
    return {first, normalized(cross(axis, first))};
}

[[nodiscard]] double derivedPositionTolerance(const Mesh& mesh,
                                              const SurfaceFitOptions& options) {
    if (options.positionTolerance > 0.0) return options.positionTolerance;
    const double scale = std::max(mesh.bounds.diagonal(),
                                  std::numeric_limits<double>::min());
    // CAD tessellators may insert points on chords before Draco encoding. Such
    // points are inside the smooth surface, even with exact exporter normals.
    // Estimate that positional uncertainty from chord length and normal turn:
    // sag = (chord / 2) * tan(turn / 4). Ignore flat edges and use a robust
    // quantile so isolated normal discontinuities cannot set the fit budget.
    std::vector<double> chordSags;
    if (mesh.cornerNormals.size() == mesh.triangles.size()) {
        for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
            for (unsigned edge = 0; edge < 3; ++edge) {
                const unsigned next = (edge + 1) % 3;
                const Vec3d a = normalized(mesh.cornerNormals[i][edge]);
                const Vec3d b = normalized(mesh.cornerNormals[i][next]);
                if (norm(a) == 0.0 || norm(b) == 0.0) continue;
                const double turn = std::acos(clampUnit(dot(a, b)));
                if (turn < 0.002 || turn > kPi * 0.5) continue;
                const double chord = norm(mesh.vertices[mesh.triangles[i][edge]] -
                                          mesh.vertices[mesh.triangles[i][next]]);
                chordSags.push_back(chord * 0.5 * std::tan(turn * 0.25));
            }
        }
    }
    double chordUncertainty = 0.0;
    if (!chordSags.empty()) {
        const auto quantile = chordSags.begin() + static_cast<std::ptrdiff_t>(chordSags.size() * 9 / 10);
        std::nth_element(chordSags.begin(), quantile, chordSags.end());
        chordUncertainty = std::min(*quantile * 0.5, scale * 1.0e-3);
    }
    return std::max({mesh.weldTolerance * 8.0,
                     chordUncertainty,
                     scale * 2.0e-5,
                     std::numeric_limits<double>::epsilon() * scale * 256.0});
}

void validateOptions(const SurfaceFitOptions& options) {
    if (!std::isfinite(options.positionTolerance) || options.positionTolerance < 0.0 ||
        !std::isfinite(options.normalToleranceRadians) ||
        options.normalToleranceRadians <= 0.0 ||
        options.normalToleranceRadians >= kPi * 0.5 ||
        !std::isfinite(options.maximumPositionResidualFactor) ||
        options.maximumPositionResidualFactor < 1.0 ||
        !std::isfinite(options.maximumNormalResidualRadians) ||
        options.maximumNormalResidualRadians <= 0.0 ||
        options.maximumNormalResidualRadians >= kPi * 0.5 ||
        !std::isfinite(options.minimumInlierAreaFraction) ||
        options.minimumInlierAreaFraction < 0.0 ||
        options.minimumInlierAreaFraction > 1.0 ||
        !std::isfinite(options.minimumPatchArea) || options.minimumPatchArea < 0.0 ||
        !std::isfinite(options.minimumPatchAreaFraction) ||
        options.minimumPatchAreaFraction < 0.0 ||
        options.maximumRefinementIterations == 0U ||
        !std::isfinite(options.convergenceTolerance) ||
        options.convergenceTolerance <= 0.0 ||
        !std::isfinite(options.huberThresholdFactor) ||
        options.huberThresholdFactor <= 0.0 ||
        !std::isfinite(options.normalScoreWeight) || options.normalScoreWeight < 0.0 ||
        !std::isfinite(options.complexityPenaltyWeight) ||
        options.complexityPenaltyWeight < 0.0 ||
        !std::isfinite(options.boundaryRegularizationWeight) || options.boundaryRegularizationWeight < 0.0 ||
        options.maximumExtractedPatchesPerRegion == 0U ||
        options.maximumHypothesisSeeds == 0U ||
        options.hypothesisNeighborhoodTriangles < 2U ||
        !std::isfinite(options.hypothesisInlierToleranceFactor) ||
        options.hypothesisInlierToleranceFactor < 1.0 ||
        !std::isfinite(options.minimumExtractedAreaFraction) ||
        options.minimumExtractedAreaFraction < 0.0 ||
        options.minimumExtractedAreaFraction > 1.0) {
        throw std::invalid_argument("Invalid surface-fitting options");
    }
}

[[nodiscard]] PatchSamples collectSamples(
    const Mesh& mesh, std::span<const std::uint32_t> triangleIndices) {
    if (mesh.features.size() != mesh.triangles.size()) {
        throw std::invalid_argument("Mesh triangle features are missing or stale");
    }

    PatchSamples result;
    result.triangles.assign(triangleIndices.begin(), triangleIndices.end());
    std::sort(result.triangles.begin(), result.triangles.end());
    result.triangles.erase(
        std::unique(result.triangles.begin(), result.triangles.end()),
        result.triangles.end());
    result.points.reserve(result.triangles.size() * 3U);

    for (const std::uint32_t triangleIndex : result.triangles) {
        if (triangleIndex >= mesh.triangles.size()) {
            throw std::out_of_range("Surface-patch triangle index is out of range");
        }
        const TriangleFeatures& feature = mesh.features[triangleIndex];
        if (feature.degenerate || !(feature.area > 0.0) || !std::isfinite(feature.area)) {
            continue;
        }
        result.area += feature.area;
        result.centroid += feature.centroid * feature.area;
        const double weight = feature.area / 3.0;
        unsigned corner = 0;
        for (const std::uint32_t vertex : mesh.triangles[triangleIndex].vertices) {
            if (vertex >= mesh.vertices.size()) {
                throw std::out_of_range("Surface-patch vertex index is out of range");
            }
            const Vec3d normal = mesh.cornerNormals.size() == mesh.triangles.size()
                && norm(mesh.cornerNormals[triangleIndex][corner]) > 0.0
                ? mesh.cornerNormals[triangleIndex][corner] : feature.normal;
            result.points.push_back({mesh.vertices[vertex], weight, normal});
            ++corner;
        }
    }
    if (result.area > 0.0) result.centroid /= result.area;
    return result;
}

template <typename PositionResidual, typename SurfaceNormal>
[[nodiscard]] SurfaceResidualSummary evaluateModel(
    const Mesh& mesh,
    const PatchSamples& samples,
    double positionTolerance,
    double normalTolerance,
    double smoothNormalTolerance,
    PositionResidual&& positionResidual,
    SurfaceNormal&& surfaceNormal) {
    SurfaceResidualSummary result;
    result.area = samples.area;

    double pointWeight = 0.0;
    double pointSquares = 0.0;
    double normalWeight = 0.0;
    double normalSquares = 0.0;
    double normalExcessSquares = 0.0;
    double inlierArea = 0.0;

    for (const WeightedPoint& sample : samples.points) {
        const double residual = std::abs(positionResidual(sample.point));
        if (!std::isfinite(residual)) continue;
        pointWeight += sample.weight;
        pointSquares += sample.weight * residual * residual;
        result.positionMax = std::isfinite(result.positionMax)
            ? std::max(result.positionMax, residual)
            : residual;
        ++result.positionSampleCount;
    }

    // The default member is infinity so a first finite value must replace it.
    if (result.positionSampleCount > 0U && pointWeight > 0.0) {
        result.positionRms = std::sqrt(std::max(0.0, pointSquares / pointWeight));
    }

    result.normalMaxRadians = -std::numeric_limits<double>::infinity();
    result.normalExcessMaxRadians = 0.0;
    for (const std::uint32_t triangleIndex : samples.triangles) {
        const TriangleFeatures& feature = mesh.features[triangleIndex];
        if (feature.degenerate || !(feature.area > 0.0)) continue;

        const Vec3d predicted = normalized(surfaceNormal(feature.centroid));
        double angle = std::numeric_limits<double>::infinity();
        double excessAngle = angle;
        double observationTolerance = normalTolerance;
        if (norm(predicted) > 0.0 && isFinite(predicted)) {
            angle = std::acos(clampUnit(std::abs(dot(feature.normal, predicted))));
            double facetNormalVariation = 0.0;
            for (const auto vertex : mesh.triangles[triangleIndex].vertices) {
                const Vec3d vertexNormal = normalized(surfaceNormal(mesh.vertices[vertex]));
                if (norm(vertexNormal) > 0.0) {
                    facetNormalVariation = std::max(facetNormalVariation,
                        std::acos(clampUnit(dot(vertexNormal, predicted))));
                }
            }
            excessAngle = std::max(0.0, angle - facetNormalVariation);
            // Exported corner normals measure the underlying smooth surface,
            // whereas a coarse facet normal measures its chord. Keep the two
            // observations distinct and use smooth normals when supplied.
            if (mesh.cornerNormals.size() == mesh.triangles.size()) {
                double cornerSquares = 0.0;
                unsigned count = 0;
                for (std::size_t corner = 0; corner < 3; ++corner) {
                    const Vec3d observed = normalized(mesh.cornerNormals[triangleIndex][corner]);
                    if (norm(observed) == 0.0) continue;
                    const Vec3d expected = normalized(surfaceNormal(
                        mesh.vertices[mesh.triangles[triangleIndex][corner]]));
                    const double error = std::acos(clampUnit(std::abs(dot(observed, expected))));
                    cornerSquares += error * error;
                    ++count;
                }
                if (count == 3) {
                    angle = excessAngle = std::sqrt(cornerSquares / count);
                    observationTolerance = std::min(normalTolerance, smoothNormalTolerance);
                }
            }
            normalWeight += feature.area;
            normalSquares += feature.area * angle * angle;
            normalExcessSquares += feature.area * excessAngle * excessAngle;
            result.normalMaxRadians = std::max(result.normalMaxRadians, angle);
            result.normalExcessMaxRadians = std::max(
                result.normalExcessMaxRadians, excessAngle);
            ++result.normalSampleCount;
        }

        bool positionInlier = true;
        for (const std::uint32_t vertex : mesh.triangles[triangleIndex].vertices) {
            const double residual = std::abs(positionResidual(mesh.vertices[vertex]));
            if (!std::isfinite(residual) || residual > positionTolerance * 2.0) {
                positionInlier = false;
                break;
            }
        }
        if (positionInlier && excessAngle <= observationTolerance) inlierArea += feature.area;
    }
    if (result.normalSampleCount > 0U && normalWeight > 0.0) {
        result.normalRmsRadians = std::sqrt(std::max(0.0, normalSquares / normalWeight));
        result.normalExcessRmsRadians = std::sqrt(
            std::max(0.0, normalExcessSquares / normalWeight));
    }
    if (!std::isfinite(result.normalMaxRadians)) {
        result.normalMaxRadians = std::numeric_limits<double>::infinity();
    }
    result.inlierAreaFraction = samples.area > 0.0
        ? std::clamp(inlierArea / samples.area, 0.0, 1.0)
        : 0.0;
    return result;
}

[[nodiscard]] std::size_t parameterCount(SurfaceKind kind) noexcept {
    switch (kind) {
    case SurfaceKind::Plane: return 3U;
    case SurfaceKind::Cylinder: return 5U;
    case SurfaceKind::Cone: return 6U;
    case SurfaceKind::Sphere: return 4U;
    case SurfaceKind::Torus: return 7U;
    default: return 0U;
    }
}

[[nodiscard]] double modelScore(SurfaceKind kind,
                                const SurfaceResidualSummary& residuals,
                                double positionTolerance,
                                const SurfaceFitOptions& options) {
    const double position = residuals.positionRms / positionTolerance;
    const double normal = residuals.normalExcessRmsRadians / options.normalToleranceRadians;
    const double meanNormalizedSquare = position * position +
        options.normalScoreWeight * normal * normal;
    const double observations = static_cast<double>(std::max<std::size_t>(
        2U, residuals.positionSampleCount + residuals.normalSampleCount));
    const double penalty = options.complexityPenaltyWeight *
        static_cast<double>(parameterCount(kind)) * std::log(observations) / observations;
    // Residuals below the measurement tolerance carry no evidence for extra
    // parameters. A noise floor prevents a cylinder interpolating four nearly
    // coplanar quantized points from defeating the planar model.
    return std::log1p(meanNormalizedSquare) + penalty;
}

[[nodiscard]] bool passesTolerance(const SurfaceResidualSummary& residuals,
                                   double positionTolerance,
                                   const SurfaceFitOptions& options) {
    return std::isfinite(residuals.positionRms) &&
           std::isfinite(residuals.positionMax) &&
           std::isfinite(residuals.normalRmsRadians) &&
           std::isfinite(residuals.normalMaxRadians) &&
           residuals.positionRms <= positionTolerance &&
           residuals.positionMax <= positionTolerance *
               options.maximumPositionResidualFactor &&
           residuals.normalExcessRmsRadians <= options.normalToleranceRadians &&
           // Angular outliers on coarse trimmed facets are handled by the
           // inlier-area test. A single sliver must not reject an entire face.
           residuals.normalRmsRadians <= options.maximumNormalResidualRadians &&
           residuals.inlierAreaFraction >= options.minimumInlierAreaFraction;
}

[[nodiscard]] double huberWeight(double residual, double threshold) noexcept {
    const double absolute = std::abs(residual);
    return absolute <= threshold || absolute == 0.0 ? 1.0 : threshold / absolute;
}

[[nodiscard]] double huberCost(const std::vector<WeightedPoint>& points,
                               double threshold,
                               const auto& residualFunction) {
    double result = 0.0;
    for (const WeightedPoint& sample : points) {
        const double residual = residualFunction(sample.point);
        if (!std::isfinite(residual)) return std::numeric_limits<double>::infinity();
        const double absolute = std::abs(residual);
        const double loss = absolute <= threshold
            ? 0.5 * residual * residual
            : threshold * (absolute - 0.5 * threshold);
        result += sample.weight * loss;
    }
    return result;
}

void appendAxis(std::vector<Vec3d>& axes, const Vec3d& candidate) {
    Vec3d axis = canonicalDirection(candidate);
    if (!isFinite(axis) || norm(axis) == 0.0) return;
    constexpr double duplicateCosine = 0.9993908270190958; // 2 degrees.
    for (const Vec3d& existing : axes) {
        if (std::abs(dot(existing, axis)) >= duplicateCosine) return;
    }
    axes.push_back(axis);
}

[[nodiscard]] std::vector<Vec3d> candidateAxes(const Mesh& mesh,
                                               const PatchSamples& samples) {
    std::vector<Vec3d> axes;
    if (!(samples.area > 0.0)) return axes;

    // Source patches can have too little angular coverage to determine their
    // revolution axis independently. Other planar source patches provide
    // rotation-invariant shared-frame hypotheses, never imposed constraints.
    if (mesh.sourcePatchLabels.size() == mesh.triangles.size() &&
        mesh.sourcePatchCount > 1U && mesh.sourcePatchCount <= mesh.triangles.size() / 2U) {
        std::vector<Vec3d> normalSums(mesh.sourcePatchCount);
        std::vector<double> areas(mesh.sourcePatchCount, 0.0);
        for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
            const auto label = mesh.sourcePatchLabels[i];
            if (label >= areas.size()) continue;
            normalSums[label] += mesh.features[i].normal * mesh.features[i].area;
            areas[label] += mesh.features[i].area;
        }
        for (std::size_t i = 0; i < areas.size(); ++i) {
            if (areas[i] > 0.0 && norm(normalSums[i]) / areas[i] > 0.99999) {
                appendAxis(axes, normalSums[i]);
            }
        }
    }

    Eigen::Matrix3d pointCovariance = Eigen::Matrix3d::Zero();
    for (const WeightedPoint& sample : samples.points) {
        const Eigen::Vector3d difference = eigenVector(sample.point - samples.centroid);
        pointCovariance.noalias() += sample.weight * difference * difference.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> pointSolver(pointCovariance);
    if (pointSolver.info() == Eigen::Success) {
        for (Eigen::Index i = 0; i < 3; ++i) appendAxis(axes, vec3(pointSolver.eigenvectors().col(i)));
    }

    Eigen::Matrix3d normalMoment = Eigen::Matrix3d::Zero();
    std::vector<Vec3d> representativeNormals;
    representativeNormals.reserve(std::min<std::size_t>(samples.triangles.size(), 16U));
    const std::size_t stride = std::max<std::size_t>(1U, samples.triangles.size() / 16U);
    for (std::size_t i = 0; i < samples.triangles.size(); ++i) {
        const TriangleFeatures& feature = mesh.features[samples.triangles[i]];
        if (feature.degenerate || !(feature.area > 0.0)) continue;
        const Eigen::Vector3d normal = eigenVector(feature.normal);
        normalMoment.noalias() += feature.area * normal * normal.transpose();
        if (i % stride == 0U && representativeNormals.size() < 16U) {
            representativeNormals.push_back(feature.normal);
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> normalSolver(normalMoment);
    if (normalSolver.info() == Eigen::Success) {
        for (Eigen::Index i = 0; i < 3; ++i) appendAxis(axes, vec3(normalSolver.eigenvectors().col(i)));
    }

    // Cone normals lie on a small circle on the unit sphere. Its centered
    // covariance has the revolution axis as its least-variance direction.
    Eigen::Vector3d meanNormal = Eigen::Vector3d::Zero();
    for (const auto& sample : samples.points) meanNormal += sample.weight * eigenVector(sample.normal);
    meanNormal /= samples.area;
    Eigen::Matrix3d centeredNormals = Eigen::Matrix3d::Zero();
    for (const auto& sample : samples.points) {
        const Eigen::Vector3d d = eigenVector(sample.normal) - meanNormal;
        centeredNormals.noalias() += sample.weight * d * d.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> coneAxes(centeredNormals);
    if (coneAxes.info() == Eigen::Success) appendAxis(axes, vec3(coneAxes.eigenvectors().col(0)));

    // Crossed normals are strong cylinder-axis hypotheses. Limiting this to a
    // deterministic representative set avoids quadratic growth on dense meshes.
    for (std::size_t i = 0; i < representativeNormals.size(); ++i) {
        for (std::size_t j = i + 1U; j < representativeNormals.size(); ++j) {
            const Vec3d crossed = cross(representativeNormals[i], representativeNormals[j]);
            if (norm(crossed) > 0.05) appendAxis(axes, crossed);
            if (axes.size() >= 40U) return axes;
        }
    }
    return axes;
}

[[nodiscard]] SurfaceResidualSummary evaluatePlane(
    const Mesh& mesh, const PatchSamples& samples, const Plane& plane,
    double positionTolerance, const SurfaceFitOptions& options) {
    return evaluateModel(mesh, samples, positionTolerance,
        options.normalToleranceRadians, options.smoothNormalToleranceRadians,
        [&plane](const Vec3d& point) { return plane.signedDistance(point); },
        [&plane](const Vec3d&) { return plane.normal; });
}

[[nodiscard]] SurfaceResidualSummary evaluateSphere(
    const Mesh& mesh, const PatchSamples& samples, const SphereSurface& sphere,
    double positionTolerance, const SurfaceFitOptions& options) {
    return evaluateModel(mesh, samples, positionTolerance,
        options.normalToleranceRadians, options.smoothNormalToleranceRadians,
        [&sphere](const Vec3d& point) {
            return norm(point - sphere.center) - sphere.radius;
        },
        [&sphere](const Vec3d& point) {
            return normalized(point - sphere.center);
        });
}

[[nodiscard]] double cylinderResidual(const CylinderSurface& cylinder,
                                      const Vec3d& point) {
    const Vec3d difference = point - cylinder.axisPoint;
    const Vec3d radial = difference - cylinder.axisDirection *
        dot(cylinder.axisDirection, difference);
    return norm(radial) - cylinder.radius;
}

[[nodiscard]] Vec3d cylinderNormal(const CylinderSurface& cylinder,
                                   const Vec3d& point) {
    const Vec3d difference = point - cylinder.axisPoint;
    return normalized(difference - cylinder.axisDirection *
        dot(cylinder.axisDirection, difference));
}

[[nodiscard]] SurfaceResidualSummary evaluateCylinder(
    const Mesh& mesh, const PatchSamples& samples, const CylinderSurface& cylinder,
    double positionTolerance, const SurfaceFitOptions& options) {
    return evaluateModel(mesh, samples, positionTolerance,
        options.normalToleranceRadians, options.smoothNormalToleranceRadians,
        [&cylinder](const Vec3d& point) { return cylinderResidual(cylinder, point); },
        [&cylinder](const Vec3d& point) { return cylinderNormal(cylinder, point); });
}

[[nodiscard]] double coneResidual(const ConeSurface& cone, const Vec3d& point) {
    const Vec3d d = point - cone.apex;
    const double height = dot(d, cone.axisDirection);
    return norm(d - cone.axisDirection * height) * std::cos(cone.semiAngle) -
           height * std::sin(cone.semiAngle);
}

[[nodiscard]] Vec3d coneNormal(const ConeSurface& cone, const Vec3d& point) {
    const Vec3d d = point - cone.apex;
    const double height = dot(d, cone.axisDirection);
    return normalized(d - cone.axisDirection * height) * std::cos(cone.semiAngle) -
           cone.axisDirection * std::sin(cone.semiAngle);
}

[[nodiscard]] double torusResidual(const TorusSurface& torus, const Vec3d& point) {
    const Vec3d difference = point - torus.center;
    const double axial = dot(torus.axisDirection, difference);
    const Vec3d radial = difference - torus.axisDirection * axial;
    const double tubeCenterDistance = std::hypot(
        norm(radial) - torus.majorRadius, axial);
    return tubeCenterDistance - torus.minorRadius;
}

[[nodiscard]] Vec3d torusNormal(const TorusSurface& torus, const Vec3d& point) {
    const Vec3d difference = point - torus.center;
    const double axial = dot(torus.axisDirection, difference);
    const Vec3d radial = difference - torus.axisDirection * axial;
    const double radialLength = norm(radial);
    if (!(radialLength > std::numeric_limits<double>::min())) return {};
    const Vec3d radialDirection = radial / radialLength;
    return normalized(radialDirection * (radialLength - torus.majorRadius) +
                      torus.axisDirection * axial);
}

[[nodiscard]] SurfaceResidualSummary evaluateTorus(
    const Mesh& mesh, const PatchSamples& samples, const TorusSurface& torus,
    double positionTolerance, const SurfaceFitOptions& options) {
    return evaluateModel(mesh, samples, positionTolerance,
        options.normalToleranceRadians, options.smoothNormalToleranceRadians,
        [&torus](const Vec3d& point) { return torusResidual(torus, point); },
        [&torus](const Vec3d& point) { return torusNormal(torus, point); });
}

[[nodiscard]] bool initializeCylinder(const PatchSamples& samples,
                                      const Vec3d& proposedAxis,
                                      CylinderSurface& result) {
    const Vec3d axis = canonicalDirection(proposedAxis);
    if (norm(axis) == 0.0) return false;
    const auto [first, second] = perpendicularFrame(axis);

    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d right = Eigen::Vector3d::Zero();
    for (const WeightedPoint& sample : samples.points) {
        const Vec3d centered = sample.point - samples.centroid;
        const double x = dot(centered, first);
        const double y = dot(centered, second);
        const Eigen::Vector3d row(2.0 * x, 2.0 * y, 1.0);
        normal.noalias() += sample.weight * row * row.transpose();
        right.noalias() += sample.weight * row * (x * x + y * y);
    }
    Eigen::LDLT<Eigen::Matrix3d> solver(normal);
    if (solver.info() != Eigen::Success) return false;
    const Eigen::Vector3d solution = solver.solve(right);
    if (solver.info() != Eigen::Success || !solution.allFinite()) return false;
    const double radiusSquared = solution[0] * solution[0] +
                                 solution[1] * solution[1] + solution[2];
    if (!(radiusSquared > 0.0) || !std::isfinite(radiusSquared)) return false;

    result.axisDirection = axis;
    result.axisPoint = samples.centroid + first * solution[0] + second * solution[1];
    result.radius = std::sqrt(radiusSquared);
    return isFinite(result.axisPoint) && std::isfinite(result.radius);
}

[[nodiscard]] CylinderSurface refineCylinder(
    const PatchSamples& samples,
    CylinderSurface model,
    double positionTolerance,
    const SurfaceFitOptions& options) {
    const double threshold = std::max(positionTolerance * options.huberThresholdFactor,
                                      std::numeric_limits<double>::epsilon());
    double damping = 1.0e-6;
    double cost = huberCost(samples.points, threshold,
        [&model](const Vec3d& point) { return cylinderResidual(model, point); });

    for (std::size_t iteration = 0;
         iteration < options.maximumRefinementIterations; ++iteration) {
        const auto [first, second] = perpendicularFrame(model.axisDirection);
        Eigen::Matrix<double, 5, 5> normal = Eigen::Matrix<double, 5, 5>::Zero();
        Eigen::Matrix<double, 5, 1> gradient = Eigen::Matrix<double, 5, 1>::Zero();

        for (const WeightedPoint& sample : samples.points) {
            const Vec3d difference = sample.point - model.axisPoint;
            const double axial = dot(model.axisDirection, difference);
            const Vec3d radial = difference - model.axisDirection * axial;
            const double radialLength = norm(radial);
            if (!(radialLength > std::numeric_limits<double>::min())) continue;
            const Vec3d radialDirection = radial / radialLength;
            const double residual = radialLength - model.radius;
            const double weight = sample.weight * huberWeight(residual, threshold);
            Eigen::Matrix<double, 5, 1> jacobian;
            jacobian << -dot(radialDirection, first),
                        -dot(radialDirection, second),
                        -axial * dot(radialDirection, first),
                        -axial * dot(radialDirection, second),
                        -1.0;
            normal.noalias() += weight * jacobian * jacobian.transpose();
            gradient.noalias() += weight * jacobian * residual;
        }

        bool accepted = false;
        Eigen::Matrix<double, 5, 1> acceptedStep =
            Eigen::Matrix<double, 5, 1>::Zero();
        for (std::size_t attempt = 0; attempt < 12U; ++attempt) {
            Eigen::Matrix<double, 5, 5> damped = normal;
            for (Eigen::Index i = 0; i < 5; ++i) {
                damped(i, i) += damping * std::max(normal(i, i), 1.0e-12);
            }
            const Eigen::Matrix<double, 5, 1> step = damped.ldlt().solve(-gradient);
            if (!step.allFinite()) {
                damping *= 10.0;
                continue;
            }
            CylinderSurface candidate = model;
            candidate.axisPoint += first * step[0] + second * step[1];
            candidate.axisDirection = normalized(model.axisDirection +
                first * step[2] + second * step[3]);
            candidate.radius += step[4];
            if (!(candidate.radius > 0.0) || norm(candidate.axisDirection) == 0.0) {
                damping *= 10.0;
                continue;
            }
            // Remove the free translation along the infinite cylinder axis.
            candidate.axisPoint += candidate.axisDirection *
                dot(samples.centroid - candidate.axisPoint, candidate.axisDirection);
            candidate.axisDirection = canonicalDirection(candidate.axisDirection);
            const double candidateCost = huberCost(samples.points, threshold,
                [&candidate](const Vec3d& point) {
                    return cylinderResidual(candidate, point);
                });
            if (candidateCost <= cost) {
                model = candidate;
                cost = candidateCost;
                acceptedStep = step;
                damping = std::max(damping * 0.25, 1.0e-15);
                accepted = true;
                break;
            }
            damping *= 10.0;
        }
        if (!accepted) break;
        const double scale = std::max(model.radius, positionTolerance);
        const double stepMetric = std::max({
            std::abs(acceptedStep[0]) / scale,
            std::abs(acceptedStep[1]) / scale,
            std::abs(acceptedStep[2]),
            std::abs(acceptedStep[3]),
            std::abs(acceptedStep[4]) / scale});
        if (stepMetric <= options.convergenceTolerance) break;
    }
    return model;
}

[[nodiscard]] bool initializeTorus(const Mesh& mesh,
                                   const PatchSamples& samples,
                                   const Vec3d& proposedAxis,
                                   TorusSurface& result) {
    const Vec3d axis = canonicalDirection(proposedAxis);
    if (norm(axis) == 0.0) return false;
    const auto [first, second] = perpendicularFrame(axis);

    // For a torus, the projection of a surface normal onto the equatorial
    // plane is radial. The corresponding projected normal lines therefore
    // intersect at the torus center. Solve that line-intersection problem in
    // weighted least squares; line direction is sign-independent.
    Eigen::Matrix2d centerNormal = Eigen::Matrix2d::Zero();
    Eigen::Vector2d centerRight = Eigen::Vector2d::Zero();
    std::size_t usableLines = 0U;
    for (const std::uint32_t triangleIndex : samples.triangles) {
        const TriangleFeatures& feature = mesh.features[triangleIndex];
        if (feature.degenerate || !(feature.area > 0.0)) continue;
        const Vec3d projectedNormal = feature.normal - axis * dot(axis, feature.normal);
        const double projectedLength = norm(projectedNormal);
        if (projectedLength < 1.0e-4) continue;
        const Vec3d direction = projectedNormal / projectedLength;
        const Eigen::Vector2d lineNormal(-dot(direction, second), dot(direction, first));
        const Vec3d centered = feature.centroid - samples.centroid;
        const Eigen::Vector2d point(dot(centered, first), dot(centered, second));
        const double weight = feature.area * projectedLength * projectedLength;
        centerNormal.noalias() += weight * lineNormal * lineNormal.transpose();
        centerRight.noalias() += weight * lineNormal * lineNormal.dot(point);
        ++usableLines;
    }
    if (usableLines < 3U) return false;
    Eigen::LDLT<Eigen::Matrix2d> centerSolver(centerNormal);
    if (centerSolver.info() != Eigen::Success) return false;
    const Eigen::Vector2d center2 = centerSolver.solve(centerRight);
    if (centerSolver.info() != Eigen::Success || !center2.allFinite()) return false;

    // With the equatorial center fixed, both axial and radial equations are
    // linear in axial center, major radius, and a signed minor radius. The
    // sign absorbs whether mesh normals face toward or away from the material.
    Eigen::Matrix3d parameterNormal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d parameterRight = Eigen::Vector3d::Zero();
    std::size_t equations = 0U;
    for (const std::uint32_t triangleIndex : samples.triangles) {
        const TriangleFeatures& feature = mesh.features[triangleIndex];
        if (feature.degenerate || !(feature.area > 0.0)) continue;
        const Vec3d centered = feature.centroid - samples.centroid;
        const double x = dot(centered, first) - center2[0];
        const double y = dot(centered, second) - center2[1];
        const double rho = std::hypot(x, y);
        if (!(rho > std::numeric_limits<double>::min())) continue;
        const Vec3d radialDirection = first * (x / rho) + second * (y / rho);
        const double axialCoordinate = dot(centered, axis);
        const double axialNormal = dot(feature.normal, axis);
        const double radialNormal = dot(feature.normal, radialDirection);

        const Eigen::Vector3d axialRow(1.0, 0.0, axialNormal);
        const Eigen::Vector3d radialRow(0.0, 1.0, radialNormal);
        parameterNormal.noalias() += feature.area *
            (axialRow * axialRow.transpose() + radialRow * radialRow.transpose());
        parameterRight.noalias() += feature.area *
            (axialRow * axialCoordinate + radialRow * rho);
        equations += 2U;
    }
    if (equations < 6U) return false;
    Eigen::LDLT<Eigen::Matrix3d> parameterSolver(parameterNormal);
    if (parameterSolver.info() != Eigen::Success) return false;
    const Eigen::Vector3d parameter = parameterSolver.solve(parameterRight);
    if (parameterSolver.info() != Eigen::Success || !parameter.allFinite()) return false;

    result.center = samples.centroid + first * center2[0] + second * center2[1] +
                    axis * parameter[0];
    result.axisDirection = axis;
    result.majorRadius = parameter[1];
    result.minorRadius = std::abs(parameter[2]);
    return isFinite(result.center) && result.majorRadius > 0.0 &&
           result.minorRadius > 0.0 && std::isfinite(result.majorRadius) &&
           std::isfinite(result.minorRadius);
}

[[nodiscard]] TorusSurface refineTorus(
    const PatchSamples& samples,
    TorusSurface model,
    double positionTolerance,
    const SurfaceFitOptions& options,
    bool fixedAxis = false) {
    const double threshold = std::max(positionTolerance * options.huberThresholdFactor,
                                      std::numeric_limits<double>::epsilon());
    double damping = 1.0e-6;
    double cost = huberCost(samples.points, threshold,
        [&model](const Vec3d& point) { return torusResidual(model, point); });

    for (std::size_t iteration = 0;
         iteration < options.maximumRefinementIterations; ++iteration) {
        const auto [first, second] = perpendicularFrame(model.axisDirection);
        Eigen::Matrix<double, 7, 7> normal = Eigen::Matrix<double, 7, 7>::Zero();
        Eigen::Matrix<double, 7, 1> gradient = Eigen::Matrix<double, 7, 1>::Zero();

        for (const WeightedPoint& sample : samples.points) {
            const Vec3d difference = sample.point - model.center;
            const double axial = dot(model.axisDirection, difference);
            const Vec3d radial = difference - model.axisDirection * axial;
            const double radialLength = norm(radial);
            if (!(radialLength > std::numeric_limits<double>::min())) continue;
            const Vec3d radialDirection = radial / radialLength;
            const double ringOffset = radialLength - model.majorRadius;
            const double tubeDistance = std::hypot(ringOffset, axial);
            if (!(tubeDistance > std::numeric_limits<double>::min())) continue;
            const double residual = tubeDistance - model.minorRadius;
            const Vec3d pointGradient =
                (radialDirection * ringOffset + model.axisDirection * axial) /
                tubeDistance;
            const Vec3d axisGradient = (difference - radialDirection * ringOffset) *
                (axial / tubeDistance);
            const double weight = sample.weight * huberWeight(residual, threshold);

            Eigen::Matrix<double, 7, 1> jacobian;
            jacobian << -pointGradient.x,
                        -pointGradient.y,
                        -pointGradient.z,
                         dot(axisGradient, first),
                         dot(axisGradient, second),
                        -ringOffset / tubeDistance,
                        -1.0;
            normal.noalias() += weight * jacobian * jacobian.transpose();
            gradient.noalias() += weight * jacobian * residual;
        }

        bool accepted = false;
        Eigen::Matrix<double, 7, 1> acceptedStep =
            Eigen::Matrix<double, 7, 1>::Zero();
        if (fixedAxis) {
            for (const Eigen::Index index : {Eigen::Index(3), Eigen::Index(4)}) {
                normal.row(index).setZero();
                normal.col(index).setZero();
                normal(index, index) = 1.0;
                gradient[index] = 0.0;
            }
        }
        for (std::size_t attempt = 0; attempt < 12U; ++attempt) {
            Eigen::Matrix<double, 7, 7> damped = normal;
            for (Eigen::Index i = 0; i < 7; ++i) {
                damped(i, i) += damping * std::max(normal(i, i), 1.0e-12);
            }
            const Eigen::Matrix<double, 7, 1> step = damped.ldlt().solve(-gradient);
            if (!step.allFinite()) {
                damping *= 10.0;
                continue;
            }
            TorusSurface candidate = model;
            candidate.center += Vec3d{step[0], step[1], step[2]};
            candidate.axisDirection = normalized(model.axisDirection +
                first * step[3] + second * step[4]);
            candidate.majorRadius += step[5];
            candidate.minorRadius += step[6];
            if (!(candidate.majorRadius > 0.0) || !(candidate.minorRadius > 0.0) ||
                norm(candidate.axisDirection) == 0.0) {
                damping *= 10.0;
                continue;
            }
            candidate.axisDirection = canonicalDirection(candidate.axisDirection);
            const double candidateCost = huberCost(samples.points, threshold,
                [&candidate](const Vec3d& point) { return torusResidual(candidate, point); });
            if (candidateCost <= cost) {
                model = candidate;
                cost = candidateCost;
                acceptedStep = step;
                damping = std::max(damping * 0.25, 1.0e-15);
                accepted = true;
                break;
            }
            damping *= 10.0;
        }
        if (!accepted) break;
        const double scale = std::max({model.majorRadius, model.minorRadius,
                                       positionTolerance});
        const double centerStep = std::hypot(
            acceptedStep[0], std::hypot(acceptedStep[1], acceptedStep[2]));
        const double stepMetric = std::max({
            centerStep / scale,
            std::abs(acceptedStep[3]),
            std::abs(acceptedStep[4]),
            std::abs(acceptedStep[5]) / scale,
            std::abs(acceptedStep[6]) / scale});
        if (stepMetric <= options.convergenceTolerance) break;
    }
    return model;
}

[[nodiscard]] Candidate planeCandidate(const Mesh& mesh,
                                       const PatchSamples& samples,
                                       double positionTolerance,
                                       const SurfaceFitOptions& options) {
    Candidate result;
    result.kind = SurfaceKind::Plane;
    try {
        const PlaneSegment fit = fitPlaneTLS(mesh, samples.triangles);
        result.parameters = fit.plane;
        result.residuals = evaluatePlane(
            mesh, samples, fit.plane, positionTolerance, options);
        result.score = modelScore(result.kind, result.residuals,
                                  positionTolerance, options);
        result.accepted = passesTolerance(result.residuals,
                                          positionTolerance, options);
        result.diagnostic = result.accepted ? "ok" : "Plane residual exceeds tolerance";
    } catch (const std::exception& exception) {
        result.diagnostic = exception.what();
    }
    return result;
}

[[nodiscard]] Candidate sphereCandidate(const Mesh& mesh,
                                        const PatchSamples& samples,
                                        double positionTolerance,
                                        const SurfaceFitOptions& options) {
    Candidate result;
    result.kind = SurfaceKind::Sphere;
    SphereFitOptions sphereOptions;
    sphereOptions.maximumIterations = options.maximumRefinementIterations;
    sphereOptions.convergenceTolerance = options.convergenceTolerance;
    sphereOptions.huberThreshold = positionTolerance * options.huberThresholdFactor;
    const SphereFitResult fit = fitSphere(mesh, samples.triangles, sphereOptions);
    if (!fit.valid) {
        result.diagnostic = fit.diagnostic;
        return result;
    }
    const SphereSurface sphere{fit.center, fit.radius};
    result.parameters = sphere;
    result.residuals = evaluateSphere(
        mesh, samples, sphere, positionTolerance, options);
    result.score = modelScore(result.kind, result.residuals,
                              positionTolerance, options);
    result.accepted = passesTolerance(result.residuals,
                                      positionTolerance, options);
    result.diagnostic = result.accepted ? "ok" : "Sphere residual exceeds tolerance";
    return result;
}

[[nodiscard]] Candidate cylinderCandidate(const Mesh& mesh,
                                          const PatchSamples& samples,
                                          double positionTolerance,
                                          const SurfaceFitOptions& options,
                                          const std::vector<Vec3d>& axes) {
    Candidate best;
    best.kind = SurfaceKind::Cylinder;
    best.diagnostic = "No non-degenerate cylinder initialization succeeded";
    for (const Vec3d& axis : axes) {
        CylinderSurface cylinder;
        if (!initializeCylinder(samples, axis, cylinder)) continue;
        cylinder = refineCylinder(samples, cylinder, positionTolerance, options);
        if (!(cylinder.radius > positionTolerance) || !isFinite(cylinder.axisPoint) ||
            !isFinite(cylinder.axisDirection)) {
            continue;
        }
        const SurfaceResidualSummary residuals = evaluateCylinder(
            mesh, samples, cylinder, positionTolerance, options);
        const double score = modelScore(
            SurfaceKind::Cylinder, residuals, positionTolerance, options);
        const bool accepted = passesTolerance(residuals, positionTolerance, options);
        if (!std::isfinite(score) ||
            (best.accepted && !accepted) ||
            (best.accepted == accepted && score >= best.score)) continue;
        best.parameters = cylinder;
        best.residuals = residuals;
        best.score = score;
        best.accepted = accepted;
        best.diagnostic = best.accepted ? "ok" : "Cylinder residual exceeds tolerance";
    }
    return best;
}

[[nodiscard]] Candidate torusCandidate(const Mesh& mesh,
                                       const PatchSamples& samples,
                                       double positionTolerance,
                                       const SurfaceFitOptions& options,
                                       const std::vector<Vec3d>& axes) {
    Candidate best;
    best.kind = SurfaceKind::Torus;
    best.diagnostic = "No non-degenerate torus initialization succeeded";
    std::vector<Vec3d> torusAxes = axes;
    // A deterministic spherical design covers directions missed by local PCA.
    // Use only for torus initialization; fitted dimensions remain data-derived.
    constexpr std::size_t directionCount = 32U;
    constexpr double goldenAngle = 2.39996322972865332;
    for (std::size_t i = 0; i < directionCount; ++i) {
        const double z = 1.0 - 2.0 * (static_cast<double>(i) + 0.5) / directionCount;
        const double rho = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double angle = goldenAngle * static_cast<double>(i);
        appendAxis(torusAxes, {rho * std::cos(angle), rho * std::sin(angle), z});
    }
    for (const Vec3d& axis : torusAxes) {
        TorusSurface torus;
        if (!initializeTorus(mesh, samples, axis, torus)) continue;
        torus = refineTorus(samples, torus, positionTolerance, options, true);
        torus = refineTorus(samples, torus, positionTolerance, options);
        if (!(torus.majorRadius > positionTolerance) ||
            !(torus.minorRadius > positionTolerance) ||
            torus.majorRadius <= torus.minorRadius ||
            torus.majorRadius > mesh.bounds.diagonal() * 100.0 ||
            !isFinite(torus.center) || !isFinite(torus.axisDirection)) {
            continue;
        }
        const SurfaceResidualSummary residuals = evaluateTorus(
            mesh, samples, torus, positionTolerance, options);
        const double score = modelScore(
            SurfaceKind::Torus, residuals, positionTolerance, options);
        const bool accepted = passesTolerance(residuals, positionTolerance, options);
        if (!std::isfinite(score) ||
            (best.accepted && !accepted) ||
            (best.accepted == accepted && score >= best.score)) continue;
        best.parameters = torus;
        best.residuals = residuals;
        best.score = score;
        best.accepted = accepted;
        best.diagnostic = best.accepted ? "ok" : "Torus residual exceeds tolerance";
    }
    return best;
}

// Initialize a surface of revolution from radial normal lines, then regress
// radius against axial position. Refine the six cone parameters by robust LM
// using orthogonal point-to-cone distances, independent of pose and units.
[[nodiscard]] Candidate coneCandidate(
    const Mesh& mesh, const PatchSamples& samples, double tolerance,
    const SurfaceFitOptions& options, const std::vector<Vec3d>& axes) {
    Candidate best;
    best.kind = SurfaceKind::Cone;
    std::vector<std::pair<double, ConeSurface>> initializations;
    for (const auto& direction : axes) {
        const Vec3d axis = canonicalDirection(direction);
        const auto [x, y] = perpendicularFrame(axis);
        Eigen::Matrix2d matrix = Eigen::Matrix2d::Zero();
        Eigen::Vector2d right = Eigen::Vector2d::Zero();
        for (const auto& sample : samples.points) {
            const Vec3d d = sample.point - samples.centroid;
            const Eigen::Vector2d row(dot(sample.normal, y), -dot(sample.normal, x));
            matrix.noalias() += sample.weight * row * row.transpose();
            right.noalias() += sample.weight * row *
                (row[0] * dot(d, x) + row[1] * dot(d, y));
        }
        const Eigen::Vector2d center = matrix.completeOrthogonalDecomposition().solve(right);
        if (!center.allFinite()) continue;
        const Vec3d axisPoint = samples.centroid + x * center[0] + y * center[1];
        Eigen::Matrix2d regression = Eigen::Matrix2d::Zero();
        Eigen::Vector2d values = Eigen::Vector2d::Zero();
        for (const auto& sample : samples.points) {
            const Vec3d d = sample.point - axisPoint;
            const double z = dot(d, axis);
            const Eigen::Vector2d row(1.0, z);
            regression.noalias() += sample.weight * row * row.transpose();
            values.noalias() += sample.weight * row * norm(d - axis * z);
        }
        const Eigen::Vector2d line = regression.ldlt().solve(values);
        if (!line.allFinite() || std::abs(line[1]) < 0.005 || line[0] <= 0.0) continue;
        ConeSurface cone{axisPoint - axis * (line[0] / line[1]),
                         axis * (line[1] > 0.0 ? 1.0 : -1.0),
                         std::atan(std::abs(line[1]))};
        const double cost = huberCost(samples.points, tolerance,
            [&](const Vec3d& p) { return coneResidual(cone, p); });
        if (std::isfinite(cost)) initializations.emplace_back(cost, cone);
    }
    std::sort(initializations.begin(), initializations.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    if (initializations.size() > 4) initializations.resize(4);
    const double scale = std::max(mesh.bounds.diagonal(), tolerance);
    for (const auto& initialization : initializations) {
        ConeSurface model = initialization.second;
        double cost = initialization.first;
        double damping = 1.0e-5;
        for (std::size_t iteration = 0; iteration < options.maximumRefinementIterations; ++iteration) {
            const auto [x, y] = perpendicularFrame(model.axisDirection);
            const auto displaced = [&](const Eigen::Matrix<double, 6, 1>& step) {
                ConeSurface c = model;
                c.apex += Vec3d{step[0], step[1], step[2]} * scale;
                c.axisDirection = normalized(c.axisDirection + x * step[3] + y * step[4]);
                c.semiAngle += step[5];
                return c;
            };
            Eigen::Matrix<double, 6, 6> matrix = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
            for (const auto& sample : samples.points) {
                const double residual = coneResidual(model, sample.point);
                Eigen::Matrix<double, 6, 1> jacobian;
                for (int j = 0; j < 6; ++j) {
                    Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
                    delta[j] = 1.0e-6;
                    jacobian[j] = (coneResidual(displaced(delta), sample.point) -
                                   coneResidual(displaced(-delta), sample.point)) / 2.0e-6;
                }
                const double weight = sample.weight * huberWeight(residual, tolerance);
                matrix.noalias() += weight * jacobian * jacobian.transpose();
                gradient.noalias() += weight * jacobian * residual;
            }
            bool improved = false;
            for (unsigned attempt = 0; attempt < 12; ++attempt) {
                Eigen::Matrix<double, 6, 6> damped = matrix;
                for (int j = 0; j < 6; ++j) damped(j, j) += damping * std::max(matrix(j, j), 1.0e-12);
                const Eigen::Matrix<double, 6, 1> step = damped.ldlt().solve(-gradient);
                if (!step.allFinite()) { damping *= 10.0; continue; }
                const ConeSurface next = displaced(step);
                if (next.semiAngle <= 0.005 || next.semiAngle >= kPi * 0.5 - 0.005) {
                    damping *= 10.0;
                    continue;
                }
                const double nextCost = huberCost(samples.points, tolerance,
                    [&](const Vec3d& p) { return coneResidual(next, p); });
                if (nextCost <= cost) {
                    model = next;
                    cost = nextCost;
                    damping = std::max(1.0e-15, damping * 0.25);
                    improved = step.norm() > options.convergenceTolerance;
                    break;
                }
                damping *= 10.0;
            }
            if (!improved) break;
        }
        Candidate candidate;
        candidate.kind = SurfaceKind::Cone;
        candidate.parameters = model;
        candidate.residuals = evaluateModel(mesh, samples, tolerance,
            options.normalToleranceRadians, options.smoothNormalToleranceRadians,
            [&](const Vec3d& p) { return coneResidual(model, p); },
            [&](const Vec3d& p) { return coneNormal(model, p); });
        candidate.score = modelScore(candidate.kind, candidate.residuals, tolerance, options);
        candidate.accepted = passesTolerance(candidate.residuals, tolerance, options);
        candidate.diagnostic = candidate.accepted ? "ok" : "Cone residual exceeds tolerance";
        if (candidate.accepted > best.accepted ||
            (candidate.accepted == best.accepted && candidate.score < best.score)) best = candidate;
    }
    return best;
}

[[nodiscard]] std::vector<Candidate> analyticCandidates(
    const Mesh& mesh,
    const PatchSamples& samples,
    double positionTolerance,
    const SurfaceFitOptions& options,
    std::size_t maximumAxisHypotheses = 0U) {
    std::vector<Candidate> result;
    result.reserve(5U);
    const auto canImprove = [&](SurfaceKind kind) {
        const double observations = static_cast<double>(std::max<std::size_t>(
            2, samples.points.size() + samples.triangles.size()));
        const double lowerBound = options.complexityPenaltyWeight *
            static_cast<double>(parameterCount(kind)) * std::log(observations) / observations;
        return std::none_of(result.begin(), result.end(), [&](const Candidate& c) {
            return c.accepted && c.score <= lowerBound;
        });
    };
    if (options.enablePlanes) {
        result.push_back(planeCandidate(mesh, samples, positionTolerance, options));
        if (result.back().accepted &&
            result.back().residuals.positionMax < positionTolerance * 0.05 &&
            result.back().residuals.normalMaxRadians < 1.0e-4) return result;
    }
    if (options.enableSpheres) {
        result.push_back(sphereCandidate(mesh, samples, positionTolerance, options));
        if (result.back().accepted &&
            result.back().residuals.positionMax < positionTolerance * 0.05) return result;
    }
    std::vector<Vec3d> axes;
    if (options.enableCylinders || options.enableCones || options.enableTori) {
        axes = candidateAxes(mesh, samples);
        if (maximumAxisHypotheses > 0U && axes.size() > maximumAxisHypotheses) {
            axes.resize(maximumAxisHypotheses);
        }
    }
    if (options.enableCylinders) {
        if (!canImprove(SurfaceKind::Cylinder)) return result;
        result.push_back(cylinderCandidate(
            mesh, samples, positionTolerance, options, axes));
        if (result.back().accepted &&
            result.back().residuals.positionMax < positionTolerance * 0.05) return result;
    }
    if (options.enableCones) {
        if (!canImprove(SurfaceKind::Cone)) return result;
        result.push_back(coneCandidate(mesh, samples, positionTolerance, options, axes));
        if (result.back().accepted &&
            result.back().residuals.positionMax < positionTolerance * 0.05) return result;
    }
    if (options.enableTori) {
        if (!canImprove(SurfaceKind::Torus)) return result;
        result.push_back(torusCandidate(
            mesh, samples, positionTolerance, options, axes));
    }
    return result;
}

[[nodiscard]] double candidatePositionResidual(const Candidate& candidate,
                                               const Vec3d& point) {
    switch (candidate.kind) {
    case SurfaceKind::Plane:
        return std::get<Plane>(candidate.parameters).signedDistance(point);
    case SurfaceKind::Cylinder:
        return cylinderResidual(std::get<CylinderSurface>(candidate.parameters), point);
    case SurfaceKind::Cone:
        return coneResidual(std::get<ConeSurface>(candidate.parameters), point);
    case SurfaceKind::Sphere: {
        const SphereSurface& sphere = std::get<SphereSurface>(candidate.parameters);
        return norm(point - sphere.center) - sphere.radius;
    }
    case SurfaceKind::Torus:
        return torusResidual(std::get<TorusSurface>(candidate.parameters), point);
    default:
        return std::numeric_limits<double>::infinity();
    }
}

[[nodiscard]] Vec3d candidateNormal(const Candidate& candidate, const Vec3d& point) {
    switch (candidate.kind) {
    case SurfaceKind::Plane:
        return std::get<Plane>(candidate.parameters).normal;
    case SurfaceKind::Cylinder:
        return cylinderNormal(std::get<CylinderSurface>(candidate.parameters), point);
    case SurfaceKind::Cone:
        return coneNormal(std::get<ConeSurface>(candidate.parameters), point);
    case SurfaceKind::Sphere:
        return normalized(point - std::get<SphereSurface>(candidate.parameters).center);
    case SurfaceKind::Torus:
        return torusNormal(std::get<TorusSurface>(candidate.parameters), point);
    default:
        return {};
    }
}

[[nodiscard]] bool triangleFitsCandidate(const Mesh& mesh,
                                         std::uint32_t triangleIndex,
                                         const Candidate& candidate,
                                         double positionTolerance,
                                         double normalTolerance,
                                         double smoothNormalTolerance) {
    if (triangleIndex >= mesh.triangles.size() ||
        candidate.parameters.index() == 0U) {
        return false;
    }
    const TriangleFeatures& feature = mesh.features[triangleIndex];
    if (feature.degenerate || !(feature.area > 0.0)) return false;
    const Vec3d predicted = normalized(candidateNormal(candidate, feature.centroid));
    if (norm(predicted) == 0.0 || !isFinite(predicted)) return false;
    const double angle = std::acos(clampUnit(std::abs(dot(feature.normal, predicted))));
    double variation = 0.0;
    for (const auto vertex : mesh.triangles[triangleIndex].vertices) {
        const Vec3d expected = normalized(candidateNormal(candidate, mesh.vertices[vertex]));
        variation = std::max(variation, std::acos(clampUnit(dot(expected, predicted))));
    }
    if (!std::isfinite(angle) || angle > normalTolerance + variation) return false;
    for (const std::uint32_t vertex : mesh.triangles[triangleIndex].vertices) {
        const double residual = std::abs(candidatePositionResidual(
            candidate, mesh.vertices[vertex]));
        if (!std::isfinite(residual) || residual > positionTolerance) return false;
    }
    if (mesh.cornerNormals.size() == mesh.triangles.size()) {
        for (std::size_t corner = 0; corner < 3; ++corner) {
            const Vec3d observed = normalized(mesh.cornerNormals[triangleIndex][corner]);
            if (norm(observed) == 0.0) continue;
            const Vec3d expected = normalized(candidateNormal(candidate,
                mesh.vertices[mesh.triangles[triangleIndex][corner]]));
            if (std::acos(clampUnit(std::abs(dot(observed, expected)))) >
                std::min(normalTolerance, smoothNormalTolerance)) return false;
        }
    }
    return true;
}

[[nodiscard]] std::vector<std::vector<std::uint32_t>> connectedComponents(
    const Mesh& mesh, std::span<const std::uint32_t> triangles) {
    if (triangles.empty()) return {};
    if (mesh.triangleNeighbors.size() != mesh.triangles.size()) {
        return {std::vector<std::uint32_t>(triangles.begin(), triangles.end())};
    }
    std::vector<unsigned char> selected(mesh.triangles.size(), 0U);
    for (const std::uint32_t triangle : triangles) {
        if (triangle < selected.size()) selected[triangle] = 1U;
    }
    std::vector<unsigned char> visited(mesh.triangles.size(), 0U);
    std::vector<std::vector<std::uint32_t>> result;
    std::vector<std::uint32_t> queue;
    queue.reserve(triangles.size());
    for (const std::uint32_t start : triangles) {
        if (start >= selected.size() || !selected[start] || visited[start]) continue;
        result.emplace_back();
        std::vector<std::uint32_t>& component = result.back();
        queue.clear();
        queue.push_back(start);
        visited[start] = 1U;
        for (std::size_t head = 0; head < queue.size(); ++head) {
            const std::uint32_t current = queue[head];
            component.push_back(current);
            for (const std::uint32_t neighbor : mesh.triangleNeighbors[current]) {
                if (neighbor < selected.size() && selected[neighbor] && !visited[neighbor]) {
                    visited[neighbor] = 1U;
                    queue.push_back(neighbor);
                }
            }
        }
        std::sort(component.begin(), component.end());
    }
    std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.front() < rhs.front();
    });
    return result;
}

[[nodiscard]] std::vector<std::uint32_t> localNeighborhood(
    const Mesh& mesh,
    const std::vector<unsigned char>& regionMembership,
    std::uint32_t seed,
    std::size_t maximumTriangles) {
    if (seed >= mesh.triangles.size() || !regionMembership[seed]) return {};
    if (mesh.triangleNeighbors.size() != mesh.triangles.size()) return {seed};
    std::vector<unsigned char> visited(mesh.triangles.size(), 0U);
    std::vector<std::uint32_t> queue;
    queue.reserve(maximumTriangles);
    queue.push_back(seed);
    visited[seed] = 1U;
    for (std::size_t head = 0;
         head < queue.size() && queue.size() < maximumTriangles; ++head) {
        const std::uint32_t current = queue[head];
        for (const std::uint32_t neighbor : mesh.triangleNeighbors[current]) {
            if (queue.size() >= maximumTriangles) break;
            if (neighbor < regionMembership.size() && regionMembership[neighbor] &&
                !visited[neighbor]) {
                visited[neighbor] = 1U;
                queue.push_back(neighbor);
            }
        }
    }
    std::sort(queue.begin(), queue.end());
    return queue;
}

[[nodiscard]] std::uint64_t triangleSetSignature(
    std::span<const std::uint32_t> triangles) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const std::uint32_t triangle : triangles) {
        hash ^= static_cast<std::uint64_t>(triangle) + 0x9e3779b97f4a7c15ULL;
        hash *= 1099511628211ULL;
    }
    hash ^= static_cast<std::uint64_t>(triangles.size());
    return hash;
}

[[nodiscard]] Candidate seedPlaneCandidate(const Mesh& mesh,
                                           std::uint32_t triangleIndex) {
    Candidate result;
    result.kind = SurfaceKind::Plane;
    if (triangleIndex >= mesh.features.size()) return result;
    const TriangleFeatures& feature = mesh.features[triangleIndex];
    if (!feature.degenerate) {
        result.parameters = Plane{feature.normal, feature.planeOffset};
    }
    return result;
}

[[nodiscard]] SurfacePatchFit bestExtractedPatch(
    const Mesh& mesh,
    const std::vector<std::uint32_t>& region,
    const SurfaceFitOptions& options,
    double positionTolerance,
    double minimumArea) {
    SurfacePatchFit best;
    if (region.size() < options.minimumTriangleCount || region.empty()) return best;

    std::vector<unsigned char> membership(mesh.triangles.size(), 0U);
    for (const std::uint32_t triangle : region) membership[triangle] = 1U;
    const std::size_t seedCount = std::min(
        options.maximumHypothesisSeeds, region.size());
    std::unordered_set<std::uint64_t> testedComponents;

    const auto considerCandidate = [&](const Candidate& hypothesis,
                                       SurfacePatchFit& currentBest) {
        if (hypothesis.parameters.index() == 0U) return;
        std::vector<std::uint32_t> inliers;
        inliers.reserve(region.size());
        const double positionThreshold = positionTolerance *
            options.hypothesisInlierToleranceFactor;
        const double normalThreshold = std::min(
            options.maximumNormalResidualRadians,
            options.normalToleranceRadians * options.hypothesisInlierToleranceFactor);
        for (const std::uint32_t triangle : region) {
            if (triangleFitsCandidate(mesh, triangle, hypothesis,
                                      positionThreshold, normalThreshold,
                                      options.smoothNormalToleranceRadians * options.hypothesisInlierToleranceFactor)) {
                inliers.push_back(triangle);
            }
        }
        for (const std::vector<std::uint32_t>& component :
             connectedComponents(mesh, inliers)) {
            if (component.size() < options.minimumTriangleCount ||
                component.size() == region.size()) {
                continue;
            }
            const std::uint64_t signature = triangleSetSignature(component);
            if (!testedComponents.insert(signature).second) continue;
            double area = 0.0;
            for (const std::uint32_t triangle : component) {
                area += mesh.features[triangle].area;
            }
            if (area < minimumArea) continue;
            SurfacePatchFit refit = fitSurfacePatch(mesh, component, options);
            if (!refit.accepted) continue;
            if (!currentBest.accepted || refit.residuals.area > currentBest.residuals.area ||
                (refit.residuals.area == currentBest.residuals.area &&
                 refit.score < currentBest.score)) {
                currentBest = std::move(refit);
            }
        }
    };

    const std::array<std::size_t, 3> neighborhoodSizes{
        std::max<std::size_t>(4U, options.hypothesisNeighborhoodTriangles / 2U),
        options.hypothesisNeighborhoodTriangles,
        options.hypothesisNeighborhoodTriangles * 2U};
    for (std::size_t seedOrdinal = 0; seedOrdinal < seedCount; ++seedOrdinal) {
        const std::size_t regionOrdinal = seedCount == 1U
            ? 0U
            : seedOrdinal * (region.size() - 1U) / (seedCount - 1U);
        const std::uint32_t seed = region[regionOrdinal];
        if (options.enablePlanes) considerCandidate(seedPlaneCandidate(mesh, seed), best);

        std::size_t previousSize = 0U;
        for (const std::size_t requestedSize : neighborhoodSizes) {
            const std::vector<std::uint32_t> neighborhood = localNeighborhood(
                mesh, membership, seed, std::min(requestedSize, region.size()));
            if (neighborhood.size() < options.minimumTriangleCount ||
                neighborhood.size() == previousSize) {
                continue;
            }
            previousSize = neighborhood.size();
            const PatchSamples localSamples = collectSamples(mesh, neighborhood);
            if (!(localSamples.area > 0.0)) continue;

            SurfaceFitOptions localOptions = options;
            localOptions.minimumPatchArea = 0.0;
            localOptions.minimumPatchAreaFraction = 0.0;
            localOptions.minimumInlierAreaFraction = std::min(
                options.minimumInlierAreaFraction, 0.75);
            localOptions.maximumRefinementIterations = std::min<std::size_t>(
                options.maximumRefinementIterations, 24U);
            const std::vector<Candidate> hypotheses = analyticCandidates(
                mesh, localSamples, positionTolerance, localOptions, 12U);
            for (const Candidate& hypothesis : hypotheses) {
                if (std::isfinite(hypothesis.score)) considerCandidate(hypothesis, best);
            }
        }
    }
    return best;
}

[[nodiscard]] std::vector<SurfacePatchFit> fitAndSplitRegion(
    const Mesh& mesh,
    const std::vector<std::uint32_t>& inputRegion,
    const SurfaceFitOptions& options) {
    std::vector<SurfacePatchFit> result;
    std::vector<std::vector<std::uint32_t>> pending =
        connectedComponents(mesh, inputRegion);
    const double totalMeshArea = std::accumulate(
        mesh.features.begin(), mesh.features.end(), 0.0,
        [](double sum, const TriangleFeatures& feature) {
            return sum + (feature.degenerate ? 0.0 : feature.area);
        });
    const double positionTolerance = derivedPositionTolerance(mesh, options);

    while (!pending.empty()) {
        std::vector<std::uint32_t> region = std::move(pending.back());
        pending.pop_back();
        SurfacePatchFit direct = fitSurfacePatch(mesh, region, options);
        if (direct.accepted || !options.splitMixedRegions ||
            result.size() + pending.size() + 1U >=
                options.maximumExtractedPatchesPerRegion) {
            result.push_back(std::move(direct));
            continue;
        }

        double regionArea = 0.0;
        for (const std::uint32_t triangle : region) {
            regionArea += mesh.features[triangle].area;
        }
        const double minimumArea = std::max({
            options.minimumPatchArea,
            totalMeshArea * options.minimumPatchAreaFraction,
            regionArea * options.minimumExtractedAreaFraction});
        SurfacePatchFit extracted = bestExtractedPatch(
            mesh, region, options, positionTolerance, minimumArea);
        if (!extracted.accepted || extracted.triangleIndices.empty()) {
            result.push_back(std::move(direct));
            continue;
        }

        std::vector<unsigned char> removed(mesh.triangles.size(), 0U);
        for (const std::uint32_t triangle : extracted.triangleIndices) {
            removed[triangle] = 1U;
        }
        std::vector<std::uint32_t> remainder;
        remainder.reserve(region.size() - extracted.triangleIndices.size());
        for (const std::uint32_t triangle : region) {
            if (!removed[triangle]) remainder.push_back(triangle);
        }
        result.push_back(std::move(extracted));
        for (std::vector<std::uint32_t>& component :
             connectedComponents(mesh, remainder)) {
            pending.push_back(std::move(component));
        }
    }

    std::sort(result.begin(), result.end(), [](const SurfacePatchFit& lhs,
                                               const SurfacePatchFit& rhs) {
        const std::uint32_t left = lhs.triangleIndices.empty()
            ? std::numeric_limits<std::uint32_t>::max()
            : lhs.triangleIndices.front();
        const std::uint32_t right = rhs.triangleIndices.empty()
            ? std::numeric_limits<std::uint32_t>::max()
            : rhs.triangleIndices.front();
        return left < right;
    });
    return result;
}

// Minimize an area-weighted surface data term plus boundary length using
// metric Potts alpha expansion (Boykov/Veksler/Zabih, PAMI 2001).
// Complexity is charged per supporting observation, so tiny planar facets do
// not defeat a well-supported curved surface merely by interpolating vertices.
std::vector<SurfacePatchFit> refineSurfaceLabels(
    const Mesh& mesh, const std::vector<SurfacePatchFit>& seeds,
    const SurfaceFitOptions& options) {
    const auto familyOptions = [&](SurfaceKind kind) {
        auto fit = options;
        fit.enablePlanes = kind == SurfaceKind::Plane;
        fit.enableSpheres = kind == SurfaceKind::Sphere;
        fit.enableCylinders = kind == SurfaceKind::Cylinder;
        fit.enableCones = kind == SurfaceKind::Cone;
        fit.enableTori = kind == SurfaceKind::Torus;
        return fit;
    };
    std::vector<SurfacePatchFit> ordered = seeds;
    std::stable_sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
        return a.residuals.area > b.residuals.area;
    });
    std::vector<SurfacePatchFit> models;
    for (const auto& seed : ordered) {
        if (!seed.accepted) continue;
        bool combined = false;
        for (auto& model : models) {
            if (model.kind != seed.kind) continue;
            Candidate candidate;
            candidate.kind = model.kind;
            candidate.parameters = model.parameters;
            double supportedArea = 0.0;
            for (const auto triangle : seed.triangleIndices) {
                if (triangleFitsCandidate(mesh, triangle, candidate, options.positionTolerance * 2.0,
                                          options.normalToleranceRadians,
                                          options.smoothNormalToleranceRadians * options.hypothesisInlierToleranceFactor)) supportedArea += mesh.features[triangle].area;
            }
            if (supportedArea < seed.residuals.area * options.minimumInlierAreaFraction) continue;
            auto triangles = model.triangleIndices;
            triangles.insert(triangles.end(), seed.triangleIndices.begin(), seed.triangleIndices.end());
            auto fit = fitSurfacePatch(mesh, triangles, familyOptions(model.kind));
            if (!fit.accepted) continue;
            model = std::move(fit);
            combined = true;
            break;
        }
        if (!combined) models.push_back(seed);
    }
    if (models.empty()) return seeds;
    const std::size_t count = mesh.triangles.size();
    std::vector<bool> included(count, false);
    for (const auto& seed : seeds) for (const auto triangle : seed.triangleIndices) included[triangle] = true;
    std::vector<double> costs(count * models.size(), std::numeric_limits<double>::infinity());
    std::vector<int> labels(count, -1);
    for (std::size_t triangle = 0; triangle < count; ++triangle) {
        if (!included[triangle]) continue;
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t modelIndex = 0; modelIndex < models.size(); ++modelIndex) {
            const auto& fit = models[modelIndex];
            Candidate model;
            model.kind = fit.kind;
            model.parameters = fit.parameters;
            double squares = 0.0, maximum = 0.0;
            for (const auto vertex : mesh.triangles[triangle].vertices) {
                const double error = std::abs(candidatePositionResidual(model, mesh.vertices[vertex]));
                squares += error * error;
                maximum = std::max(maximum, error);
            }
            if (maximum > options.positionTolerance * 2.0) continue;
            const Vec3d expected = normalized(candidateNormal(model, mesh.features[triangle].centroid));
            double angle = std::acos(clampUnit(std::abs(dot(expected, mesh.features[triangle].normal))));
            double observationTolerance = options.normalToleranceRadians;
            if (mesh.cornerNormals.size() == count) {
                const auto& normals = mesh.cornerNormals[triangle];
                double normalSquares = 0.0;
                unsigned normalCount = 0;
                for (std::size_t corner = 0; corner < 3; ++corner) {
                    const Vec3d observed = normalized(normals[corner]);
                    if (norm(observed) == 0.0) continue;
                    const Vec3d predicted = normalized(candidateNormal(model,
                        mesh.vertices[mesh.triangles[triangle][corner]]));
                    const double error = std::acos(clampUnit(std::abs(dot(observed, predicted))));
                    normalSquares += error * error;
                    ++normalCount;
                }
                if (normalCount == 3) {
                    angle = std::sqrt(normalSquares / normalCount);
                    observationTolerance = options.smoothNormalToleranceRadians;
                }
            }
            // Triangle counts overstate evidence on tessellator slivers. Charge
            // model complexity per physical support area at the fit resolution.
            const double complexity = options.complexityPenaltyWeight *
                static_cast<double>(parameterCount(fit.kind)) * options.positionTolerance *
                mesh.bounds.diagonal() / std::max(fit.residuals.area, options.positionTolerance * options.positionTolerance);
            const double value = (squares / 3.0 + maximum * maximum) * 0.5 /
                (options.positionTolerance * options.positionTolerance) + complexity +
                options.normalScoreWeight * std::min(4.0, angle * angle /
                    (observationTolerance * observationTolerance));
            costs[triangle * models.size() + modelIndex] = value;
            if (value < best) { best = value; labels[triangle] = static_cast<int>(modelIndex); }
        }
    }
    // Model selection chooses the supported hypotheses. Boundary placement
    // then minimizes geometric error without repeatedly charging activation
    // cost to each triangle (which would bias a tangent seam toward large faces).
    std::vector<bool> active(models.size(), false);
    for (const auto label : labels) if (label >= 0) active[label] = true;
    double meanArea = 0.0;
    std::size_t includedCount = 0;
    for (std::size_t triangle = 0; triangle < count; ++triangle) {
        if (included[triangle]) { meanArea += mesh.features[triangle].area; ++includedCount; }
    }
    meanArea /= static_cast<double>(std::max<std::size_t>(1, includedCount));
    const double edgeScale = std::sqrt(meanArea);
    for (std::size_t model = 0; model < models.size(); ++model) {
        const double complexity = options.complexityPenaltyWeight *
            static_cast<double>(parameterCount(models[model].kind)) * options.positionTolerance *
            mesh.bounds.diagonal() / std::max(models[model].residuals.area, options.positionTolerance * options.positionTolerance);
        for (std::size_t triangle = 0; triangle < count; ++triangle) {
            auto& cost = costs[triangle * models.size() + model];
            cost = active[model] ? std::max(0.0, cost - complexity) * mesh.features[triangle].area / meanArea
                                 : std::numeric_limits<double>::infinity();
        }
    }
    std::vector<PottsEdge> edges;
    for (std::size_t triangle = 0; triangle < count; ++triangle) {
        if (!included[triangle]) continue;
        for (const auto neighbor : mesh.triangleNeighbors[triangle]) {
            if (neighbor <= triangle || !included[neighbor]) continue;
            std::array<std::uint32_t, 2> shared{};
            unsigned found = 0;
            for (const auto a : mesh.triangles[triangle].vertices) {
                for (const auto b : mesh.triangles[neighbor].vertices) {
                    if (a == b && found < 2) shared[found++] = a;
                }
            }
            if (found == 2) edges.push_back({triangle, neighbor,
                options.boundaryRegularizationWeight *
                norm(mesh.vertices[shared[0]] - mesh.vertices[shared[1]]) / edgeScale});
        }
    }
    labels = minimizePottsEnergy(costs, models.size(), edges, labels);
    std::vector<std::vector<std::uint32_t>> groups(models.size() + 1);
    for (std::size_t i = 0; i < count; ++i) {
        if (!included[i]) continue;
        groups[labels[i] < 0 ? models.size() : static_cast<std::size_t>(labels[i])].push_back(static_cast<std::uint32_t>(i));
    }
    std::vector<SurfacePatchFit> result;
    for (std::size_t group = 0; group < groups.size(); ++group) {
        for (const auto& component : connectedComponents(mesh, groups[group])) {
            if (group < models.size()) {
                result.push_back(fitSurfacePatch(mesh, component, familyOptions(models[group].kind)));
            } else {
                auto fitted = fitAndSplitRegion(mesh, component, options);
                result.insert(result.end(), std::make_move_iterator(fitted.begin()), std::make_move_iterator(fitted.end()));
            }
        }
    }
    return result;
}

} // namespace

const char* toString(SurfaceKind kind) noexcept {
    switch (kind) {
    case SurfaceKind::Unknown: return "unknown";
    case SurfaceKind::Plane: return "plane";
    case SurfaceKind::Cylinder: return "cylinder";
    case SurfaceKind::Cone: return "cone";
    case SurfaceKind::Sphere: return "sphere";
    case SurfaceKind::Torus: return "torus";
    case SurfaceKind::BSpline: return "bspline";
    }
    return "unknown";
}

SurfacePatchFit fitSurfacePatch(
    const Mesh& mesh,
    std::span<const std::uint32_t> triangleIndices,
    const SurfaceFitOptions& options) {
    validateOptions(options);
    SurfacePatchFit result;
    const PatchSamples samples = collectSamples(mesh, triangleIndices);
    result.triangleIndices = samples.triangles;
    result.residuals.area = samples.area;

    if (samples.triangles.size() < options.minimumTriangleCount) {
        result.diagnostic = "Patch has fewer than the required number of triangles";
        return result;
    }
    const double totalMeshArea = std::accumulate(
        mesh.features.begin(), mesh.features.end(), 0.0,
        [](double sum, const TriangleFeatures& feature) {
            return sum + (feature.degenerate ? 0.0 : feature.area);
        });
    const double minimumArea = std::max(
        options.minimumPatchArea, totalMeshArea * options.minimumPatchAreaFraction);
    if (!(samples.area > 0.0) || samples.area < minimumArea || samples.points.empty()) {
        result.diagnostic = "Patch has insufficient non-degenerate surface area";
        return result;
    }

    const double positionTolerance = derivedPositionTolerance(mesh, options);
    const std::vector<Candidate> candidates = analyticCandidates(
        mesh, samples, positionTolerance, options);

    const Candidate* best = nullptr;
    for (const Candidate& candidate : candidates) {
        if (!candidate.accepted || !std::isfinite(candidate.score)) continue;
        if (best == nullptr || candidate.score < best->score) best = &candidate;
    }
    if (best == nullptr) {
        std::ostringstream reason;
        reason << "No analytic surface passed tolerance " << positionTolerance;
        for (const auto& candidate : candidates) {
            reason << "; " << toString(candidate.kind) << " rms=" << candidate.residuals.positionRms
                   << " max=" << candidate.residuals.positionMax
                   << " normal=" << candidate.residuals.normalRmsRadians
                   << " inlier=" << candidate.residuals.inlierAreaFraction;
        }
        result.diagnostic = reason.str();
        return result;
    }
    result.kind = best->kind;
    result.parameters = best->parameters;
    result.residuals = best->residuals;
    result.score = best->score;
    result.accepted = true;
    result.diagnostic = best->diagnostic;
    return result;
}

std::vector<SurfacePatchFit> fitSurfacePatches(
    const Mesh& mesh,
    const std::vector<std::vector<std::uint32_t>>& regions,
    const SurfaceFitOptions& options) {
    validateOptions(options);
    SurfaceFitOptions resolvedOptions = options;
    resolvedOptions.positionTolerance = derivedPositionTolerance(mesh, options);
    std::vector<SurfacePatchFit> result;
    if (options.respectSourcePatchBoundaries) {
        for (const auto& region : regions) {
            auto fitted = fitAndSplitRegion(mesh, region, resolvedOptions);
            result.insert(result.end(), std::make_move_iterator(fitted.begin()),
                          std::make_move_iterator(fitted.end()));
        }
        return result;
    }
    // CUDA regions are seeds, not immutable face boundaries. Grow accepted
    // models through adjacent inliers before fitting smaller, noisy fragments.
    // This is deterministic model-based region growing with a bounded refit.
    std::vector<unsigned char> remaining(mesh.triangles.size(), 0);
    std::vector<std::size_t> order(regions.size());
    std::iota(order.begin(), order.end(), 0);
    std::vector<double> areas(regions.size(), 0.0);
    for (std::size_t i = 0; i < regions.size(); ++i) {
        for (const auto triangle : regions[i]) {
            if (triangle >= remaining.size() || remaining[triangle]) throw std::invalid_argument("Overlapping or invalid seed regions");
            remaining[triangle] = 1;
            areas[i] += mesh.features[triangle].area;
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return areas[a] > areas[b]; });
    const auto grow = [&](SurfacePatchFit fit) {
        Candidate model;
        model.kind = fit.kind;
        model.parameters = fit.parameters;
        std::vector<unsigned char> visited(mesh.triangles.size(), 0);
        auto queue = fit.triangleIndices;
        for (const auto triangle : queue) visited[triangle] = 1;
        for (std::size_t head = 0; head < queue.size(); ++head) {
            for (const auto neighbor : mesh.triangleNeighbors[queue[head]]) {
                if (!remaining[neighbor] || visited[neighbor]) continue;
                visited[neighbor] = 1;
                if (triangleFitsCandidate(mesh, neighbor, model,
                    resolvedOptions.positionTolerance * options.hypothesisInlierToleranceFactor,
                    options.normalToleranceRadians * options.hypothesisInlierToleranceFactor,
                    options.smoothNormalToleranceRadians * options.hypothesisInlierToleranceFactor)) queue.push_back(neighbor);
            }
        }
        if (queue.size() > fit.triangleIndices.size()) {
            std::sort(queue.begin(), queue.end());
            const auto refined = fitSurfacePatch(mesh, queue, resolvedOptions);
            if (refined.accepted) fit = refined;
        }
        for (const auto triangle : fit.triangleIndices) remaining[triangle] = 0;
        result.push_back(std::move(fit));
    };
    for (const auto index : order) {
        // A seed smaller than the spatial noise footprint cannot establish a
        // separate surface reliably. Keep its triangles for joint assignment;
        // any region unexplained by existing models is fitted afterwards.
        if (areas[index] < kPi * resolvedOptions.positionTolerance * resolvedOptions.positionTolerance) continue;
        std::vector<std::uint32_t> seed;
        for (const auto triangle : regions[index]) if (remaining[triangle]) seed.push_back(triangle);
        if (seed.size() < options.minimumTriangleCount) continue;
        for (auto component : connectedComponents(mesh, seed)) {
            std::erase_if(component, [&](auto triangle) { return !remaining[triangle]; });
            if (component.empty()) continue;
            auto fit = fitSurfacePatch(mesh, component, resolvedOptions);
            if (fit.accepted) grow(std::move(fit));
        }
    }
    std::vector<std::uint32_t> unresolved;
    for (std::size_t i = 0; i < remaining.size(); ++i) if (remaining[i]) unresolved.push_back(static_cast<std::uint32_t>(i));
    const bool hasHypotheses = !result.empty();
    for (const auto& component : connectedComponents(mesh, unresolved)) {
        if (!hasHypotheses) {
            auto fitted = fitAndSplitRegion(mesh, component, resolvedOptions);
            result.insert(result.end(), std::make_move_iterator(fitted.begin()), std::make_move_iterator(fitted.end()));
        } else {
            // Joint assignment must see the established hypotheses before
            // tiny residual fragments generate poorly constrained new models.
            SurfacePatchFit pending;
            pending.triangleIndices = component;
            for (const auto triangle : component) pending.residuals.area += mesh.features[triangle].area;
            pending.diagnostic = "Awaiting joint surface assignment";
            result.push_back(std::move(pending));
        }
    }
    result = refineSurfaceLabels(mesh, result, resolvedOptions);
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.triangleIndices.front() < b.triangleIndices.front(); });
    return result;
}

std::vector<SurfacePatchFit> fitSurfacePatches(
    const Mesh& mesh,
    std::span<const std::int32_t> triangleLabels,
    const SurfaceFitOptions& options) {
    validateOptions(options);
    if (triangleLabels.size() != mesh.triangles.size()) {
        throw std::invalid_argument(
            "Surface-region label count does not match mesh triangle count");
    }
    std::int32_t maximumLabel = -1;
    for (const std::int32_t label : triangleLabels) {
        if (label > maximumLabel) maximumLabel = label;
    }
    std::vector<std::vector<std::uint32_t>> regions(
        maximumLabel >= 0 ? static_cast<std::size_t>(maximumLabel) + 1U : 0U);
    for (std::size_t triangle = 0; triangle < triangleLabels.size(); ++triangle) {
        const std::int32_t label = triangleLabels[triangle];
        if (label >= 0) {
            regions[static_cast<std::size_t>(label)].push_back(
                static_cast<std::uint32_t>(triangle));
        }
    }
    return fitSurfacePatches(mesh, regions, options);
}

} // namespace recad
