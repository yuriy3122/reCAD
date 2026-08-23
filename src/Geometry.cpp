#include "Geometry.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace recad {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

[[nodiscard]] Eigen::Vector3d eigenVector(const Vec3d& value) noexcept {
    return {value.x, value.y, value.z};
}

[[nodiscard]] Vec3d vec3(const Eigen::Vector3d& value) noexcept {
    return {value.x(), value.y(), value.z()};
}

[[nodiscard]] Eigen::Matrix3d outerProduct(const Vec3d& lhs, const Vec3d& rhs) noexcept {
    return eigenVector(lhs) * eigenVector(rhs).transpose();
}

[[nodiscard]] double clampUnit(double value) noexcept {
    return std::clamp(value, -1.0, 1.0);
}

[[nodiscard]] Vec3d canonicalDirection(Vec3d value) noexcept {
    value = normalized(value);
    const std::array<double, 3> absolute{std::abs(value.x), std::abs(value.y), std::abs(value.z)};
    std::size_t dominant = 0;
    if (absolute[1] > absolute[dominant]) dominant = 1;
    if (absolute[2] > absolute[dominant]) dominant = 2;
    if (value[dominant] < 0.0) value = -value;
    return value;
}

struct EdgeKey {
    std::uint32_t low{};
    std::uint32_t high{};

    [[nodiscard]] bool operator==(const EdgeKey&) const noexcept = default;
};

struct EdgeKeyHash {
    [[nodiscard]] std::size_t operator()(const EdgeKey& key) const noexcept {
        const std::uint64_t packed = (static_cast<std::uint64_t>(key.low) << 32U) |
                                     static_cast<std::uint64_t>(key.high);
        std::uint64_t mixed = packed + 0x9e3779b97f4a7c15ULL;
        mixed = (mixed ^ (mixed >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        mixed = (mixed ^ (mixed >> 27U)) * 0x94d049bb133111ebULL;
        return static_cast<std::size_t>(mixed ^ (mixed >> 31U));
    }
};

struct EdgeUse {
    std::uint32_t triangle{};
    bool lowToHigh{};
};

[[nodiscard]] EdgeKey edgeKey(std::uint32_t lhs, std::uint32_t rhs) noexcept {
    return {std::min(lhs, rhs), std::max(lhs, rhs)};
}

[[nodiscard]] const TriangleFeatures& checkedFeature(const Mesh& mesh, std::size_t index) {
    if (mesh.features.size() != mesh.triangles.size()) {
        throw std::invalid_argument("Mesh triangle features are missing or stale");
    }
    return mesh.features[index];
}

[[nodiscard]] bool triangleFitsPlane(const Mesh& mesh,
                                     std::uint32_t triangleIndex,
                                     const Plane& plane,
                                     double cosineAngle,
                                     double tolerance) {
    const TriangleFeatures& feature = checkedFeature(mesh, triangleIndex);
    if (feature.degenerate || std::abs(dot(feature.normal, plane.normal)) < cosineAngle) {
        return false;
    }
    const Triangle& triangle = mesh.triangles[triangleIndex];
    for (const std::uint32_t vertexIndex : triangle.vertices) {
        if (std::abs(plane.signedDistance(mesh.vertices[vertexIndex])) > tolerance) return false;
    }
    return true;
}

[[nodiscard]] std::vector<std::uint32_t> largestConnectedSubset(
    const Mesh& mesh,
    const std::vector<unsigned char>& eligible,
    const std::vector<unsigned char>& remaining,
    double* areaOut = nullptr) {
    const std::size_t count = mesh.triangles.size();
    std::vector<unsigned char> visited(count, 0U);
    std::vector<std::uint32_t> best;
    double bestArea = 0.0;
    std::queue<std::uint32_t> queue;

    for (std::size_t start = 0; start < count; ++start) {
        if (!eligible[start] || !remaining[start] || visited[start]) continue;
        std::vector<std::uint32_t> component;
        double componentArea = 0.0;
        visited[start] = 1U;
        queue.push(static_cast<std::uint32_t>(start));
        while (!queue.empty()) {
            const std::uint32_t triangle = queue.front();
            queue.pop();
            component.push_back(triangle);
            componentArea += mesh.features[triangle].area;
            if (triangle < mesh.triangleNeighbors.size()) {
                for (const std::uint32_t neighbor : mesh.triangleNeighbors[triangle]) {
                    if (neighbor < count && eligible[neighbor] && remaining[neighbor] && !visited[neighbor]) {
                        visited[neighbor] = 1U;
                        queue.push(neighbor);
                    }
                }
            }
        }
        if (componentArea > bestArea ||
            (componentArea == bestArea && component.size() > best.size())) {
            bestArea = componentArea;
            best = std::move(component);
        }
    }
    if (areaOut != nullptr) *areaOut = bestArea;
    std::sort(best.begin(), best.end());
    return best;
}

[[nodiscard]] std::size_t connectedComponentCount(
    const Mesh& mesh,
    std::span<const std::uint32_t> triangles) {
    if (triangles.empty()) return 0;
    std::unordered_set<std::uint32_t> selected;
    selected.reserve(triangles.size() * 2U);
    for (const std::uint32_t triangle : triangles) selected.insert(triangle);
    std::unordered_set<std::uint32_t> visited;
    visited.reserve(selected.size() * 2U);
    std::queue<std::uint32_t> queue;
    std::size_t components = 0;
    for (const std::uint32_t start : triangles) {
        if (!visited.insert(start).second) continue;
        ++components;
        queue.push(start);
        while (!queue.empty()) {
            const std::uint32_t current = queue.front();
            queue.pop();
            if (current >= mesh.triangleNeighbors.size()) continue;
            for (const std::uint32_t neighbor : mesh.triangleNeighbors[current]) {
                if (selected.contains(neighbor) && visited.insert(neighbor).second) queue.push(neighbor);
            }
        }
    }
    return components;
}

[[nodiscard]] double weightedMedian(std::vector<std::pair<double, double>> values) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) return lhs.first < rhs.first;
        return lhs.second < rhs.second;
    });
    const double total = std::accumulate(values.begin(), values.end(), 0.0,
        [](double sum, const auto& item) { return sum + std::max(0.0, item.second); });
    if (!(total > 0.0)) return values[values.size() / 2U].first;
    double cumulative = 0.0;
    for (const auto& [value, weight] : values) {
        cumulative += std::max(0.0, weight);
        if (cumulative >= total * 0.5) return value;
    }
    return values.back().first;
}

[[nodiscard]] double medianAbsolute(std::span<const double> values) {
    if (values.empty()) return 0.0;
    std::vector<double> absolute;
    absolute.reserve(values.size());
    for (const double value : values) absolute.push_back(std::abs(value));
    const auto middle = absolute.begin() + static_cast<std::ptrdiff_t>(absolute.size() / 2U);
    std::nth_element(absolute.begin(), middle, absolute.end());
    double result = *middle;
    if (absolute.size() % 2U == 0U) {
        const auto lower = std::max_element(absolute.begin(), middle);
        result = 0.5 * (result + *lower);
    }
    return result;
}

[[nodiscard]] double huberCost(std::span<const Eigen::Vector3d> points,
                               std::span<const double> weights,
                               const Eigen::Vector3d& center,
                               double radius,
                               double threshold) {
    double cost = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const double residual = (points[i] - center).norm() - radius;
        const double absolute = std::abs(residual);
        const double loss = absolute <= threshold
            ? 0.5 * residual * residual
            : threshold * (absolute - 0.5 * threshold);
        cost += weights[i] * loss;
    }
    return cost;
}

[[nodiscard]] std::tuple<int, double, double, double> axisOrderKey(const Vec3d& axis) noexcept {
    const Vec3d canonical = canonicalDirection(axis);
    const std::array<double, 3> absolute{std::abs(canonical.x), std::abs(canonical.y),
                                         std::abs(canonical.z)};
    int dominant = 0;
    if (absolute[1] > absolute[static_cast<std::size_t>(dominant)]) dominant = 1;
    if (absolute[2] > absolute[static_cast<std::size_t>(dominant)]) dominant = 2;
    return {dominant, -canonical.x, -canonical.y, -canonical.z};
}

} // namespace

double norm(const Vec3d& value) noexcept {
    return std::sqrt(squaredNorm(value));
}

Vec3d normalized(const Vec3d& value, double minimumNorm) noexcept {
    const double length = norm(value);
    if (!std::isfinite(length) || length <= minimumNorm) return {};
    return value / length;
}

bool isFinite(const Vec3d& value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

void Aabb::expand(const Vec3d& point) noexcept {
    minimum.x = std::min(minimum.x, point.x);
    minimum.y = std::min(minimum.y, point.y);
    minimum.z = std::min(minimum.z, point.z);
    maximum.x = std::max(maximum.x, point.x);
    maximum.y = std::max(maximum.y, point.y);
    maximum.z = std::max(maximum.z, point.z);
}

bool Aabb::valid() const noexcept {
    return isFinite(minimum) && isFinite(maximum) &&
           minimum.x <= maximum.x && minimum.y <= maximum.y && minimum.z <= maximum.z;
}

Vec3d Aabb::center() const noexcept {
    return valid() ? (minimum + maximum) * 0.5 : Vec3d{};
}

Vec3d Aabb::extent() const noexcept {
    return valid() ? maximum - minimum : Vec3d{};
}

double Aabb::diagonal() const noexcept {
    return norm(extent());
}

void computeTriangleFeatures(Mesh& mesh, double degeneracyThreshold) {
    if (degeneracyThreshold < 0.0 || !std::isfinite(degeneracyThreshold)) {
        throw std::invalid_argument("Triangle degeneracy threshold must be finite and nonnegative");
    }
    mesh.features.assign(mesh.triangles.size(), {});
    for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
        const Triangle& triangle = mesh.triangles[i];
        for (const std::uint32_t index : triangle.vertices) {
            if (index >= mesh.vertices.size()) throw std::out_of_range("Triangle vertex index is out of range");
        }
        const Vec3d& a = mesh.vertices[triangle.vertices[0]];
        const Vec3d& b = mesh.vertices[triangle.vertices[1]];
        const Vec3d& c = mesh.vertices[triangle.vertices[2]];
        const Vec3d ab = b - a;
        const Vec3d ac = c - a;
        const Vec3d bc = c - b;
        const Vec3d twiceAreaVector = cross(ab, ac);
        const double twiceArea = norm(twiceAreaVector);

        TriangleFeatures& feature = mesh.features[i];
        feature.centroid = (a + b + c) / 3.0;
        feature.area = 0.5 * twiceArea;
        feature.maxEdgeLength = std::max({norm(ab), norm(ac), norm(bc)});
        feature.degenerate = !std::isfinite(twiceArea) || twiceArea <= degeneracyThreshold;
        if (!feature.degenerate) {
            feature.normal = twiceAreaVector / twiceArea;
            feature.planeOffset = -dot(feature.normal, a);
        }
    }
}

void rebuildTopology(Mesh& mesh) {
    const std::size_t triangleCount = mesh.triangles.size();
    mesh.triangleNeighbors.assign(triangleCount, {});
    mesh.vertexTriangles.assign(mesh.vertices.size(), {});

    std::unordered_map<EdgeKey, std::vector<EdgeUse>, EdgeKeyHash> edges;
    edges.reserve(triangleCount * 3U);
    for (std::size_t triangleIndex = 0; triangleIndex < triangleCount; ++triangleIndex) {
        const Triangle& triangle = mesh.triangles[triangleIndex];
        for (const std::uint32_t vertex : triangle.vertices) {
            if (vertex >= mesh.vertices.size()) throw std::out_of_range("Triangle vertex index is out of range");
            mesh.vertexTriangles[vertex].push_back(static_cast<std::uint32_t>(triangleIndex));
        }
        for (std::size_t edge = 0; edge < 3U; ++edge) {
            const std::uint32_t from = triangle.vertices[edge];
            const std::uint32_t to = triangle.vertices[(edge + 1U) % 3U];
            const EdgeKey key = edgeKey(from, to);
            edges[key].push_back({static_cast<std::uint32_t>(triangleIndex), from == key.low});
        }
    }

    TopologyStats& stats = mesh.stats;
    stats.weldedVertexCount = mesh.vertices.size();
    stats.triangleCount = triangleCount;
    stats.uniqueEdgeCount = edges.size();
    stats.boundaryEdgeCount = 0;
    stats.manifoldEdgeCount = 0;
    stats.nonManifoldEdgeCount = 0;
    stats.consistentlyOriented = true;

    for (const auto& [key, uses] : edges) {
        static_cast<void>(key);
        if (uses.size() == 1U) {
            ++stats.boundaryEdgeCount;
        } else if (uses.size() == 2U) {
            ++stats.manifoldEdgeCount;
            if (uses[0].lowToHigh == uses[1].lowToHigh) stats.consistentlyOriented = false;
        } else {
            ++stats.nonManifoldEdgeCount;
            stats.consistentlyOriented = false;
        }
        for (std::size_t i = 0; i < uses.size(); ++i) {
            for (std::size_t j = i + 1U; j < uses.size(); ++j) {
                const std::uint32_t lhs = uses[i].triangle;
                const std::uint32_t rhs = uses[j].triangle;
                mesh.triangleNeighbors[lhs].push_back(rhs);
                mesh.triangleNeighbors[rhs].push_back(lhs);
            }
        }
    }
    for (auto& neighbors : mesh.triangleNeighbors) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    std::vector<std::int32_t> component(triangleCount, -1);
    std::queue<std::uint32_t> queue;
    std::size_t componentCount = 0;
    for (std::size_t start = 0; start < triangleCount; ++start) {
        if (component[start] >= 0) continue;
        component[start] = static_cast<std::int32_t>(componentCount);
        queue.push(static_cast<std::uint32_t>(start));
        while (!queue.empty()) {
            const std::uint32_t current = queue.front();
            queue.pop();
            for (const std::uint32_t neighbor : mesh.triangleNeighbors[current]) {
                if (component[neighbor] < 0) {
                    component[neighbor] = static_cast<std::int32_t>(componentCount);
                    queue.push(neighbor);
                }
            }
        }
        ++componentCount;
    }
    stats.connectedComponentCount = componentCount;

    std::vector<unsigned char> componentClosed(componentCount, 1U);
    for (const auto& [key, uses] : edges) {
        static_cast<void>(key);
        if (uses.empty()) continue;
        const std::size_t owner = static_cast<std::size_t>(component[uses.front().triangle]);
        if (uses.size() != 2U) componentClosed[owner] = 0U;
    }
    stats.closedComponentCount = static_cast<std::size_t>(
        std::count(componentClosed.begin(), componentClosed.end(), static_cast<unsigned char>(1U)));
    const std::size_t usedVertices = static_cast<std::size_t>(std::count_if(
        mesh.vertexTriangles.begin(), mesh.vertexTriangles.end(), [](const auto& incident) {
            return !incident.empty();
        }));
    stats.eulerCharacteristic = static_cast<std::int64_t>(usedVertices) -
                                static_cast<std::int64_t>(edges.size()) +
                                static_cast<std::int64_t>(triangleCount);
    stats.watertight = triangleCount > 0U && stats.boundaryEdgeCount == 0U &&
                       stats.nonManifoldEdgeCount == 0U && stats.consistentlyOriented;
}

double Plane::distance(const Vec3d& point) const noexcept {
    return std::abs(signedDistance(point));
}

Vec3d Plane::project(const Vec3d& point) const noexcept {
    return point - normal * signedDistance(point);
}

PlaneSegment fitPlaneTLS(const Mesh& mesh,
                         std::span<const std::uint32_t> triangleIndices) {
    if (triangleIndices.empty()) throw std::invalid_argument("Cannot fit a plane to no triangles");
    if (mesh.features.size() != mesh.triangles.size()) {
        throw std::invalid_argument("Mesh triangle features are missing or stale");
    }

    double totalArea = 0.0;
    Eigen::Vector3d firstMoment = Eigen::Vector3d::Zero();
    Eigen::Matrix3d secondMoment = Eigen::Matrix3d::Zero();
    Vec3d orientedNormalSum{};
    for (const std::uint32_t index : triangleIndices) {
        if (index >= mesh.triangles.size()) throw std::out_of_range("Plane-fit triangle index is out of range");
        const TriangleFeatures& feature = mesh.features[index];
        if (feature.degenerate || !(feature.area > 0.0)) continue;
        const Triangle& triangle = mesh.triangles[index];
        const Vec3d& p0 = mesh.vertices[triangle.vertices[0]];
        const Vec3d& p1 = mesh.vertices[triangle.vertices[1]];
        const Vec3d& p2 = mesh.vertices[triangle.vertices[2]];
        const Vec3d sum = p0 + p1 + p2;
        const Eigen::Matrix3d expectation =
            (outerProduct(sum, sum) + outerProduct(p0, p0) +
             outerProduct(p1, p1) + outerProduct(p2, p2)) / 12.0;
        totalArea += feature.area;
        firstMoment += feature.area * eigenVector(feature.centroid);
        secondMoment += feature.area * expectation;
        orientedNormalSum += feature.normal * feature.area;
    }
    if (!(totalArea > std::numeric_limits<double>::min()) || !std::isfinite(totalArea)) {
        throw std::invalid_argument("Plane-fit selection has zero usable area");
    }

    const Eigen::Vector3d centroid = firstMoment / totalArea;
    Eigen::Matrix3d covariance = secondMoment / totalArea - centroid * centroid.transpose();
    covariance = 0.5 * (covariance + covariance.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    if (solver.info() != Eigen::Success) throw std::runtime_error("TLS plane eigensolve failed");
    Vec3d normal = normalized(vec3(solver.eigenvectors().col(0)));
    if (norm(normal) == 0.0) throw std::runtime_error("TLS plane normal is indeterminate");
    if (norm(orientedNormalSum) > totalArea * 1.0e-10) {
        if (dot(normal, orientedNormalSum) < 0.0) normal = -normal;
    } else {
        normal = canonicalDirection(normal);
    }

    PlaneSegment result;
    result.plane.normal = normal;
    result.plane.offset = -dot(normal, vec3(centroid));
    result.triangleIndices.assign(triangleIndices.begin(), triangleIndices.end());
    std::sort(result.triangleIndices.begin(), result.triangleIndices.end());
    result.triangleIndices.erase(
        std::unique(result.triangleIndices.begin(), result.triangleIndices.end()),
        result.triangleIndices.end());
    result.area = totalArea;
    result.centroid = vec3(centroid);
    const Eigen::Vector3d n = eigenVector(normal);
    result.rmsResidual = std::sqrt(std::max(0.0, (n.transpose() * covariance * n).value()));
    result.maxResidual = 0.0;
    for (const std::uint32_t index : result.triangleIndices) {
        for (const std::uint32_t vertex : mesh.triangles[index].vertices) {
            result.maxResidual = std::max(result.maxResidual,
                                          result.plane.distance(mesh.vertices[vertex]));
        }
    }
    result.connectedComponents = connectedComponentCount(mesh, result.triangleIndices);
    return result;
}

PlaneSegmentationResult segmentPlanarRegions(const Mesh& mesh,
                                             const PlaneSegmentationOptions& options) {
    if (mesh.features.size() != mesh.triangles.size()) {
        throw std::invalid_argument("Mesh triangle features are missing or stale");
    }
    if (!std::isfinite(options.distanceTolerance) || options.distanceTolerance < 0.0 ||
        !std::isfinite(options.angleToleranceRadians) || options.angleToleranceRadians <= 0.0 ||
        options.angleToleranceRadians >= kPi * 0.5 ||
        !std::isfinite(options.minimumSupportAreaFraction) ||
        options.minimumSupportAreaFraction < 0.0) {
        throw std::invalid_argument("Invalid plane segmentation tolerances");
    }

    PlaneSegmentationResult result;
    const std::size_t triangleCount = mesh.triangles.size();
    result.triangleLabels.assign(triangleCount, -1);
    result.angleToleranceRadians = options.angleToleranceRadians;
    const double diagonal = std::max(
        mesh.bounds.diagonal(), std::numeric_limits<double>::min());
    result.distanceTolerance = options.distanceTolerance > 0.0
        ? options.distanceTolerance
        : std::max({mesh.weldTolerance * 4.0,
                    diagonal * 2.0e-7,
                    std::numeric_limits<double>::epsilon() * diagonal * 128.0});
    const double cosineAngle = std::cos(options.angleToleranceRadians);
    const double totalArea = std::accumulate(mesh.features.begin(), mesh.features.end(), 0.0,
        [](double sum, const TriangleFeatures& feature) {
            return sum + (feature.degenerate ? 0.0 : feature.area);
        });
    const double minimumArea = std::max(options.minimumSupportArea,
                                        totalArea * options.minimumSupportAreaFraction);

    std::vector<unsigned char> remaining(triangleCount, 1U);
    for (std::size_t i = 0; i < triangleCount; ++i) {
        if (mesh.features[i].degenerate) remaining[i] = 0U;
    }

    std::vector<PlaneSegment> connectedSegments;
    std::vector<unsigned char> eligible(triangleCount, 0U);
    while (true) {
        std::vector<std::uint32_t> best;
        double bestArea = 0.0;
        for (std::size_t seed = 0; seed < triangleCount; ++seed) {
            if (!remaining[seed]) continue;
            const TriangleFeatures& seedFeature = mesh.features[seed];
            const Plane seedPlane{seedFeature.normal, seedFeature.planeOffset};
            std::fill(eligible.begin(), eligible.end(), static_cast<unsigned char>(0));
            for (std::size_t candidate = 0; candidate < triangleCount; ++candidate) {
                if (remaining[candidate] && triangleFitsPlane(
                        mesh, static_cast<std::uint32_t>(candidate), seedPlane,
                        cosineAngle, result.distanceTolerance)) {
                    eligible[candidate] = 1U;
                }
            }
            double componentArea = 0.0;
            std::vector<std::uint32_t> component = largestConnectedSubset(
                mesh, eligible, remaining, &componentArea);
            if (componentArea > bestArea ||
                (componentArea == bestArea && component.size() > best.size())) {
                bestArea = componentArea;
                best = std::move(component);
            }
        }

        if (best.size() < options.minimumTriangleCount || bestArea < minimumArea) break;

        PlaneSegment refined = fitPlaneTLS(mesh, best);
        for (std::size_t iteration = 0; iteration < options.maximumRefinementIterations; ++iteration) {
            std::fill(eligible.begin(), eligible.end(), static_cast<unsigned char>(0));
            for (std::size_t candidate = 0; candidate < triangleCount; ++candidate) {
                if (remaining[candidate] && triangleFitsPlane(
                        mesh, static_cast<std::uint32_t>(candidate), refined.plane,
                        cosineAngle, result.distanceTolerance)) {
                    eligible[candidate] = 1U;
                }
            }
            std::vector<std::uint32_t> reclassified = largestConnectedSubset(
                mesh, eligible, remaining);
            if (reclassified == refined.triangleIndices ||
                reclassified.size() < options.minimumTriangleCount) break;
            PlaneSegment next = fitPlaneTLS(mesh, reclassified);
            if (next.maxResidual > result.distanceTolerance * 1.25) break;
            refined = std::move(next);
        }
        if (refined.area < minimumArea ||
            refined.triangleIndices.size() < options.minimumTriangleCount ||
            refined.maxResidual > result.distanceTolerance * 1.25) {
            // Prevent repeatedly selecting a numerically marginal seed.
            for (const std::uint32_t index : best) remaining[index] = 0U;
            continue;
        }
        for (const std::uint32_t index : refined.triangleIndices) remaining[index] = 0U;
        connectedSegments.push_back(std::move(refined));
    }

    if (options.mergeDisconnectedCoplanarRegions && connectedSegments.size() > 1U) {
        const std::size_t count = connectedSegments.size();
        std::vector<std::size_t> parent(count);
        std::iota(parent.begin(), parent.end(), 0U);
        const auto findRoot = [&parent](std::size_t value) {
            std::size_t root = value;
            while (parent[root] != root) root = parent[root];
            while (parent[value] != value) {
                const std::size_t next = parent[value];
                parent[value] = root;
                value = next;
            }
            return root;
        };
        const double mergeCosine = std::cos(options.angleToleranceRadians);
        for (std::size_t i = 0; i < count; ++i) {
            for (std::size_t j = i + 1U; j < count; ++j) {
                const double alignment = dot(connectedSegments[i].plane.normal,
                                             connectedSegments[j].plane.normal);
                if (std::abs(alignment) < mergeCosine) continue;
                const double secondOffset = alignment >= 0.0
                    ? connectedSegments[j].plane.offset
                    : -connectedSegments[j].plane.offset;
                const double mergeTolerance = result.distanceTolerance * 2.0 +
                    connectedSegments[i].maxResidual + connectedSegments[j].maxResidual;
                if (std::abs(connectedSegments[i].plane.offset - secondOffset) > mergeTolerance) continue;
                const std::size_t rootI = findRoot(i);
                const std::size_t rootJ = findRoot(j);
                if (rootI != rootJ) parent[rootJ] = rootI;
            }
        }
        std::unordered_map<std::size_t, std::vector<std::uint32_t>> groups;
        for (std::size_t i = 0; i < count; ++i) {
            auto& indices = groups[findRoot(i)];
            indices.insert(indices.end(), connectedSegments[i].triangleIndices.begin(),
                           connectedSegments[i].triangleIndices.end());
        }
        result.segments.reserve(groups.size());
        for (auto& [root, indices] : groups) {
            static_cast<void>(root);
            PlaneSegment merged = fitPlaneTLS(mesh, indices);
            if (merged.maxResidual <= result.distanceTolerance * 2.0) {
                result.segments.push_back(std::move(merged));
            } else {
                // A transitive merge can bridge two distinct planes; retain the
                // independently fitted connected pieces in that unlikely case.
                for (std::size_t i = 0; i < count; ++i) {
                    if (findRoot(i) == root) result.segments.push_back(connectedSegments[i]);
                }
            }
        }
    } else {
        result.segments = std::move(connectedSegments);
    }

    std::sort(result.segments.begin(), result.segments.end(), [](const PlaneSegment& lhs,
                                                                 const PlaneSegment& rhs) {
        if (lhs.area != rhs.area) return lhs.area > rhs.area;
        if (lhs.plane.offset != rhs.plane.offset) return lhs.plane.offset < rhs.plane.offset;
        return lhs.triangleIndices < rhs.triangleIndices;
    });
    for (std::size_t label = 0; label < result.segments.size(); ++label) {
        PlaneSegment& segment = result.segments[label];
        segment.connectedComponents = connectedComponentCount(mesh, segment.triangleIndices);
        result.planarArea += segment.area;
        for (const std::uint32_t triangle : segment.triangleIndices) {
            result.triangleLabels[triangle] = static_cast<std::int32_t>(label);
        }
    }
    for (std::size_t triangle = 0; triangle < triangleCount; ++triangle) {
        if (result.triangleLabels[triangle] < 0) {
            result.nonPlanarTriangles.push_back(static_cast<std::uint32_t>(triangle));
            if (!mesh.features[triangle].degenerate) result.nonPlanarArea += mesh.features[triangle].area;
        }
    }
    return result;
}

SphereFitResult fitSphere(const Mesh& mesh,
                          std::span<const std::uint32_t> surfaceTriangles,
                          const SphereFitOptions& options) {
    SphereFitResult result;
    if (surfaceTriangles.empty()) {
        result.diagnostic = "No surface triangles were supplied";
        return result;
    }
    if (mesh.features.size() != mesh.triangles.size()) {
        result.diagnostic = "Mesh triangle features are missing or stale";
        return result;
    }
    if (options.maximumIterations == 0U || options.convergenceTolerance <= 0.0 ||
        !std::isfinite(options.convergenceTolerance) || options.initialDamping <= 0.0 ||
        !std::isfinite(options.initialDamping) || options.huberThreshold < 0.0 ||
        !std::isfinite(options.huberThreshold)) {
        result.diagnostic = "Invalid sphere-fit options";
        return result;
    }

    std::vector<double> vertexWeights(mesh.vertices.size(), 0.0);
    for (const std::uint32_t triangleIndex : surfaceTriangles) {
        if (triangleIndex >= mesh.triangles.size()) {
            result.diagnostic = "Surface triangle index is out of range";
            return result;
        }
        const TriangleFeatures& feature = mesh.features[triangleIndex];
        if (feature.degenerate || !(feature.area > 0.0)) continue;
        const double contribution = feature.area / 3.0;
        for (const std::uint32_t vertex : mesh.triangles[triangleIndex].vertices) {
            vertexWeights[vertex] += contribution;
        }
    }
    std::vector<std::uint32_t> vertexIndices;
    for (std::size_t i = 0; i < vertexWeights.size(); ++i) {
        if (vertexWeights[i] > 0.0) vertexIndices.push_back(static_cast<std::uint32_t>(i));
    }
    result.pointCount = vertexIndices.size();
    if (vertexIndices.size() < 4U) {
        result.diagnostic = "Fewer than four weighted surface vertices are available";
        return result;
    }

    double totalWeight = 0.0;
    Vec3d weightedCenter{};
    for (const std::uint32_t index : vertexIndices) {
        totalWeight += vertexWeights[index];
        weightedCenter += mesh.vertices[index] * vertexWeights[index];
    }
    weightedCenter /= totalWeight;
    const double scale = std::max(mesh.bounds.diagonal(),
                                  std::numeric_limits<double>::epsilon() * 1024.0);
    std::vector<Eigen::Vector3d> points;
    std::vector<double> weights;
    points.reserve(vertexIndices.size());
    weights.reserve(vertexIndices.size());
    for (const std::uint32_t index : vertexIndices) {
        points.push_back(eigenVector((mesh.vertices[index] - weightedCenter) / scale));
        weights.push_back(vertexWeights[index] / totalWeight);
    }

    Eigen::MatrixXd design(static_cast<Eigen::Index>(points.size()), 4);
    Eigen::VectorXd rightHandSide(static_cast<Eigen::Index>(points.size()));
    for (std::size_t i = 0; i < points.size(); ++i) {
        const double squareRootWeight = std::sqrt(weights[i]);
        design.row(static_cast<Eigen::Index>(i)) <<
            2.0 * points[i].x() * squareRootWeight,
            2.0 * points[i].y() * squareRootWeight,
            2.0 * points[i].z() * squareRootWeight,
            squareRootWeight;
        rightHandSide(static_cast<Eigen::Index>(i)) = points[i].squaredNorm() * squareRootWeight;
    }
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(design);
    qr.setThreshold(1.0e-12);
    if (qr.rank() < 4) {
        result.diagnostic = "Sphere initialization is rank deficient (insufficient angular coverage)";
        return result;
    }
    const Eigen::Vector4d algebraic = qr.solve(rightHandSide);
    Eigen::Vector3d center = algebraic.head<3>();
    const double radiusSquared = center.squaredNorm() + algebraic[3];
    if (!(radiusSquared > 0.0) || !std::isfinite(radiusSquared) || !center.allFinite()) {
        result.diagnostic = "Algebraic sphere initialization is invalid";
        return result;
    }
    double radius = std::sqrt(radiusSquared);

    std::vector<double> initialResiduals(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        initialResiduals[i] = (points[i] - center).norm() - radius;
    }
    double robustThreshold = options.huberThreshold > 0.0
        ? options.huberThreshold / scale
        : medianAbsolute(initialResiduals) * 1.4826 * 1.345;
    robustThreshold = std::max(robustThreshold,
                               std::numeric_limits<double>::epsilon() * 128.0);
    double damping = options.initialDamping;
    double cost = huberCost(points, weights, center, radius, robustThreshold);

    for (std::size_t iteration = 0; iteration < options.maximumIterations; ++iteration) {
        Eigen::Matrix4d normalMatrix = Eigen::Matrix4d::Zero();
        Eigen::Vector4d gradient = Eigen::Vector4d::Zero();
        for (std::size_t i = 0; i < points.size(); ++i) {
            const Eigen::Vector3d difference = points[i] - center;
            const double distance = difference.norm();
            if (distance <= std::numeric_limits<double>::min()) continue;
            const double residual = distance - radius;
            const double robustWeight = std::abs(residual) <= robustThreshold
                ? 1.0
                : robustThreshold / std::abs(residual);
            Eigen::Vector4d jacobian;
            jacobian.head<3>() = -difference / distance;
            jacobian[3] = -1.0;
            const double weight = weights[i] * robustWeight;
            normalMatrix.noalias() += weight * jacobian * jacobian.transpose();
            gradient.noalias() += weight * jacobian * residual;
        }

        bool accepted = false;
        Eigen::Vector4d acceptedStep = Eigen::Vector4d::Zero();
        for (std::size_t attempt = 0; attempt < 10U; ++attempt) {
            Eigen::Matrix4d damped = normalMatrix;
            for (Eigen::Index diagonal = 0; diagonal < 4; ++diagonal) {
                damped(diagonal, diagonal) += damping *
                    std::max(normalMatrix(diagonal, diagonal), 1.0e-12);
            }
            const Eigen::Vector4d step = damped.ldlt().solve(-gradient);
            if (!step.allFinite()) {
                damping *= 10.0;
                continue;
            }
            const Eigen::Vector3d candidateCenter = center + step.head<3>();
            const double candidateRadius = radius + step[3];
            if (!(candidateRadius > 0.0)) {
                damping *= 10.0;
                continue;
            }
            const double candidateCost = huberCost(
                points, weights, candidateCenter, candidateRadius, robustThreshold);
            if (candidateCost <= cost) {
                center = candidateCenter;
                radius = candidateRadius;
                cost = candidateCost;
                damping = std::max(damping * 0.25, 1.0e-15);
                acceptedStep = step;
                accepted = true;
                break;
            }
            damping *= 10.0;
        }
        result.iterations = iteration + 1U;
        if (!accepted) break;
        if (acceptedStep.norm() <= options.convergenceTolerance *
            (1.0 + center.norm() + radius)) break;
    }

    // Reject fits whose radial directions do not span three dimensions.
    Eigen::Matrix3d radialMoment = Eigen::Matrix3d::Zero();
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3d direction = (points[i] - center).normalized();
        radialMoment.noalias() += weights[i] * direction * direction.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> coverageSolver(radialMoment);
    if (coverageSolver.info() != Eigen::Success ||
        coverageSolver.eigenvalues().minCoeff() < 1.0e-5) {
        result.diagnostic = "Sphere fit has insufficient angular coverage";
        return result;
    }

    result.center = weightedCenter + vec3(center) * scale;
    result.radius = radius * scale;
    std::vector<double> worldResiduals;
    worldResiduals.reserve(points.size());
    double squaredResidualSum = 0.0;
    double maximumResidual = 0.0;
    double inlierWeight = 0.0;
    const double worldThreshold = robustThreshold * scale;
    for (std::size_t i = 0; i < vertexIndices.size(); ++i) {
        const double residual = norm(mesh.vertices[vertexIndices[i]] - result.center) - result.radius;
        worldResiduals.push_back(residual);
        squaredResidualSum += weights[i] * residual * residual;
        maximumResidual = std::max(maximumResidual, std::abs(residual));
        if (std::abs(residual) <= worldThreshold * 2.5) inlierWeight += weights[i];
    }
    result.rmsResidual = std::sqrt(std::max(0.0, squaredResidualSum));
    result.maxResidual = maximumResidual;
    result.medianResidual = medianAbsolute(worldResiduals);
    result.inlierFraction = std::clamp(inlierWeight, 0.0, 1.0);

    double orientationScore = 0.0;
    double orientationArea = 0.0;
    for (const std::uint32_t triangleIndex : surfaceTriangles) {
        const TriangleFeatures& feature = mesh.features[triangleIndex];
        if (feature.degenerate) continue;
        const Vec3d radial = normalized(feature.centroid - result.center);
        const double alignment = dot(feature.normal, radial);
        if (std::isfinite(alignment)) {
            orientationScore += feature.area * alignment;
            orientationArea += feature.area;
        }
    }
    if (orientationArea > 0.0) {
        const double normalizedScore = orientationScore / orientationArea;
        if (normalizedScore > 0.5) result.radialOrientation = 1;
        else if (normalizedScore < -0.5) result.radialOrientation = -1;
    }
    result.valid = isFinite(result.center) && std::isfinite(result.radius) && result.radius > 0.0 &&
                   std::isfinite(result.rmsResidual) && std::isfinite(result.maxResidual);
    result.diagnostic = result.valid ? "ok" : "Geometric sphere refinement produced invalid values";
    return result;
}

OrientedBoxFitResult inferOrientedBox(
    const Mesh& mesh,
    std::span<const PlaneSegment> planeSegments,
    const OrientedBoxFitOptions& options) {
    OrientedBoxFitResult result;
    if (planeSegments.size() < 6U) {
        result.diagnostic = "Fewer than six planar supports are available";
        return result;
    }
    if (options.parallelToleranceRadians <= 0.0 ||
        options.parallelToleranceRadians >= kPi * 0.5 ||
        options.orthogonalityToleranceRadians <= 0.0 ||
        options.orthogonalityToleranceRadians >= kPi * 0.5 ||
        options.planeDistanceTolerance < 0.0) {
        result.diagnostic = "Invalid oriented-box fit tolerances";
        return result;
    }

    const double scale = std::max(
        mesh.bounds.diagonal(), std::numeric_limits<double>::min());
    double largestPlaneResidual = 0.0;
    for (const PlaneSegment& segment : planeSegments) {
        largestPlaneResidual = std::max(largestPlaneResidual, segment.maxResidual);
    }
    const double distanceTolerance = options.planeDistanceTolerance > 0.0
        ? options.planeDistanceTolerance
        : std::max({scale * 2.0e-6, mesh.weldTolerance * 8.0,
                    largestPlaneResidual * 4.0});
    const double parallelCosine = std::cos(options.parallelToleranceRadians);
    const double maximumOrthogonalDot = std::sin(options.orthogonalityToleranceRadians);

    struct PairCandidate {
        std::uint32_t first{};
        std::uint32_t second{};
        Vec3d axis{};
        double separation{};
        double supportArea{};
        double parallelError{};
    };
    std::vector<PairCandidate> pairs;
    for (std::size_t i = 0; i < planeSegments.size(); ++i) {
        const Vec3d firstNormal = normalized(planeSegments[i].plane.normal);
        if (norm(firstNormal) == 0.0) continue;
        for (std::size_t j = i + 1U; j < planeSegments.size(); ++j) {
            const Vec3d secondNormal = normalized(planeSegments[j].plane.normal);
            const double alignment = dot(firstNormal, secondNormal);
            if (std::abs(alignment) < parallelCosine) continue;
            Vec3d axis = normalized(firstNormal + (alignment >= 0.0 ? secondNormal : -secondNormal));
            axis = canonicalDirection(axis);
            const double denominatorFirst = dot(firstNormal, axis);
            const double denominatorSecond = dot(secondNormal, axis);
            if (std::abs(denominatorFirst) < 0.5 || std::abs(denominatorSecond) < 0.5) continue;
            const double coordinateFirst = -planeSegments[i].plane.offset / denominatorFirst;
            const double coordinateSecond = -planeSegments[j].plane.offset / denominatorSecond;
            const double separation = std::abs(coordinateSecond - coordinateFirst);
            if (separation <= distanceTolerance) continue;
            pairs.push_back({static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(j), axis,
                             separation, planeSegments[i].area + planeSegments[j].area,
                             std::acos(clampUnit(std::abs(alignment)))});
        }
    }
    if (pairs.size() < 3U) {
        result.diagnostic = "Could not find three opposite parallel plane pairs";
        return result;
    }

    const double totalMeshArea = std::max(std::numeric_limits<double>::min(), std::accumulate(
        mesh.features.begin(), mesh.features.end(), 0.0,
        [](double sum, const TriangleFeatures& feature) { return sum + feature.area; }));

    for (std::size_t ia = 0; ia < pairs.size(); ++ia) {
        for (std::size_t ib = ia + 1U; ib < pairs.size(); ++ib) {
            for (std::size_t ic = ib + 1U; ic < pairs.size(); ++ic) {
                std::array<PairCandidate, 3> selected{pairs[ia], pairs[ib], pairs[ic]};
                std::array<std::uint32_t, 6> supportIndices{
                    selected[0].first, selected[0].second,
                    selected[1].first, selected[1].second,
                    selected[2].first, selected[2].second};
                std::sort(supportIndices.begin(), supportIndices.end());
                if (std::adjacent_find(supportIndices.begin(), supportIndices.end()) !=
                    supportIndices.end()) continue;
                const double orthogonality = std::max({
                    std::abs(dot(selected[0].axis, selected[1].axis)),
                    std::abs(dot(selected[0].axis, selected[2].axis)),
                    std::abs(dot(selected[1].axis, selected[2].axis))});
                if (orthogonality > maximumOrthogonalDot) continue;

                std::sort(selected.begin(), selected.end(), [](const PairCandidate& lhs,
                                                                const PairCandidate& rhs) {
                    return axisOrderKey(lhs.axis) < axisOrderKey(rhs.axis);
                });
                Eigen::Matrix3d directions;
                for (Eigen::Index column = 0; column < 3; ++column) {
                    directions.col(column) = eigenVector(selected[static_cast<std::size_t>(column)].axis);
                }
                Eigen::JacobiSVD<Eigen::Matrix3d> svd(
                    directions, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::Matrix3d frame = svd.matrixU() * svd.matrixV().transpose();
                if (frame.determinant() < 0.0) frame.col(2) *= -1.0;

                std::array<Vec3d, 3> axes{
                    vec3(frame.col(0)), vec3(frame.col(1)), vec3(frame.col(2))};
                std::array<double, 3> low{};
                std::array<double, 3> high{};
                bool offsetsValid = true;
                double supportedArea = 0.0;
                for (std::size_t axisIndex = 0; axisIndex < 3U; ++axisIndex) {
                    std::array<double, 2> coordinates{};
                    const std::array<std::uint32_t, 2> segmentIndices{
                        selected[axisIndex].first, selected[axisIndex].second};
                    for (std::size_t side = 0; side < 2U; ++side) {
                        const PlaneSegment& segment = planeSegments[segmentIndices[side]];
                        std::vector<std::pair<double, double>> projections;
                        projections.reserve(segment.triangleIndices.size() * 3U);
                        for (const std::uint32_t triangleIndex : segment.triangleIndices) {
                            if (triangleIndex >= mesh.triangles.size()) continue;
                            const double weight = mesh.features[triangleIndex].area / 3.0;
                            for (const std::uint32_t vertex : mesh.triangles[triangleIndex].vertices) {
                                projections.emplace_back(dot(axes[axisIndex], mesh.vertices[vertex]), weight);
                            }
                        }
                        coordinates[side] = weightedMedian(std::move(projections));
                        supportedArea += segment.area;
                    }
                    low[axisIndex] = std::min(coordinates[0], coordinates[1]);
                    high[axisIndex] = std::max(coordinates[0], coordinates[1]);
                    if (!std::isfinite(low[axisIndex]) || !std::isfinite(high[axisIndex]) ||
                        high[axisIndex] - low[axisIndex] <= distanceTolerance) {
                        offsetsValid = false;
                        break;
                    }
                }
                if (!offsetsValid) continue;

                double residualWeight = 0.0;
                double residualSquares = 0.0;
                double maximumResidual = 0.0;
                for (std::size_t axisIndex = 0; axisIndex < 3U; ++axisIndex) {
                    const std::array<std::uint32_t, 2> segmentIndices{
                        selected[axisIndex].first, selected[axisIndex].second};
                    for (const std::uint32_t segmentIndex : segmentIndices) {
                        const PlaneSegment& segment = planeSegments[segmentIndex];
                        const double planeCoordinate = std::abs(
                            -segment.plane.offset / dot(segment.plane.normal, axes[axisIndex]) - low[axisIndex]) <
                            std::abs(-segment.plane.offset / dot(segment.plane.normal, axes[axisIndex]) - high[axisIndex])
                            ? low[axisIndex] : high[axisIndex];
                        for (const std::uint32_t triangleIndex : segment.triangleIndices) {
                            const double weight = mesh.features[triangleIndex].area / 3.0;
                            for (const std::uint32_t vertex : mesh.triangles[triangleIndex].vertices) {
                                const double residual = std::abs(
                                    dot(axes[axisIndex], mesh.vertices[vertex]) - planeCoordinate);
                                residualSquares += weight * residual * residual;
                                residualWeight += weight;
                                maximumResidual = std::max(maximumResidual, residual);
                            }
                        }
                    }
                }
                const double rmsResidual = residualWeight > 0.0
                    ? std::sqrt(residualSquares / residualWeight)
                    : std::numeric_limits<double>::infinity();
                double containmentViolation = 0.0;
                for (const Vec3d& vertex : mesh.vertices) {
                    for (std::size_t axisIndex = 0; axisIndex < 3U; ++axisIndex) {
                        const double projection = dot(axes[axisIndex], vertex);
                        containmentViolation = std::max({containmentViolation,
                            low[axisIndex] - projection, projection - high[axisIndex]});
                    }
                }
                containmentViolation = std::max(0.0, containmentViolation);
                const double parallelError = selected[0].parallelError +
                                             selected[1].parallelError +
                                             selected[2].parallelError;
                const double score = supportedArea / totalMeshArea -
                    10.0 * orthogonality - 2.0 * parallelError -
                    10.0 * rmsResidual / scale - 20.0 * containmentViolation / scale;
                if (score <= result.score) continue;

                result.valid = true;
                result.axes = axes;
                result.lowCoordinates = {low[0], low[1], low[2]};
                result.highCoordinates = {high[0], high[1], high[2]};
                result.dimensions = {high[0] - low[0], high[1] - low[1], high[2] - low[2]};
                result.center = axes[0] * ((low[0] + high[0]) * 0.5) +
                                axes[1] * ((low[1] + high[1]) * 0.5) +
                                axes[2] * ((low[2] + high[2]) * 0.5);
                for (std::size_t axisIndex = 0; axisIndex < 3U; ++axisIndex) {
                    result.supportingPlanePairs[axisIndex] = {
                        selected[axisIndex].first, selected[axisIndex].second};
                }
                result.rmsResidual = rmsResidual;
                result.maxResidual = maximumResidual;
                result.orthogonalityError = orthogonality;
                result.supportedArea = supportedArea;
                result.score = score;
                result.diagnostic = "ok";
            }
        }
    }
    if (!result.valid) result.diagnostic = "No mutually orthogonal set of three opposite plane pairs was found";
    return result;
}

} // namespace recad
