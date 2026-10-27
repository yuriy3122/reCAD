#include "CudaGeometry.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace recad {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr double kPi = 3.141592653589793238462643383279502884;

[[noreturn]] void throwCudaError(cudaError_t error,
                                 const char* expression,
                                 const char* file,
                                 int line) {
    std::ostringstream message;
    message << "CUDA call failed: " << expression << " at " << file << ':' << line
            << ": " << cudaGetErrorName(error) << " (" << cudaGetErrorString(error) << ')';
    throw std::runtime_error(message.str());
}

#define RECAD_CUDA_CHECK(expression)                                                        \
    do {                                                                                    \
        const cudaError_t recadCudaStatus = (expression);                                   \
        if (recadCudaStatus != cudaSuccess) {                                               \
            throwCudaError(recadCudaStatus, #expression, __FILE__, __LINE__);               \
        }                                                                                   \
    } while (false)

struct DeviceVec3 {
    double x;
    double y;
    double z;
};

struct DeviceTriangle {
    std::uint32_t v0;
    std::uint32_t v1;
    std::uint32_t v2;
};

struct DeviceTriangleFeature {
    DeviceVec3 normal;
    DeviceVec3 centroid;
    double area;
    double planeOffset;
    double maxEdgeLength;
    int degenerate;
};

struct DeviceAdjacencyPair {
    std::uint32_t first;
    std::uint32_t second;
    int sourceCompatible{1};
};

struct DeviceResidualStatistics {
    unsigned long long sampleCount;
    unsigned long long inlierCount;
    double sumSquaredResidual;
    double maxResidual;
    double weightSum;
    double weightedSumSquaredResidual;
    double inlierWeight;
};

class CudaStream final {
public:
    CudaStream() {
        RECAD_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }

    ~CudaStream() {
        if (stream_ != nullptr) {
            (void)cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_{};
};

class CudaEvent final {
public:
    CudaEvent() { RECAD_CUDA_CHECK(cudaEventCreate(&event_)); }

    ~CudaEvent() {
        if (event_ != nullptr) {
            (void)cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    void record(cudaStream_t stream) { RECAD_CUDA_CHECK(cudaEventRecord(event_, stream)); }
    void synchronize() { RECAD_CUDA_CHECK(cudaEventSynchronize(event_)); }

    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_{};
};

template <typename T>
class DeviceBuffer final {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) { allocate(count); }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0U)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                (void)cudaFree(data_);
            }
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0U);
        }
        return *this;
    }

    void allocate(std::size_t count) {
        if (count == 0U) {
            return;
        }
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::length_error("CUDA allocation size overflow");
        }
        RECAD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
        count_ = count;
    }

    [[nodiscard]] T* get() noexcept { return data_; }
    [[nodiscard]] const T* get() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
    T* data_{};
    std::size_t count_{};
};

[[nodiscard]] float elapsedMilliseconds(const CudaEvent& begin, const CudaEvent& end) {
    float value{};
    RECAD_CUDA_CHECK(cudaEventElapsedTime(&value, begin.get(), end.get()));
    return value;
}

[[nodiscard]] CudaDeviceInfo makeDeviceInfo(int ordinal, const cudaDeviceProp& properties) {
    CudaDeviceInfo result;
    result.ordinal = ordinal;
    result.name = properties.name;
    result.computeCapabilityMajor = properties.major;
    result.computeCapabilityMinor = properties.minor;
    result.totalGlobalMemoryBytes = static_cast<std::uint64_t>(properties.totalGlobalMem);
    result.multiprocessorCount = properties.multiProcessorCount;
    result.maxThreadsPerBlock = properties.maxThreadsPerBlock;
    RECAD_CUDA_CHECK(cudaRuntimeGetVersion(&result.runtimeVersion));
    RECAD_CUDA_CHECK(cudaDriverGetVersion(&result.driverVersion));
    return result;
}

struct SelectedDevice {
    CudaDeviceInfo info;
    cudaDeviceProp properties{};
};

[[nodiscard]] SelectedDevice selectDevice(int requestedOrdinal) {
    int count{};
    const cudaError_t countStatus = cudaGetDeviceCount(&count);
    if (countStatus == cudaErrorNoDevice) {
        (void)cudaGetLastError();
        throw std::runtime_error("No CUDA compute device is available");
    }
    if (countStatus != cudaSuccess) {
        throwCudaError(countStatus, "cudaGetDeviceCount(&count)", __FILE__, __LINE__);
    }
    if (count <= 0) {
        throw std::runtime_error("No CUDA compute device is available");
    }

    int ordinal = requestedOrdinal;
    if (ordinal < 0) {
        RECAD_CUDA_CHECK(cudaGetDevice(&ordinal));
    }
    if (ordinal < 0 || ordinal >= count) {
        std::ostringstream message;
        message << "CUDA device ordinal " << ordinal << " is outside [0, " << count << ')';
        throw std::out_of_range(message.str());
    }

    RECAD_CUDA_CHECK(cudaSetDevice(ordinal));
    SelectedDevice selected;
    RECAD_CUDA_CHECK(cudaGetDeviceProperties(&selected.properties, ordinal));
    selected.info = makeDeviceInfo(ordinal, selected.properties);
    return selected;
}

[[nodiscard]] bool finite(const Vec3d& value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

void validatePoints(const std::vector<Vec3d>& points) {
    for (std::size_t index = 0; index < points.size(); ++index) {
        if (!finite(points[index])) {
            std::ostringstream message;
            message << "Point " << index << " contains a non-finite coordinate";
            throw std::invalid_argument(message.str());
        }
    }
}

void validateWeights(const std::vector<double>& weights, std::size_t pointCount) {
    if (!weights.empty() && weights.size() != pointCount) {
        throw std::invalid_argument("Residual weight count must match point count");
    }
    for (std::size_t index = 0; index < weights.size(); ++index) {
        if (!std::isfinite(weights[index]) || weights[index] < 0.0) {
            std::ostringstream message;
            message << "Residual weight " << index << " must be finite and non-negative";
            throw std::invalid_argument(message.str());
        }
    }
}

[[nodiscard]] std::vector<DeviceVec3> packPoints(const std::vector<Vec3d>& points) {
    std::vector<DeviceVec3> packed;
    packed.reserve(points.size());
    for (const Vec3d& point : points) {
        packed.push_back({point.x, point.y, point.z});
    }
    return packed;
}

[[nodiscard]] int launchBlockCount(std::size_t itemCount, const cudaDeviceProp& properties) {
    const std::size_t requested =
        (itemCount + static_cast<std::size_t>(kThreadsPerBlock) - 1U) /
        static_cast<std::size_t>(kThreadsPerBlock);
    const std::size_t deviceLimit = static_cast<std::size_t>(properties.maxGridSize[0]);
    return static_cast<int>(std::max<std::size_t>(1U, std::min(requested, deviceLimit)));
}

__device__ __forceinline__ DeviceVec3 subtract(DeviceVec3 left, DeviceVec3 right) {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

__device__ __forceinline__ DeviceVec3 crossProduct(DeviceVec3 left, DeviceVec3 right) {
    return {left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x};
}

__device__ __forceinline__ double dotProduct(DeviceVec3 left, DeviceVec3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

__device__ __forceinline__ double vectorLength(DeviceVec3 value) {
    return sqrt(dotProduct(value, value));
}

__global__ void triangleFeatureKernel(const DeviceVec3* vertices,
                                      const DeviceTriangle* triangles,
                                      std::size_t triangleCount,
                                      double areaTolerance,
                                      DeviceTriangleFeature* output) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < triangleCount;
         index += stride) {
        const DeviceTriangle triangle = triangles[index];
        const DeviceVec3 a = vertices[triangle.v0];
        const DeviceVec3 b = vertices[triangle.v1];
        const DeviceVec3 c = vertices[triangle.v2];

        const DeviceVec3 edgeAB = subtract(b, a);
        const DeviceVec3 edgeAC = subtract(c, a);
        const DeviceVec3 edgeBC = subtract(c, b);
        const DeviceVec3 twiceAreaVector = crossProduct(edgeAB, edgeAC);
        const double twiceArea = vectorLength(twiceAreaVector);
        const double area = 0.5 * twiceArea;
        const bool degenerate = !isfinite(area) || area <= areaTolerance;

        DeviceTriangleFeature feature{};
        feature.centroid = {(a.x + b.x + c.x) / 3.0,
                            (a.y + b.y + c.y) / 3.0,
                            (a.z + b.z + c.z) / 3.0};
        feature.area = area;
        feature.maxEdgeLength = fmax(vectorLength(edgeAB),
                                     fmax(vectorLength(edgeAC), vectorLength(edgeBC)));
        feature.degenerate = degenerate ? 1 : 0;
        if (!degenerate) {
            const double inverseLength = 1.0 / twiceArea;
            feature.normal = {twiceAreaVector.x * inverseLength,
                              twiceAreaVector.y * inverseLength,
                              twiceAreaVector.z * inverseLength};
            feature.planeOffset = -dotProduct(feature.normal, feature.centroid);
        }
        output[index] = feature;
    }
}

__global__ void triangleAdjacencyMergeKernel(
    const DeviceTriangleFeature* features,
    const DeviceAdjacencyPair* adjacencyPairs,
    std::size_t pairCount,
    double minimumNormalDot,
    double maximumCentroidDistance,
    double maximumCentroidDistanceToEdgeRatio,
    bool treatOppositeNormalsAsAligned,
    std::uint8_t* mergeMask) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < pairCount;
         index += stride) {
        const DeviceAdjacencyPair pair = adjacencyPairs[index];
        const DeviceTriangleFeature first = features[pair.first];
        const DeviceTriangleFeature second = features[pair.second];
        bool merge = pair.sourceCompatible != 0 &&
                     first.degenerate == 0 && second.degenerate == 0;

        if (merge) {
            const double firstNormalSquared = dotProduct(first.normal, first.normal);
            const double secondNormalSquared = dotProduct(second.normal, second.normal);
            if (!(firstNormalSquared > 0.0) || !(secondNormalSquared > 0.0)) {
                merge = false;
            } else {
                double normalDot = dotProduct(first.normal, second.normal) /
                    sqrt(firstNormalSquared * secondNormalSquared);
                normalDot = fmax(-1.0, fmin(1.0, normalDot));
                if (treatOppositeNormalsAsAligned) normalDot = fabs(normalDot);
                merge = normalDot >= minimumNormalDot;
            }
        }

        if (merge && (maximumCentroidDistance > 0.0 ||
                      maximumCentroidDistanceToEdgeRatio > 0.0)) {
            const double centroidDistance = vectorLength(
                subtract(first.centroid, second.centroid));
            if (!isfinite(centroidDistance)) {
                merge = false;
            }
            if (merge && maximumCentroidDistance > 0.0 &&
                centroidDistance > maximumCentroidDistance) {
                merge = false;
            }
            if (merge && maximumCentroidDistanceToEdgeRatio > 0.0) {
                const double localScale = fmax(first.maxEdgeLength, second.maxEdgeLength);
                merge = localScale > 0.0 &&
                    centroidDistance <= maximumCentroidDistanceToEdgeRatio * localScale;
            }
        }

        mergeMask[index] = merge ? std::uint8_t{1} : std::uint8_t{0};
    }
}

__device__ __forceinline__ double atomicAddDouble(double* address, double value) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    auto* integerAddress = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *integerAddress;
    unsigned long long assumed{};
    do {
        assumed = old;
        old = atomicCAS(integerAddress,
                        assumed,
                        __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

// Residuals are non-negative, which makes the integer CAS comparison safe and
// avoids unavailable architecture-specific floating-point max intrinsics.
__device__ __forceinline__ double atomicMaxNonNegativeDouble(double* address, double value) {
    auto* integerAddress = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *integerAddress;
    while (value > __longlong_as_double(old)) {
        const unsigned long long assumed = old;
        old = atomicCAS(integerAddress, assumed, __double_as_longlong(value));
        if (old == assumed) {
            break;
        }
    }
    return __longlong_as_double(old);
}

struct PlaneResidual {
    DeviceVec3 normal;
    double offset;

    __device__ __forceinline__ double operator()(DeviceVec3 point) const {
        return fabs(dotProduct(normal, point) + offset);
    }
};

struct SphereResidual {
    DeviceVec3 center;
    double radius;

    __device__ __forceinline__ double operator()(DeviceVec3 point) const {
        return fabs(vectorLength(subtract(point, center)) - radius);
    }
};

template <typename ResidualFunction>
__global__ void residualKernel(const DeviceVec3* points,
                               const double* weights,
                               std::size_t pointCount,
                               double inlierTolerance,
                               ResidualFunction residualFunction,
                               double* residuals,
                               std::uint8_t* inlierMask,
                               DeviceResidualStatistics* statistics) {
    __shared__ double blockSumSquared[kThreadsPerBlock];
    __shared__ double blockMaximum[kThreadsPerBlock];
    __shared__ double blockWeight[kThreadsPerBlock];
    __shared__ double blockWeightedSumSquared[kThreadsPerBlock];
    __shared__ double blockInlierWeight[kThreadsPerBlock];
    __shared__ unsigned long long blockSamples[kThreadsPerBlock];
    __shared__ unsigned long long blockInliers[kThreadsPerBlock];

    double localSumSquared = 0.0;
    double localMaximum = 0.0;
    double localWeight = 0.0;
    double localWeightedSumSquared = 0.0;
    double localInlierWeight = 0.0;
    unsigned long long localSamples = 0ULL;
    unsigned long long localInliers = 0ULL;

    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < pointCount;
         index += stride) {
        const double residual = residualFunction(points[index]);
        const bool inlier = residual <= inlierTolerance;
        const double weight = weights == nullptr ? 1.0 : weights[index];
        const double squared = residual * residual;

        residuals[index] = residual;
        inlierMask[index] = inlier ? std::uint8_t{1} : std::uint8_t{0};
        localSumSquared += squared;
        localMaximum = fmax(localMaximum, residual);
        localWeight += weight;
        localWeightedSumSquared += weight * squared;
        localInlierWeight += inlier ? weight : 0.0;
        ++localSamples;
        localInliers += inlier ? 1ULL : 0ULL;
    }

    const unsigned int lane = threadIdx.x;
    blockSumSquared[lane] = localSumSquared;
    blockMaximum[lane] = localMaximum;
    blockWeight[lane] = localWeight;
    blockWeightedSumSquared[lane] = localWeightedSumSquared;
    blockInlierWeight[lane] = localInlierWeight;
    blockSamples[lane] = localSamples;
    blockInliers[lane] = localInliers;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2U; offset != 0U; offset /= 2U) {
        if (lane < offset) {
            blockSumSquared[lane] += blockSumSquared[lane + offset];
            blockMaximum[lane] = fmax(blockMaximum[lane], blockMaximum[lane + offset]);
            blockWeight[lane] += blockWeight[lane + offset];
            blockWeightedSumSquared[lane] += blockWeightedSumSquared[lane + offset];
            blockInlierWeight[lane] += blockInlierWeight[lane + offset];
            blockSamples[lane] += blockSamples[lane + offset];
            blockInliers[lane] += blockInliers[lane + offset];
        }
        __syncthreads();
    }

    if (lane == 0U) {
        atomicAdd(&statistics->sampleCount, blockSamples[0]);
        atomicAdd(&statistics->inlierCount, blockInliers[0]);
        atomicAddDouble(&statistics->sumSquaredResidual, blockSumSquared[0]);
        atomicMaxNonNegativeDouble(&statistics->maxResidual, blockMaximum[0]);
        atomicAddDouble(&statistics->weightSum, blockWeight[0]);
        atomicAddDouble(&statistics->weightedSumSquaredResidual,
                        blockWeightedSumSquared[0]);
        atomicAddDouble(&statistics->inlierWeight, blockInlierWeight[0]);
    }
}

template <typename ResidualFunction>
[[nodiscard]] CudaResidualResult evaluateResiduals(
    const std::vector<Vec3d>& points,
    double inlierTolerance,
    const std::vector<double>& weights,
    int deviceOrdinal,
    ResidualFunction residualFunction) {
    validatePoints(points);
    validateWeights(weights, points.size());
    if (!std::isfinite(inlierTolerance) || inlierTolerance < 0.0) {
        throw std::invalid_argument("Residual inlier tolerance must be finite and non-negative");
    }

    const SelectedDevice selected = selectDevice(deviceOrdinal);
    CudaResidualResult result;
    result.device = selected.info;
    result.residuals.resize(points.size());
    result.inlierMask.resize(points.size());
    if (points.empty()) {
        return result;
    }

    const std::vector<DeviceVec3> packedPoints = packPoints(points);
    DeviceBuffer<DeviceVec3> devicePoints(points.size());
    DeviceBuffer<double> deviceWeights(weights.size());
    DeviceBuffer<double> deviceResiduals(points.size());
    DeviceBuffer<std::uint8_t> deviceInlierMask(points.size());
    DeviceBuffer<DeviceResidualStatistics> deviceStatistics(1U);
    DeviceResidualStatistics hostStatistics{};

    CudaStream stream;
    CudaEvent started;
    CudaEvent uploaded;
    CudaEvent kernelFinished;
    CudaEvent downloaded;

    started.record(stream.get());
    RECAD_CUDA_CHECK(cudaMemcpyAsync(devicePoints.get(),
                                     packedPoints.data(),
                                     packedPoints.size() * sizeof(DeviceVec3),
                                     cudaMemcpyHostToDevice,
                                     stream.get()));
    if (!weights.empty()) {
        RECAD_CUDA_CHECK(cudaMemcpyAsync(deviceWeights.get(),
                                         weights.data(),
                                         weights.size() * sizeof(double),
                                         cudaMemcpyHostToDevice,
                                         stream.get()));
    }
    RECAD_CUDA_CHECK(cudaMemsetAsync(deviceStatistics.get(),
                                     0,
                                     sizeof(DeviceResidualStatistics),
                                     stream.get()));
    uploaded.record(stream.get());

    const int blocks = launchBlockCount(points.size(), selected.properties);
    residualKernel<<<blocks, kThreadsPerBlock, 0, stream.get()>>>(
        devicePoints.get(),
        weights.empty() ? nullptr : deviceWeights.get(),
        points.size(),
        inlierTolerance,
        residualFunction,
        deviceResiduals.get(),
        deviceInlierMask.get(),
        deviceStatistics.get());
    RECAD_CUDA_CHECK(cudaGetLastError());
    kernelFinished.record(stream.get());

    RECAD_CUDA_CHECK(cudaMemcpyAsync(result.residuals.data(),
                                     deviceResiduals.get(),
                                     result.residuals.size() * sizeof(double),
                                     cudaMemcpyDeviceToHost,
                                     stream.get()));
    RECAD_CUDA_CHECK(cudaMemcpyAsync(result.inlierMask.data(),
                                     deviceInlierMask.get(),
                                     result.inlierMask.size() * sizeof(std::uint8_t),
                                     cudaMemcpyDeviceToHost,
                                     stream.get()));
    RECAD_CUDA_CHECK(cudaMemcpyAsync(&hostStatistics,
                                     deviceStatistics.get(),
                                     sizeof(DeviceResidualStatistics),
                                     cudaMemcpyDeviceToHost,
                                     stream.get()));
    downloaded.record(stream.get());
    downloaded.synchronize();

    result.timings.hostToDeviceMs = elapsedMilliseconds(started, uploaded);
    result.timings.kernelMs = elapsedMilliseconds(uploaded, kernelFinished);
    result.timings.deviceToHostMs = elapsedMilliseconds(kernelFinished, downloaded);
    result.timings.totalGpuMs = elapsedMilliseconds(started, downloaded);

    if (hostStatistics.sampleCount != static_cast<unsigned long long>(points.size())) {
        throw std::runtime_error("CUDA residual reduction returned an inconsistent sample count");
    }

    CudaResidualStatistics& statistics = result.statistics;
    statistics.sampleCount = static_cast<std::uint64_t>(hostStatistics.sampleCount);
    statistics.inlierCount = static_cast<std::uint64_t>(hostStatistics.inlierCount);
    statistics.sumSquaredResidual = hostStatistics.sumSquaredResidual;
    statistics.rmsResidual = std::sqrt(hostStatistics.sumSquaredResidual /
                                       static_cast<double>(hostStatistics.sampleCount));
    statistics.maxResidual = hostStatistics.maxResidual;
    statistics.inlierFraction = static_cast<double>(hostStatistics.inlierCount) /
                                static_cast<double>(hostStatistics.sampleCount);
    statistics.weightSum = hostStatistics.weightSum;
    statistics.weightedSumSquaredResidual = hostStatistics.weightedSumSquaredResidual;
    statistics.weightedRmsResidual = hostStatistics.weightSum > 0.0
                                         ? std::sqrt(hostStatistics.weightedSumSquaredResidual /
                                                     hostStatistics.weightSum)
                                         : 0.0;
    statistics.inlierWeight = hostStatistics.inlierWeight;
    statistics.inlierWeightFraction = hostStatistics.weightSum > 0.0
                                          ? hostStatistics.inlierWeight /
                                                hostStatistics.weightSum
                                          : 0.0;
    return result;
}

} // namespace

std::vector<CudaDeviceInfo> enumerateCudaDevices() {
    int count{};
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice) {
        (void)cudaGetLastError();
        return {};
    }
    if (status != cudaSuccess) {
        throwCudaError(status, "cudaGetDeviceCount(&count)", __FILE__, __LINE__);
    }

    std::vector<CudaDeviceInfo> devices;
    devices.reserve(static_cast<std::size_t>(count));
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        cudaDeviceProp properties{};
        RECAD_CUDA_CHECK(cudaGetDeviceProperties(&properties, ordinal));
        devices.push_back(makeDeviceInfo(ordinal, properties));
    }
    return devices;
}

CudaTriangleFeatureResult computeTriangleFeaturesCuda(
    const std::vector<Vec3d>& vertices,
    const std::vector<Triangle>& triangles,
    double degenerateAreaTolerance,
    int deviceOrdinal) {
    validatePoints(vertices);
    if (!std::isfinite(degenerateAreaTolerance) || degenerateAreaTolerance < 0.0) {
        throw std::invalid_argument(
            "Triangle degeneracy area tolerance must be finite and non-negative");
    }

    std::vector<DeviceTriangle> packedTriangles;
    packedTriangles.reserve(triangles.size());
    for (std::size_t index = 0; index < triangles.size(); ++index) {
        const Triangle& triangle = triangles[index];
        if (triangle.vertices[0] >= vertices.size() ||
            triangle.vertices[1] >= vertices.size() ||
            triangle.vertices[2] >= vertices.size()) {
            std::ostringstream message;
            message << "Triangle " << index << " contains an out-of-range vertex index";
            throw std::out_of_range(message.str());
        }
        packedTriangles.push_back(
            {triangle.vertices[0], triangle.vertices[1], triangle.vertices[2]});
    }

    const SelectedDevice selected = selectDevice(deviceOrdinal);
    CudaTriangleFeatureResult result;
    result.device = selected.info;
    result.features.resize(triangles.size());
    if (triangles.empty()) {
        return result;
    }

    const std::vector<DeviceVec3> packedVertices = packPoints(vertices);
    std::vector<DeviceTriangleFeature> hostFeatures(triangles.size());
    DeviceBuffer<DeviceVec3> deviceVertices(vertices.size());
    DeviceBuffer<DeviceTriangle> deviceTriangles(triangles.size());
    DeviceBuffer<DeviceTriangleFeature> deviceFeatures(triangles.size());

    CudaStream stream;
    CudaEvent started;
    CudaEvent uploaded;
    CudaEvent kernelFinished;
    CudaEvent downloaded;

    started.record(stream.get());
    RECAD_CUDA_CHECK(cudaMemcpyAsync(deviceVertices.get(),
                                     packedVertices.data(),
                                     packedVertices.size() * sizeof(DeviceVec3),
                                     cudaMemcpyHostToDevice,
                                     stream.get()));
    RECAD_CUDA_CHECK(cudaMemcpyAsync(deviceTriangles.get(),
                                     packedTriangles.data(),
                                     packedTriangles.size() * sizeof(DeviceTriangle),
                                     cudaMemcpyHostToDevice,
                                     stream.get()));
    uploaded.record(stream.get());

    const int blocks = launchBlockCount(triangles.size(), selected.properties);
    triangleFeatureKernel<<<blocks, kThreadsPerBlock, 0, stream.get()>>>(
        deviceVertices.get(),
        deviceTriangles.get(),
        triangles.size(),
        degenerateAreaTolerance,
        deviceFeatures.get());
    RECAD_CUDA_CHECK(cudaGetLastError());
    kernelFinished.record(stream.get());

    RECAD_CUDA_CHECK(cudaMemcpyAsync(hostFeatures.data(),
                                     deviceFeatures.get(),
                                     hostFeatures.size() * sizeof(DeviceTriangleFeature),
                                     cudaMemcpyDeviceToHost,
                                     stream.get()));
    downloaded.record(stream.get());
    downloaded.synchronize();

    result.timings.hostToDeviceMs = elapsedMilliseconds(started, uploaded);
    result.timings.kernelMs = elapsedMilliseconds(uploaded, kernelFinished);
    result.timings.deviceToHostMs = elapsedMilliseconds(kernelFinished, downloaded);
    result.timings.totalGpuMs = elapsedMilliseconds(started, downloaded);

    for (std::size_t index = 0; index < hostFeatures.size(); ++index) {
        const DeviceTriangleFeature& source = hostFeatures[index];
        TriangleFeatures& destination = result.features[index];
        destination.normal = {source.normal.x, source.normal.y, source.normal.z};
        destination.centroid = {source.centroid.x, source.centroid.y, source.centroid.z};
        destination.area = source.area;
        destination.planeOffset = source.planeOffset;
        destination.maxEdgeLength = source.maxEdgeLength;
        destination.degenerate = source.degenerate != 0;
        result.degenerateCount += destination.degenerate ? 1U : 0U;
    }
    return result;
}

CudaTriangleSegmentationResult segmentTriangleRegionsCuda(
    const Mesh& mesh,
    const CudaTriangleSegmentationOptions& options,
    int deviceOrdinal) {
    if (!std::isfinite(options.maximumDihedralAngleRadians) ||
        options.maximumDihedralAngleRadians < 0.0 ||
        options.maximumDihedralAngleRadians > kPi) {
        throw std::invalid_argument(
            "Maximum dihedral angle must be finite and in the range [0, pi]");
    }
    if (options.treatOppositeNormalsAsAligned &&
        options.maximumDihedralAngleRadians > 0.5 * kPi) {
        throw std::invalid_argument(
            "Opposite-normal alignment requires a maximum dihedral angle no greater than pi/2");
    }
    if (!std::isfinite(options.maximumCentroidDistance) ||
        options.maximumCentroidDistance < 0.0) {
        throw std::invalid_argument(
            "Maximum centroid distance must be finite and non-negative");
    }
    if (!std::isfinite(options.maximumCentroidDistanceToEdgeRatio) ||
        options.maximumCentroidDistanceToEdgeRatio < 0.0) {
        throw std::invalid_argument(
            "Maximum centroid-distance-to-edge ratio must be finite and non-negative");
    }

    const std::size_t triangleCount = mesh.triangles.size();
    if (triangleCount > static_cast<std::size_t>(
                            std::numeric_limits<std::int32_t>::max())) {
        throw std::length_error("Triangle count exceeds the supported label range");
    }
    if (mesh.features.size() != triangleCount) {
        throw std::invalid_argument(
            "CUDA triangle segmentation requires one feature record per triangle");
    }
    if (mesh.triangleNeighbors.size() != triangleCount) {
        throw std::invalid_argument(
            "CUDA triangle segmentation requires one adjacency list per triangle");
    }
    const bool hasSourcePatches = options.respectSourcePatchBoundaries &&
                                  !mesh.sourcePatchLabels.empty();
    if (hasSourcePatches && mesh.sourcePatchLabels.size() != triangleCount) {
        throw std::invalid_argument(
            "Source patch labels must be empty or contain one label per triangle");
    }
    if (hasSourcePatches && mesh.sourcePatchCount == 0U) {
        throw std::invalid_argument(
            "A mesh with source patch labels must declare a non-zero patch count");
    }
    if (hasSourcePatches) {
        for (const auto label : mesh.sourcePatchLabels) {
            if (label >= mesh.sourcePatchCount) {
                throw std::invalid_argument("A source patch label is outside its declared range");
            }
        }
    }

    validatePoints(mesh.vertices);
    for (std::size_t triangleIndex = 0; triangleIndex < triangleCount; ++triangleIndex) {
        const Triangle& triangle = mesh.triangles[triangleIndex];
        for (const std::uint32_t vertex : triangle.vertices) {
            if (vertex >= mesh.vertices.size()) {
                std::ostringstream message;
                message << "Triangle " << triangleIndex
                        << " contains an out-of-range vertex index";
                throw std::out_of_range(message.str());
            }
        }

        const TriangleFeatures& feature = mesh.features[triangleIndex];
        if (!finite(feature.centroid) || !std::isfinite(feature.area) ||
            feature.area < 0.0 || !std::isfinite(feature.maxEdgeLength) ||
            feature.maxEdgeLength < 0.0) {
            std::ostringstream message;
            message << "Triangle feature " << triangleIndex
                    << " contains an invalid centroid, area, or edge length";
            throw std::invalid_argument(message.str());
        }
        if (!feature.degenerate) {
            const double normalSquared = feature.normal.x * feature.normal.x +
                feature.normal.y * feature.normal.y + feature.normal.z * feature.normal.z;
            if (!finite(feature.normal) || !std::isfinite(normalSquared) ||
                !(normalSquared > 0.0) || !(feature.area > 0.0) ||
                !(feature.maxEdgeLength > 0.0)) {
                std::ostringstream message;
                message << "Non-degenerate triangle feature " << triangleIndex
                        << " has invalid geometry";
                throw std::invalid_argument(message.str());
            }
        }
    }

    std::vector<DeviceAdjacencyPair> adjacencyPairs;
    for (std::size_t triangleIndex = 0; triangleIndex < triangleCount; ++triangleIndex) {
        for (const std::uint32_t neighbor : mesh.triangleNeighbors[triangleIndex]) {
            if (neighbor >= triangleCount) {
                std::ostringstream message;
                message << "Triangle " << triangleIndex
                        << " has an out-of-range adjacent triangle index";
                throw std::out_of_range(message.str());
            }
            if (neighbor == triangleIndex) {
                std::ostringstream message;
                message << "Triangle " << triangleIndex << " is adjacent to itself";
                throw std::invalid_argument(message.str());
            }
            const auto& reverseNeighbors = mesh.triangleNeighbors[neighbor];
            if (std::find(reverseNeighbors.begin(), reverseNeighbors.end(),
                          static_cast<std::uint32_t>(triangleIndex)) == reverseNeighbors.end()) {
                std::ostringstream message;
                message << "Triangle adjacency is not symmetric for pair "
                        << triangleIndex << " and " << neighbor;
                throw std::invalid_argument(message.str());
            }
            const std::uint32_t first = static_cast<std::uint32_t>(
                std::min<std::size_t>(triangleIndex, neighbor));
            const std::uint32_t second = static_cast<std::uint32_t>(
                std::max<std::size_t>(triangleIndex, neighbor));
            adjacencyPairs.push_back({first, second});
        }
    }
    std::sort(adjacencyPairs.begin(), adjacencyPairs.end(),
              [](const DeviceAdjacencyPair& left, const DeviceAdjacencyPair& right) {
                  return left.first < right.first ||
                      (left.first == right.first && left.second < right.second);
              });
    adjacencyPairs.erase(
        std::unique(adjacencyPairs.begin(), adjacencyPairs.end(),
                    [](const DeviceAdjacencyPair& left, const DeviceAdjacencyPair& right) {
                        return left.first == right.first && left.second == right.second;
                    }),
        adjacencyPairs.end());

    for (const DeviceAdjacencyPair pair : adjacencyPairs) {
        std::size_t sharedVertexCount = 0U;
        const Triangle& first = mesh.triangles[pair.first];
        const Triangle& second = mesh.triangles[pair.second];
        for (std::size_t i = 0; i < first.vertices.size(); ++i) {
            bool alreadyCounted = false;
            for (std::size_t previous = 0; previous < i; ++previous) {
                alreadyCounted = alreadyCounted ||
                    first.vertices[previous] == first.vertices[i];
            }
            if (alreadyCounted) continue;
            if (std::find(second.vertices.begin(), second.vertices.end(),
                          first.vertices[i]) != second.vertices.end()) {
                ++sharedVertexCount;
            }
        }
        if (sharedVertexCount < 2U) {
            std::ostringstream message;
            message << "Adjacent triangles " << pair.first << " and " << pair.second
                    << " do not share a mesh edge";
            throw std::invalid_argument(message.str());
        }
    }

    const SelectedDevice selected = selectDevice(deviceOrdinal);
    CudaTriangleSegmentationResult result;
    result.device = selected.info;
    result.adjacencyPairCount = adjacencyPairs.size();
    result.triangleLabels.assign(triangleCount, -1);

    if (hasSourcePatches) {
        for (auto& pair : adjacencyPairs) {
            pair.sourceCompatible = mesh.sourcePatchLabels[pair.first] ==
                                    mesh.sourcePatchLabels[pair.second] ? 1 : 0;
            if (pair.sourceCompatible == 0) ++result.sourceBoundaryPairCount;
        }
    }

    std::vector<std::uint8_t> mergeMask(adjacencyPairs.size());
    if (!adjacencyPairs.empty()) {
        std::vector<DeviceTriangleFeature> packedFeatures;
        packedFeatures.reserve(triangleCount);
        for (const TriangleFeatures& feature : mesh.features) {
            packedFeatures.push_back({{feature.normal.x, feature.normal.y, feature.normal.z},
                                      {feature.centroid.x, feature.centroid.y, feature.centroid.z},
                                      feature.area,
                                      feature.planeOffset,
                                      feature.maxEdgeLength,
                                      feature.degenerate ? 1 : 0});
        }

        DeviceBuffer<DeviceTriangleFeature> deviceFeatures(triangleCount);
        DeviceBuffer<DeviceAdjacencyPair> deviceAdjacencyPairs(adjacencyPairs.size());
        DeviceBuffer<std::uint8_t> deviceMergeMask(adjacencyPairs.size());
        CudaStream stream;
        CudaEvent started;
        CudaEvent uploaded;
        CudaEvent kernelFinished;
        CudaEvent downloaded;

        started.record(stream.get());
        RECAD_CUDA_CHECK(cudaMemcpyAsync(deviceFeatures.get(),
                                         packedFeatures.data(),
                                         packedFeatures.size() * sizeof(DeviceTriangleFeature),
                                         cudaMemcpyHostToDevice,
                                         stream.get()));
        RECAD_CUDA_CHECK(cudaMemcpyAsync(deviceAdjacencyPairs.get(),
                                         adjacencyPairs.data(),
                                         adjacencyPairs.size() * sizeof(DeviceAdjacencyPair),
                                         cudaMemcpyHostToDevice,
                                         stream.get()));
        uploaded.record(stream.get());

        const int blocks = launchBlockCount(adjacencyPairs.size(), selected.properties);
        triangleAdjacencyMergeKernel<<<blocks, kThreadsPerBlock, 0, stream.get()>>>(
            deviceFeatures.get(),
            deviceAdjacencyPairs.get(),
            adjacencyPairs.size(),
            std::cos(options.maximumDihedralAngleRadians),
            options.maximumCentroidDistance,
            options.maximumCentroidDistanceToEdgeRatio,
            options.treatOppositeNormalsAsAligned,
            deviceMergeMask.get());
        RECAD_CUDA_CHECK(cudaGetLastError());
        kernelFinished.record(stream.get());

        RECAD_CUDA_CHECK(cudaMemcpyAsync(mergeMask.data(),
                                         deviceMergeMask.get(),
                                         mergeMask.size() * sizeof(std::uint8_t),
                                         cudaMemcpyDeviceToHost,
                                         stream.get()));
        downloaded.record(stream.get());
        downloaded.synchronize();

        result.timings.hostToDeviceMs = elapsedMilliseconds(started, uploaded);
        result.timings.kernelMs = elapsedMilliseconds(uploaded, kernelFinished);
        result.timings.deviceToHostMs = elapsedMilliseconds(kernelFinished, downloaded);
        result.timings.totalGpuMs = elapsedMilliseconds(started, downloaded);
    }

    const auto unionFindStarted = std::chrono::steady_clock::now();
    std::vector<std::uint32_t> parents(triangleCount);
    for (std::size_t index = 0; index < triangleCount; ++index) {
        parents[index] = static_cast<std::uint32_t>(index);
    }
    const auto findRoot = [&parents](std::uint32_t value) {
        while (parents[value] != value) {
            parents[value] = parents[parents[value]];
            value = parents[value];
        }
        return value;
    };
    for (std::size_t pairIndex = 0; pairIndex < adjacencyPairs.size(); ++pairIndex) {
        if (mergeMask[pairIndex] == 0U) continue;
        ++result.mergedAdjacencyPairCount;
        const std::uint32_t firstRoot = findRoot(adjacencyPairs[pairIndex].first);
        const std::uint32_t secondRoot = findRoot(adjacencyPairs[pairIndex].second);
        if (firstRoot == secondRoot) continue;
        const std::uint32_t lower = std::min(firstRoot, secondRoot);
        const std::uint32_t upper = std::max(firstRoot, secondRoot);
        parents[upper] = lower;
    }

    std::vector<std::int32_t> rootLabels(triangleCount, -1);
    for (std::size_t triangleIndex = 0; triangleIndex < triangleCount; ++triangleIndex) {
        const TriangleFeatures& feature = mesh.features[triangleIndex];
        if (feature.degenerate) {
            ++result.degenerateTriangleCount;
            continue;
        }
        const std::uint32_t root = findRoot(static_cast<std::uint32_t>(triangleIndex));
        std::int32_t& label = rootLabels[root];
        if (label < 0) {
            label = static_cast<std::int32_t>(result.regions.size());
            result.regions.emplace_back();
        }
        result.triangleLabels[triangleIndex] = label;
        CudaTriangleRegion& region = result.regions[static_cast<std::size_t>(label)];
        region.triangleIndices.push_back(static_cast<std::uint32_t>(triangleIndex));
        region.area += feature.area;
        region.centroid += feature.centroid * feature.area;
    }
    for (CudaTriangleRegion& region : result.regions) {
        if (region.area > 0.0) region.centroid /= region.area;
    }
    result.hostUnionFindMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - unionFindStarted).count();
    return result;
}

CudaResidualResult evaluatePlaneResidualsCuda(const std::vector<Vec3d>& points,
                                               const Vec3d& planeNormal,
                                               double planeOffset,
                                               double inlierTolerance,
                                               const std::vector<double>& weights,
                                               int deviceOrdinal) {
    if (!finite(planeNormal) || !std::isfinite(planeOffset)) {
        throw std::invalid_argument("Plane coefficients must be finite");
    }
    const double normalLength =
        std::hypot(planeNormal.x, planeNormal.y, planeNormal.z);
    if (!std::isfinite(normalLength) || normalLength <= 0.0) {
        throw std::invalid_argument("Plane normal must have non-zero finite length");
    }

    const double inverseLength = 1.0 / normalLength;
    const PlaneResidual model{{planeNormal.x * inverseLength,
                               planeNormal.y * inverseLength,
                               planeNormal.z * inverseLength},
                              planeOffset * inverseLength};
    return evaluateResiduals(points,
                             inlierTolerance,
                             weights,
                             deviceOrdinal,
                             model);
}

CudaResidualResult evaluateSphereResidualsCuda(const std::vector<Vec3d>& points,
                                                const Vec3d& center,
                                                double radius,
                                                double inlierTolerance,
                                                const std::vector<double>& weights,
                                                int deviceOrdinal) {
    if (!finite(center) || !std::isfinite(radius) || radius <= 0.0) {
        throw std::invalid_argument("Sphere center must be finite and radius must be positive");
    }
    const SphereResidual model{{center.x, center.y, center.z}, radius};
    return evaluateResiduals(points,
                             inlierTolerance,
                             weights,
                             deviceOrdinal,
                             model);
}

} // namespace recad
