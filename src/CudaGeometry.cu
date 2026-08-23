#include "CudaGeometry.h"

#include <cuda_runtime.h>

#include <algorithm>
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
