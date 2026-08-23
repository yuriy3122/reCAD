#include "MeshLoader.h"

#include <nlohmann/json.hpp>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_JSON
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace recad {
namespace {

struct Mat4d {
    // glTF matrices are column-major.
    std::array<double, 16> values{};

    [[nodiscard]] static Mat4d identity() noexcept {
        Mat4d result;
        result.values[0] = 1.0;
        result.values[5] = 1.0;
        result.values[10] = 1.0;
        result.values[15] = 1.0;
        return result;
    }
};

[[nodiscard]] Mat4d operator*(const Mat4d& lhs, const Mat4d& rhs) noexcept {
    Mat4d result;
    for (std::size_t column = 0; column < 4U; ++column) {
        for (std::size_t row = 0; row < 4U; ++row) {
            double value = 0.0;
            for (std::size_t inner = 0; inner < 4U; ++inner) {
                value += lhs.values[inner * 4U + row] * rhs.values[column * 4U + inner];
            }
            result.values[column * 4U + row] = value;
        }
    }
    return result;
}

[[nodiscard]] Vec3d transformPoint(const Mat4d& transform, const Vec3d& point) {
    const double x = transform.values[0] * point.x + transform.values[4] * point.y +
                     transform.values[8] * point.z + transform.values[12];
    const double y = transform.values[1] * point.x + transform.values[5] * point.y +
                     transform.values[9] * point.z + transform.values[13];
    const double z = transform.values[2] * point.x + transform.values[6] * point.y +
                     transform.values[10] * point.z + transform.values[14];
    const double w = transform.values[3] * point.x + transform.values[7] * point.y +
                     transform.values[11] * point.z + transform.values[15];
    if (!std::isfinite(w) || std::abs(w) <= std::numeric_limits<double>::min()) {
        throw MeshLoadError("A node transform produced an invalid homogeneous coordinate");
    }
    Vec3d result{x, y, z};
    if (std::abs(w - 1.0) > std::numeric_limits<double>::epsilon() * 16.0) result /= w;
    if (!isFinite(result)) throw MeshLoadError("A node transform produced a non-finite position");
    // IEEE equality deliberately canonicalizes both signs of zero.
    if (result.x == 0.0) result.x = 0.0;
    if (result.y == 0.0) result.y = 0.0;
    if (result.z == 0.0) result.z = 0.0;
    return result;
}

[[nodiscard]] double linearDeterminant(const Mat4d& matrix) noexcept {
    const double a00 = matrix.values[0];
    const double a01 = matrix.values[4];
    const double a02 = matrix.values[8];
    const double a10 = matrix.values[1];
    const double a11 = matrix.values[5];
    const double a12 = matrix.values[9];
    const double a20 = matrix.values[2];
    const double a21 = matrix.values[6];
    const double a22 = matrix.values[10];
    return a00 * (a11 * a22 - a12 * a21) -
           a01 * (a10 * a22 - a12 * a20) +
           a02 * (a10 * a21 - a11 * a20);
}

[[nodiscard]] Mat4d nodeTransform(const tinygltf::Node& node) {
    if (!node.matrix.empty()) {
        if (node.matrix.size() != 16U) throw MeshLoadError("A glTF node matrix does not have 16 values");
        Mat4d result;
        std::copy(node.matrix.begin(), node.matrix.end(), result.values.begin());
        return result;
    }

    const Vec3d translation = node.translation.empty()
        ? Vec3d{}
        : (node.translation.size() == 3U
            ? Vec3d{node.translation[0], node.translation[1], node.translation[2]}
            : throw MeshLoadError("A glTF node translation does not have three values"));
    const Vec3d scale = node.scale.empty()
        ? Vec3d{1.0, 1.0, 1.0}
        : (node.scale.size() == 3U
            ? Vec3d{node.scale[0], node.scale[1], node.scale[2]}
            : throw MeshLoadError("A glTF node scale does not have three values"));
    std::array<double, 4> quaternion{0.0, 0.0, 0.0, 1.0};
    if (!node.rotation.empty()) {
        if (node.rotation.size() != 4U) throw MeshLoadError("A glTF node rotation does not have four values");
        std::copy(node.rotation.begin(), node.rotation.end(), quaternion.begin());
        const double length = std::sqrt(
            quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
            quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);
        if (!(length > std::numeric_limits<double>::min()) || !std::isfinite(length)) {
            throw MeshLoadError("A glTF node has an invalid rotation quaternion");
        }
        for (double& value : quaternion) value /= length;
    }

    const double x = quaternion[0];
    const double y = quaternion[1];
    const double z = quaternion[2];
    const double w = quaternion[3];
    const double r00 = 1.0 - 2.0 * (y * y + z * z);
    const double r01 = 2.0 * (x * y - z * w);
    const double r02 = 2.0 * (x * z + y * w);
    const double r10 = 2.0 * (x * y + z * w);
    const double r11 = 1.0 - 2.0 * (x * x + z * z);
    const double r12 = 2.0 * (y * z - x * w);
    const double r20 = 2.0 * (x * z - y * w);
    const double r21 = 2.0 * (y * z + x * w);
    const double r22 = 1.0 - 2.0 * (x * x + y * y);

    Mat4d result = Mat4d::identity();
    result.values[0] = r00 * scale.x;
    result.values[1] = r10 * scale.x;
    result.values[2] = r20 * scale.x;
    result.values[4] = r01 * scale.y;
    result.values[5] = r11 * scale.y;
    result.values[6] = r21 * scale.y;
    result.values[8] = r02 * scale.z;
    result.values[9] = r12 * scale.z;
    result.values[10] = r22 * scale.z;
    result.values[12] = translation.x;
    result.values[13] = translation.y;
    result.values[14] = translation.z;
    if (!std::all_of(result.values.begin(), result.values.end(), [](double value) {
            return std::isfinite(value);
        })) {
        throw MeshLoadError("A glTF node transform contains a non-finite value");
    }
    return result;
}

[[nodiscard]] std::size_t componentSize(int componentType) {
    switch (componentType) {
    case TINYGLTF_COMPONENT_TYPE_BYTE:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return 1U;
    case TINYGLTF_COMPONENT_TYPE_SHORT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return 2U;
    case TINYGLTF_COMPONENT_TYPE_INT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
        return 4U;
    case TINYGLTF_COMPONENT_TYPE_DOUBLE:
        return 8U;
    default:
        throw MeshLoadError("Unsupported glTF accessor component type " +
                            std::to_string(componentType));
    }
}

template <class Value>
[[nodiscard]] Value readUnaligned(const unsigned char* bytes) noexcept {
    Value value{};
    std::memcpy(&value, bytes, sizeof(Value));
    return value;
}

[[nodiscard]] double readComponent(const unsigned char* bytes,
                                   int componentType,
                                   bool normalizedValue) {
    switch (componentType) {
    case TINYGLTF_COMPONENT_TYPE_BYTE: {
        const auto value = readUnaligned<std::int8_t>(bytes);
        return normalizedValue ? std::max(-1.0, static_cast<double>(value) / 127.0)
                               : static_cast<double>(value);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
        const auto value = readUnaligned<std::uint8_t>(bytes);
        return normalizedValue ? static_cast<double>(value) / 255.0
                               : static_cast<double>(value);
    }
    case TINYGLTF_COMPONENT_TYPE_SHORT: {
        const auto value = readUnaligned<std::int16_t>(bytes);
        return normalizedValue ? std::max(-1.0, static_cast<double>(value) / 32767.0)
                               : static_cast<double>(value);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
        const auto value = readUnaligned<std::uint16_t>(bytes);
        return normalizedValue ? static_cast<double>(value) / 65535.0
                               : static_cast<double>(value);
    }
    case TINYGLTF_COMPONENT_TYPE_INT: {
        const auto value = readUnaligned<std::int32_t>(bytes);
        return normalizedValue
            ? std::max(-1.0, static_cast<double>(value) / 2147483647.0)
            : static_cast<double>(value);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
        const auto value = readUnaligned<std::uint32_t>(bytes);
        return normalizedValue ? static_cast<double>(value) / 4294967295.0
                               : static_cast<double>(value);
    }
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
        return static_cast<double>(readUnaligned<float>(bytes));
    case TINYGLTF_COMPONENT_TYPE_DOUBLE:
        return readUnaligned<double>(bytes);
    default:
        throw MeshLoadError("Unsupported glTF accessor component type");
    }
}

struct ResolvedView {
    const unsigned char* data{};
    std::size_t length{};
    std::size_t stride{};
};

[[nodiscard]] ResolvedView resolveView(const tinygltf::Model& model,
                                       int viewIndex,
                                       std::size_t relativeOffset,
                                       std::size_t stride,
                                       std::size_t elementSize,
                                       std::size_t elementCount,
                                       const char* purpose) {
    if (viewIndex < 0 || static_cast<std::size_t>(viewIndex) >= model.bufferViews.size()) {
        throw MeshLoadError(std::string("Invalid buffer view for ") + purpose);
    }
    const tinygltf::BufferView& view = model.bufferViews[static_cast<std::size_t>(viewIndex)];
    if (view.buffer < 0 || static_cast<std::size_t>(view.buffer) >= model.buffers.size()) {
        throw MeshLoadError(std::string("Invalid buffer reference for ") + purpose);
    }
    const tinygltf::Buffer& buffer = model.buffers[static_cast<std::size_t>(view.buffer)];
    if (stride < elementSize) throw MeshLoadError(std::string("Buffer stride is too small for ") + purpose);
    if (relativeOffset > view.byteLength) {
        throw MeshLoadError(std::string("Accessor offset exceeds its buffer view for ") + purpose);
    }
    std::size_t required = 0U;
    if (elementCount > 0U) {
        if (elementCount - 1U > (std::numeric_limits<std::size_t>::max() - elementSize) / stride) {
            throw MeshLoadError(std::string("Accessor byte range overflows for ") + purpose);
        }
        required = (elementCount - 1U) * stride + elementSize;
    }
    if (required > view.byteLength - relativeOffset ||
        view.byteOffset > buffer.data.size() ||
        relativeOffset > buffer.data.size() - view.byteOffset ||
        required > buffer.data.size() - view.byteOffset - relativeOffset) {
        throw MeshLoadError(std::string("Accessor byte range exceeds its buffer for ") + purpose);
    }
    return {buffer.data.data() + view.byteOffset + relativeOffset, required, stride};
}

[[nodiscard]] std::vector<double> materializeAccessor(const tinygltf::Model& model,
                                                      const tinygltf::Accessor& accessor,
                                                      int expectedComponents,
                                                      const char* purpose) {
    const int components = tinygltf::GetNumComponentsInType(static_cast<std::uint32_t>(accessor.type));
    if (components != expectedComponents) {
        throw MeshLoadError(std::string("Unexpected accessor shape for ") + purpose);
    }
    const std::size_t componentBytes = componentSize(accessor.componentType);
    const std::size_t elementBytes = componentBytes * static_cast<std::size_t>(components);
    if (accessor.count > std::numeric_limits<std::size_t>::max() /
                         static_cast<std::size_t>(components)) {
        throw MeshLoadError(std::string("Accessor element count overflows for ") + purpose);
    }
    std::vector<double> values(accessor.count * static_cast<std::size_t>(components), 0.0);

    if (accessor.bufferView >= 0) {
        const tinygltf::BufferView& view = model.bufferViews.at(
            static_cast<std::size_t>(accessor.bufferView));
        const int byteStride = accessor.ByteStride(view);
        if (byteStride < 0) throw MeshLoadError(std::string("Invalid accessor stride for ") + purpose);
        const ResolvedView resolved = resolveView(
            model, accessor.bufferView, accessor.byteOffset,
            static_cast<std::size_t>(byteStride), elementBytes, accessor.count, purpose);
        for (std::size_t element = 0; element < accessor.count; ++element) {
            const unsigned char* source = resolved.data + element * resolved.stride;
            for (int component = 0; component < components; ++component) {
                values[element * static_cast<std::size_t>(components) +
                       static_cast<std::size_t>(component)] = readComponent(
                    source + static_cast<std::size_t>(component) * componentBytes,
                    accessor.componentType, accessor.normalized);
            }
        }
    } else if (!accessor.sparse.isSparse) {
        throw MeshLoadError(std::string("Accessor has neither a buffer view nor sparse values for ") + purpose);
    }

    if (accessor.sparse.isSparse) {
        if (accessor.sparse.count < 0 ||
            static_cast<std::size_t>(accessor.sparse.count) > accessor.count) {
            throw MeshLoadError(std::string("Invalid sparse accessor count for ") + purpose);
        }
        const std::size_t sparseCount = static_cast<std::size_t>(accessor.sparse.count);
        const std::size_t sparseIndexBytes = componentSize(accessor.sparse.indices.componentType);
        if (accessor.sparse.indices.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE &&
            accessor.sparse.indices.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT &&
            accessor.sparse.indices.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
            throw MeshLoadError(std::string("Sparse accessor uses an invalid index type for ") + purpose);
        }
        const tinygltf::BufferView& indexView = model.bufferViews.at(
            static_cast<std::size_t>(accessor.sparse.indices.bufferView));
        const std::size_t indexStride = indexView.byteStride != 0U
            ? indexView.byteStride : sparseIndexBytes;
        const ResolvedView sparseIndices = resolveView(
            model, accessor.sparse.indices.bufferView, accessor.sparse.indices.byteOffset,
            indexStride, sparseIndexBytes, sparseCount, "sparse indices");
        const tinygltf::BufferView& valueView = model.bufferViews.at(
            static_cast<std::size_t>(accessor.sparse.values.bufferView));
        const std::size_t valueStride = valueView.byteStride != 0U
            ? valueView.byteStride : elementBytes;
        const ResolvedView sparseValues = resolveView(
            model, accessor.sparse.values.bufferView, accessor.sparse.values.byteOffset,
            valueStride, elementBytes, sparseCount, "sparse values");
        for (std::size_t sparse = 0; sparse < sparseCount; ++sparse) {
            const double sparseIndexValue = readComponent(
                sparseIndices.data + sparse * sparseIndices.stride,
                accessor.sparse.indices.componentType, false);
            if (sparseIndexValue < 0.0 || sparseIndexValue >= static_cast<double>(accessor.count) ||
                std::floor(sparseIndexValue) != sparseIndexValue) {
                throw MeshLoadError(std::string("Sparse accessor index is out of range for ") + purpose);
            }
            const std::size_t destination = static_cast<std::size_t>(sparseIndexValue);
            const unsigned char* source = sparseValues.data + sparse * sparseValues.stride;
            for (int component = 0; component < components; ++component) {
                values[destination * static_cast<std::size_t>(components) +
                       static_cast<std::size_t>(component)] = readComponent(
                    source + static_cast<std::size_t>(component) * componentBytes,
                    accessor.componentType, accessor.normalized);
            }
        }
    }
    return values;
}

[[nodiscard]] std::vector<Vec3d> readPositions(const tinygltf::Model& model,
                                               int accessorIndex) {
    if (accessorIndex < 0 || static_cast<std::size_t>(accessorIndex) >= model.accessors.size()) {
        throw MeshLoadError("A POSITION attribute references an invalid accessor");
    }
    const tinygltf::Accessor& accessor = model.accessors[static_cast<std::size_t>(accessorIndex)];
    const std::vector<double> values = materializeAccessor(model, accessor, 3, "POSITION");
    std::vector<Vec3d> positions;
    positions.reserve(accessor.count);
    for (std::size_t i = 0; i < accessor.count; ++i) {
        Vec3d position{values[i * 3U], values[i * 3U + 1U], values[i * 3U + 2U]};
        if (!isFinite(position)) throw MeshLoadError("A POSITION accessor contains a non-finite value");
        if (position.x == 0.0) position.x = 0.0;
        if (position.y == 0.0) position.y = 0.0;
        if (position.z == 0.0) position.z = 0.0;
        positions.push_back(position);
    }
    return positions;
}

[[nodiscard]] std::vector<std::uint32_t> readIndices(const tinygltf::Model& model,
                                                     int accessorIndex,
                                                     std::size_t positionCount) {
    if (accessorIndex < 0) {
        if (positionCount > std::numeric_limits<std::uint32_t>::max()) {
            throw MeshLoadError("A non-indexed primitive has too many vertices");
        }
        std::vector<std::uint32_t> indices(positionCount);
        std::iota(indices.begin(), indices.end(), 0U);
        return indices;
    }
    if (static_cast<std::size_t>(accessorIndex) >= model.accessors.size()) {
        throw MeshLoadError("A primitive references an invalid index accessor");
    }
    const tinygltf::Accessor& accessor = model.accessors[static_cast<std::size_t>(accessorIndex)];
    if (accessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE &&
        accessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT &&
        accessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
        throw MeshLoadError("An index accessor has a signed or floating-point component type");
    }
    const std::vector<double> values = materializeAccessor(model, accessor, 1, "indices");
    std::vector<std::uint32_t> indices;
    indices.reserve(accessor.count);
    for (const double value : values) {
        if (!std::isfinite(value) || value < 0.0 ||
            value > static_cast<double>(std::numeric_limits<std::uint32_t>::max()) ||
            std::floor(value) != value || value >= static_cast<double>(positionCount)) {
            throw MeshLoadError("An index accessor contains an out-of-range index");
        }
        indices.push_back(static_cast<std::uint32_t>(value));
    }
    return indices;
}

void appendPrimitiveTriangles(const std::vector<std::uint32_t>& indices,
                              int primitiveMode,
                              std::uint32_t vertexBase,
                              bool mirrored,
                              std::vector<Triangle>& triangles,
                              std::vector<std::string>& warnings) {
    const int mode = primitiveMode < 0 ? TINYGLTF_MODE_TRIANGLES : primitiveMode;
    const auto append = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c) {
        Triangle triangle{{vertexBase + a, vertexBase + b, vertexBase + c}};
        if (mirrored) std::swap(triangle.vertices[1], triangle.vertices[2]);
        triangles.push_back(triangle);
    };
    if (mode == TINYGLTF_MODE_TRIANGLES) {
        if (indices.size() % 3U != 0U) {
            warnings.emplace_back("A triangle primitive has trailing indices; they were ignored");
        }
        for (std::size_t i = 0; i + 2U < indices.size(); i += 3U) {
            append(indices[i], indices[i + 1U], indices[i + 2U]);
        }
    } else if (mode == TINYGLTF_MODE_TRIANGLE_STRIP) {
        for (std::size_t i = 0; i + 2U < indices.size(); ++i) {
            if ((i & 1U) == 0U) append(indices[i], indices[i + 1U], indices[i + 2U]);
            else append(indices[i + 1U], indices[i], indices[i + 2U]);
        }
    } else if (mode == TINYGLTF_MODE_TRIANGLE_FAN) {
        for (std::size_t i = 1; i + 1U < indices.size(); ++i) {
            append(indices[0], indices[i], indices[i + 1U]);
        }
    } else {
        warnings.emplace_back("A non-triangle glTF primitive was skipped");
    }
}

struct GridCell {
    std::int64_t x{};
    std::int64_t y{};
    std::int64_t z{};

    [[nodiscard]] bool operator==(const GridCell&) const noexcept = default;
};

struct GridCellHash {
    [[nodiscard]] std::size_t operator()(const GridCell& cell) const noexcept {
        const auto mix = [](std::uint64_t value) {
            value += 0x9e3779b97f4a7c15ULL;
            value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
            value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
            return value ^ (value >> 31U);
        };
        const std::uint64_t hx = mix(static_cast<std::uint64_t>(cell.x));
        const std::uint64_t hy = mix(static_cast<std::uint64_t>(cell.y));
        const std::uint64_t hz = mix(static_cast<std::uint64_t>(cell.z));
        return static_cast<std::size_t>(hx ^ (hy << 1U) ^ (hz << 7U));
    }
};

[[nodiscard]] std::int64_t gridCoordinate(double coordinate, double tolerance) noexcept {
    constexpr std::int64_t margin = 2;
    const long double value = std::floor(static_cast<long double>(coordinate) /
                                         static_cast<long double>(tolerance));
    const long double low = static_cast<long double>(std::numeric_limits<std::int64_t>::min() + margin);
    const long double high = static_cast<long double>(std::numeric_limits<std::int64_t>::max() - margin);
    return static_cast<std::int64_t>(std::clamp(value, low, high));
}

[[nodiscard]] double medianEdgeLength(const std::vector<Vec3d>& vertices,
                                      const std::vector<Triangle>& triangles,
                                      double diagonal) {
    std::vector<double> lengths;
    lengths.reserve(triangles.size() * 3U);
    const double minimum = std::max(diagonal, 1.0) *
                           std::numeric_limits<double>::epsilon() * 64.0;
    for (const Triangle& triangle : triangles) {
        for (std::size_t edge = 0; edge < 3U; ++edge) {
            const double length = norm(vertices[triangle.vertices[edge]] -
                                       vertices[triangle.vertices[(edge + 1U) % 3U]]);
            if (std::isfinite(length) && length > minimum) lengths.push_back(length);
        }
    }
    if (lengths.empty()) return diagonal;
    const auto middle = lengths.begin() + static_cast<std::ptrdiff_t>(lengths.size() / 2U);
    std::nth_element(lengths.begin(), middle, lengths.end());
    return *middle;
}

[[nodiscard]] double chooseWeldTolerance(const std::vector<Vec3d>& vertices,
                                         const std::vector<Triangle>& triangles,
                                         const Aabb& bounds,
                                         const MeshLoadOptions& options) {
    if (options.weldTolerance > 0.0) return options.weldTolerance;
    const double diagonal = bounds.diagonal();
    const double medianEdge = medianEdgeLength(vertices, triangles, diagonal);
    double maximumFloatUlp = 0.0;
    for (const Vec3d& vertex : vertices) {
        for (const double coordinate : {vertex.x, vertex.y, vertex.z}) {
            const float converted = static_cast<float>(coordinate);
            if (!std::isfinite(converted)) continue;
            const float next = std::nextafter(converted, std::numeric_limits<float>::infinity());
            maximumFloatUlp = std::max(maximumFloatUlp,
                std::abs(static_cast<double>(next) - static_cast<double>(converted)));
        }
    }
    const double base = std::max({options.minimumWeldTolerance,
                                  options.weldToleranceRelativeToDiagonal * diagonal,
                                  maximumFloatUlp * 4.0});
    const double edgeCap = options.weldToleranceRelativeToMedianEdge * medianEdge;
    const double tolerance = edgeCap > 0.0 ? std::min(base, edgeCap) : base;
    return std::max(options.minimumWeldTolerance, tolerance);
}

void repairWinding(Mesh& mesh, std::vector<std::string>& warnings) {
    struct OrientedUse {
        std::uint32_t triangle{};
        bool lowToHigh{};
    };
    struct Constraint {
        std::uint32_t neighbor{};
        bool differentFlip{};
    };
    struct LocalEdge {
        std::uint32_t low{};
        std::uint32_t high{};
        [[nodiscard]] bool operator==(const LocalEdge&) const noexcept = default;
    };
    struct LocalHash {
        [[nodiscard]] std::size_t operator()(const LocalEdge& edge) const noexcept {
            return static_cast<std::size_t>((static_cast<std::uint64_t>(edge.low) << 32U) | edge.high);
        }
    };

    std::unordered_map<LocalEdge, std::vector<OrientedUse>, LocalHash> edges;
    edges.reserve(mesh.triangles.size() * 3U);
    for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
        const Triangle& triangle = mesh.triangles[i];
        for (std::size_t edgeIndex = 0; edgeIndex < 3U; ++edgeIndex) {
            const std::uint32_t from = triangle.vertices[edgeIndex];
            const std::uint32_t to = triangle.vertices[(edgeIndex + 1U) % 3U];
            const LocalEdge edge{std::min(from, to), std::max(from, to)};
            edges[edge].push_back({static_cast<std::uint32_t>(i), from == edge.low});
        }
    }
    std::vector<std::vector<Constraint>> constraints(mesh.triangles.size());
    bool nonManifold = false;
    for (const auto& [edge, uses] : edges) {
        static_cast<void>(edge);
        if (uses.size() > 2U) nonManifold = true;
        for (std::size_t i = 0; i < uses.size(); ++i) {
            for (std::size_t j = i + 1U; j < uses.size(); ++j) {
                const bool mustDiffer = uses[i].lowToHigh == uses[j].lowToHigh;
                constraints[uses[i].triangle].push_back({uses[j].triangle, mustDiffer});
                constraints[uses[j].triangle].push_back({uses[i].triangle, mustDiffer});
            }
        }
    }
    if (nonManifold) warnings.emplace_back(
        "Non-manifold edges prevent an unambiguous global winding repair");

    std::vector<int> flips(mesh.triangles.size(), -1);
    std::vector<std::vector<std::uint32_t>> components;
    std::queue<std::uint32_t> queue;
    bool contradiction = false;
    for (std::size_t start = 0; start < mesh.triangles.size(); ++start) {
        if (flips[start] >= 0) continue;
        components.emplace_back();
        flips[start] = 0;
        queue.push(static_cast<std::uint32_t>(start));
        while (!queue.empty()) {
            const std::uint32_t current = queue.front();
            queue.pop();
            components.back().push_back(current);
            for (const Constraint& constraint : constraints[current]) {
                const int required = flips[current] ^ (constraint.differentFlip ? 1 : 0);
                if (flips[constraint.neighbor] < 0) {
                    flips[constraint.neighbor] = required;
                    queue.push(constraint.neighbor);
                } else if (flips[constraint.neighbor] != required) {
                    contradiction = true;
                }
            }
        }
    }
    if (contradiction) warnings.emplace_back(
        "The mesh contains an orientation contradiction; a best-effort winding repair was applied");
    for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
        if (flips[i] == 1) std::swap(mesh.triangles[i].vertices[1], mesh.triangles[i].vertices[2]);
    }

    // Choose outward orientation independently for each closed component. A
    // translated signed-volume sum avoids cancellation for large coordinates.
    for (const std::vector<std::uint32_t>& component : components) {
        bool closed = true;
        std::unordered_map<LocalEdge, std::size_t, LocalHash> incidence;
        std::unordered_map<std::uint32_t, unsigned char> componentVertices;
        for (const std::uint32_t triangleIndex : component) {
            const Triangle& triangle = mesh.triangles[triangleIndex];
            for (const std::uint32_t vertex : triangle.vertices) {
                componentVertices.emplace(vertex, static_cast<unsigned char>(0));
            }
            for (std::size_t edgeIndex = 0; edgeIndex < 3U; ++edgeIndex) {
                const std::uint32_t from = triangle.vertices[edgeIndex];
                const std::uint32_t to = triangle.vertices[(edgeIndex + 1U) % 3U];
                ++incidence[{std::min(from, to), std::max(from, to)}];
            }
        }
        for (const auto& [edge, count] : incidence) {
            static_cast<void>(edge);
            if (count != 2U) {
                closed = false;
                break;
            }
        }
        if (!closed || componentVertices.empty()) continue;
        Vec3d reference{};
        for (const auto& [vertex, unused] : componentVertices) {
            static_cast<void>(unused);
            reference += mesh.vertices[vertex];
        }
        reference /= static_cast<double>(componentVertices.size());
        long double sixTimesVolume = 0.0L;
        for (const std::uint32_t triangleIndex : component) {
            const Triangle& triangle = mesh.triangles[triangleIndex];
            const Vec3d a = mesh.vertices[triangle.vertices[0]] - reference;
            const Vec3d b = mesh.vertices[triangle.vertices[1]] - reference;
            const Vec3d c = mesh.vertices[triangle.vertices[2]] - reference;
            sixTimesVolume += static_cast<long double>(dot(a, cross(b, c)));
        }
        if (sixTimesVolume < 0.0L) {
            for (const std::uint32_t triangleIndex : component) {
                std::swap(mesh.triangles[triangleIndex].vertices[1],
                          mesh.triangles[triangleIndex].vertices[2]);
            }
        }
    }
}

[[nodiscard]] Mesh weldAndPrepare(std::vector<Vec3d> rawVertices,
                                  const std::vector<Triangle>& rawTriangles,
                                  const MeshLoadOptions& options,
                                  std::vector<std::string>& warnings) {
    Mesh mesh;
    mesh.stats.rawVertexCount = rawVertices.size();
    mesh.stats.inputTriangleCount = rawTriangles.size();
    for (Vec3d& vertex : rawVertices) {
        if (vertex.x == 0.0) vertex.x = 0.0;
        if (vertex.y == 0.0) vertex.y = 0.0;
        if (vertex.z == 0.0) vertex.z = 0.0;
        mesh.bounds.expand(vertex);
    }
    if (!mesh.bounds.valid() || !(mesh.bounds.diagonal() > 0.0)) {
        throw MeshLoadError("The loaded mesh has empty or zero-size bounds");
    }
    mesh.weldTolerance = chooseWeldTolerance(
        rawVertices, rawTriangles, mesh.bounds, options);
    if (!(mesh.weldTolerance > 0.0) || !std::isfinite(mesh.weldTolerance)) {
        throw MeshLoadError("The computed vertex weld tolerance is invalid");
    }

    std::unordered_map<GridCell, std::vector<std::uint32_t>, GridCellHash> buckets;
    buckets.reserve(rawVertices.size());
    std::vector<std::uint32_t> remap(rawVertices.size());
    mesh.vertices.reserve(rawVertices.size());
    const double toleranceSquared = mesh.weldTolerance * mesh.weldTolerance;
    for (std::size_t rawIndex = 0; rawIndex < rawVertices.size(); ++rawIndex) {
        const Vec3d& point = rawVertices[rawIndex];
        const GridCell cell{gridCoordinate(point.x, mesh.weldTolerance),
                            gridCoordinate(point.y, mesh.weldTolerance),
                            gridCoordinate(point.z, mesh.weldTolerance)};
        std::uint32_t match = std::numeric_limits<std::uint32_t>::max();
        double bestDistance = std::numeric_limits<double>::infinity();
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    const auto found = buckets.find({cell.x + dx, cell.y + dy, cell.z + dz});
                    if (found == buckets.end()) continue;
                    for (const std::uint32_t candidate : found->second) {
                        const double distance = squaredNorm(point - mesh.vertices[candidate]);
                        if (distance <= toleranceSquared &&
                            (distance < bestDistance ||
                             (distance == bestDistance && candidate < match))) {
                            bestDistance = distance;
                            match = candidate;
                        }
                    }
                }
            }
        }
        if (match == std::numeric_limits<std::uint32_t>::max()) {
            if (mesh.vertices.size() >= std::numeric_limits<std::uint32_t>::max()) {
                throw MeshLoadError("The welded mesh exceeds 32-bit vertex indexing");
            }
            match = static_cast<std::uint32_t>(mesh.vertices.size());
            mesh.vertices.push_back(point);
            buckets[cell].push_back(match);
        }
        remap[rawIndex] = match;
    }

    mesh.triangles.reserve(rawTriangles.size());
    const double degeneracyThreshold = mesh.bounds.diagonal() * mesh.bounds.diagonal() *
                                       std::numeric_limits<double>::epsilon() * 64.0;
    for (const Triangle& rawTriangle : rawTriangles) {
        Triangle triangle{{remap.at(rawTriangle.vertices[0]),
                           remap.at(rawTriangle.vertices[1]),
                           remap.at(rawTriangle.vertices[2])}};
        bool degenerate = triangle.vertices[0] == triangle.vertices[1] ||
                          triangle.vertices[1] == triangle.vertices[2] ||
                          triangle.vertices[2] == triangle.vertices[0];
        if (!degenerate) {
            const Vec3d& a = mesh.vertices[triangle.vertices[0]];
            const Vec3d& b = mesh.vertices[triangle.vertices[1]];
            const Vec3d& c = mesh.vertices[triangle.vertices[2]];
            degenerate = norm(cross(b - a, c - a)) <= degeneracyThreshold;
        }
        if (degenerate && options.discardDegenerateTriangles) {
            ++mesh.stats.discardedDegenerateTriangles;
        } else {
            mesh.triangles.push_back(triangle);
        }
    }
    if (mesh.triangles.empty()) throw MeshLoadError("No usable triangles remain after welding");
    if (mesh.stats.discardedDegenerateTriangles > 0U) {
        warnings.push_back(std::to_string(mesh.stats.discardedDegenerateTriangles) +
                           " degenerate triangle(s) were discarded after welding");
    }

    if (options.repairTriangleWinding) repairWinding(mesh, warnings);
    computeTriangleFeatures(mesh, degeneracyThreshold);
    rebuildTopology(mesh);
    return mesh;
}

} // namespace

MeshLoadResult loadGlbMesh(const std::filesystem::path& path,
                           const MeshLoadOptions& options) {
    if (path.empty()) throw MeshLoadError("The GLB input path is empty");
    if (!std::filesystem::exists(path) || !std::filesystem::is_regular_file(path)) {
        throw MeshLoadError("GLB input does not exist or is not a regular file: " + path.string());
    }
    if (options.weldTolerance < 0.0 ||
        options.weldToleranceRelativeToDiagonal < 0.0 ||
        options.weldToleranceRelativeToMedianEdge < 0.0 ||
        !(options.minimumWeldTolerance > 0.0) ||
        !std::isfinite(options.minimumWeldTolerance)) {
        throw MeshLoadError("Invalid mesh-loading tolerance options");
    }

    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw MeshLoadError("Could not open GLB input: " + path.string());
    const std::streamoff end = stream.tellg();
    if (end <= 0 || static_cast<std::uint64_t>(end) >
                        static_cast<std::uint64_t>(std::numeric_limits<unsigned int>::max())) {
        throw MeshLoadError("GLB input is empty or exceeds tinygltf's in-memory size limit");
    }
    std::vector<unsigned char> bytes(static_cast<std::size_t>(end));
    stream.seekg(0, std::ios::beg);
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), end)) {
        throw MeshLoadError("Could not read the complete GLB input");
    }

    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string error;
    std::string warning;
    const std::string baseDirectory = path.parent_path().string();
    if (!loader.LoadBinaryFromMemory(&model, &error, &warning, bytes.data(),
                                     static_cast<unsigned int>(bytes.size()), baseDirectory)) {
        throw MeshLoadError("tinygltf could not parse the GLB: " + error);
    }

    MeshLoadResult result;
    result.sourcePath = path;
    if (!warning.empty()) result.warnings.push_back("tinygltf: " + warning);
    if (!error.empty()) result.warnings.push_back("tinygltf: " + error);

    std::vector<Vec3d> rawVertices;
    std::vector<Triangle> rawTriangles;
    std::vector<unsigned char> recursionStack(model.nodes.size(), 0U);
    std::function<void(int, const Mat4d&, std::size_t)> visitNode;
    visitNode = [&](int nodeIndex, const Mat4d& parentTransform, std::size_t depth) {
        if (nodeIndex < 0 || static_cast<std::size_t>(nodeIndex) >= model.nodes.size()) {
            throw MeshLoadError("A scene references an invalid node index");
        }
        if (depth > model.nodes.size()) throw MeshLoadError("The glTF node hierarchy is cyclic");
        if (recursionStack[static_cast<std::size_t>(nodeIndex)]) {
            throw MeshLoadError("The glTF node hierarchy contains a cycle");
        }
        recursionStack[static_cast<std::size_t>(nodeIndex)] = 1U;
        ++result.visitedNodeCount;
        const tinygltf::Node& node = model.nodes[static_cast<std::size_t>(nodeIndex)];
        const Mat4d world = options.applySceneTransforms
            ? parentTransform * nodeTransform(node)
            : Mat4d::identity();
        if (node.skin >= 0) {
            result.warnings.emplace_back("Skin deformation is not applied to static CAD mesh positions");
        }
        if (node.mesh >= 0) {
            if (static_cast<std::size_t>(node.mesh) >= model.meshes.size()) {
                throw MeshLoadError("A node references an invalid mesh index");
            }
            const tinygltf::Mesh& gltfMesh = model.meshes[static_cast<std::size_t>(node.mesh)];
            if (!node.weights.empty() || !gltfMesh.weights.empty()) {
                result.warnings.emplace_back("Morph target weights are present but are not applied");
            }
            for (const tinygltf::Primitive& primitive : gltfMesh.primitives) {
                const auto positionAttribute = primitive.attributes.find("POSITION");
                if (positionAttribute == primitive.attributes.end()) {
                    result.warnings.emplace_back("A primitive without POSITION data was skipped");
                    continue;
                }
                std::vector<Vec3d> positions = readPositions(model, positionAttribute->second);
                const std::vector<std::uint32_t> indices = readIndices(
                    model, primitive.indices, positions.size());
                if (rawVertices.size() + positions.size() >
                    static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
                    throw MeshLoadError("Expanded scene exceeds 32-bit vertex indexing");
                }
                const std::uint32_t vertexBase = static_cast<std::uint32_t>(rawVertices.size());
                for (const Vec3d& position : positions) rawVertices.push_back(transformPoint(world, position));
                const std::size_t triangleCountBefore = rawTriangles.size();
                appendPrimitiveTriangles(indices, primitive.mode, vertexBase,
                                         options.applySceneTransforms && linearDeterminant(world) < 0.0,
                                         rawTriangles, result.warnings);
                if (rawTriangles.size() > triangleCountBefore) ++result.loadedPrimitiveCount;
            }
        }
        for (const int child : node.children) visitNode(child, world, depth + 1U);
        recursionStack[static_cast<std::size_t>(nodeIndex)] = 0U;
    };

    std::vector<int> rootNodes;
    if (!model.scenes.empty()) {
        int sceneIndex = options.sceneIndex;
        if (sceneIndex < 0) sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
        if (sceneIndex < 0 || static_cast<std::size_t>(sceneIndex) >= model.scenes.size()) {
            throw MeshLoadError("The requested glTF scene index is out of range");
        }
        rootNodes = model.scenes[static_cast<std::size_t>(sceneIndex)].nodes;
    } else {
        std::vector<unsigned char> isChild(model.nodes.size(), 0U);
        for (const tinygltf::Node& node : model.nodes) {
            for (const int child : node.children) {
                if (child < 0 || static_cast<std::size_t>(child) >= model.nodes.size()) {
                    throw MeshLoadError("A node references an invalid child index");
                }
                isChild[static_cast<std::size_t>(child)] = 1U;
            }
        }
        for (std::size_t i = 0; i < isChild.size(); ++i) {
            if (!isChild[i]) rootNodes.push_back(static_cast<int>(i));
        }
        result.warnings.emplace_back("The glTF has no scene; all root nodes were traversed");
    }
    for (const int root : rootNodes) visitNode(root, Mat4d::identity(), 0U);
    if (rawTriangles.empty() || rawVertices.empty()) {
        throw MeshLoadError("The selected glTF scene contains no triangle geometry");
    }
    result.mesh = weldAndPrepare(std::move(rawVertices), rawTriangles, options, result.warnings);
    return result;
}

} // namespace recad
