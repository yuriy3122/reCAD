#include "CadReconstruction.h"
#include "CudaGeometry.h"
#include "Geometry.h"
#include "MeshLoader.h"

#include <Standard_Version.hxx>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
using nlohmann::json;

namespace recad {
namespace {

struct CommandLine {
    fs::path input{"testData/box_sphere_test-01.glb"};
    fs::path output{"output"};
    bool verifyExpectedStructure{};
};

CommandLine parseCommandLine(int argc, char** argv) {
    CommandLine result;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if ((argument == "--input" || argument == "-i") && i + 1 < argc) {
            result.input = fs::path(argv[++i]);
        } else if ((argument == "--output" || argument == "-o") && i + 1 < argc) {
            result.output = fs::path(argv[++i]);
        } else if (argument == "--verify-expected-structure") {
            result.verifyExpectedStructure = true;
        } else if (argument == "--help" || argument == "-h") {
            std::cout << "mesh2brep --input model.glb --output directory "
                         "[--verify-expected-structure]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown or incomplete argument: " + std::string(argument));
        }
    }
    return result;
}

std::vector<Vec3d> uniqueSurfacePoints(
    const Mesh& mesh, const std::vector<std::uint32_t>& triangles,
    std::vector<double>* weights = nullptr) {
    std::vector<double> vertexWeights(mesh.vertices.size(), 0.0);
    for (const std::uint32_t triangleIndex : triangles) {
        const Triangle& triangle = mesh.triangles.at(triangleIndex);
        const double contribution = mesh.features.at(triangleIndex).area / 3.0;
        for (const std::uint32_t vertex : triangle.vertices) vertexWeights[vertex] += contribution;
    }
    std::vector<Vec3d> points;
    points.reserve(mesh.vertices.size());
    if (weights) weights->clear();
    for (std::size_t i = 0; i < vertexWeights.size(); ++i) {
        if (vertexWeights[i] <= 0.0) continue;
        points.push_back(mesh.vertices[i]);
        if (weights) weights->push_back(vertexWeights[i]);
    }
    return points;
}

SphereFitResult refineAnalyticConstraints(
    const SphereFitResult& raw, const OrientedBoxFitResult& box,
    double scale, std::vector<std::string>& appliedConstraints) {
    SphereFitResult result = raw;
    const double tolerance = std::max({scale * 2.0e-5,
                                       raw.maxResidual * 32.0,
                                       std::numeric_limits<float>::epsilon() * scale * 16.0});
    std::vector<double> radiusObservations;

    for (std::size_t axisIndex = 0; axisIndex < 3; ++axisIndex) {
        const Vec3d& axis = box.axes[axisIndex];
        const double low = box.lowCoordinates[axisIndex];
        const double high = box.highCoordinates[axisIndex];
        const double middle = 0.5 * (low + high);
        const double halfDimension = 0.5 * (high - low);
        double coordinate = dot(result.center, axis);

        // An opposite pair of tangent planes jointly constrains both center
        // coordinate and radius. This is inferred from fit residuals, not from
        // an expected dimension or axis.
        if (std::abs(coordinate - middle) <= tolerance &&
            std::abs(result.radius - halfDimension) <= tolerance) {
            result.center += axis * (middle - coordinate);
            coordinate = middle;
            radiusObservations.push_back(halfDimension);
            appliedConstraints.push_back(
                "sphere tangent to opposite box supports on recovered axis " +
                std::to_string(axisIndex));
        }

        if (std::abs(coordinate - low) <= tolerance) {
            result.center += axis * (low - coordinate);
            appliedConstraints.push_back(
                "sphere center lies on low box support of recovered axis " +
                std::to_string(axisIndex));
        } else if (std::abs(coordinate - high) <= tolerance) {
            result.center += axis * (high - coordinate);
            appliedConstraints.push_back(
                "sphere center lies on high box support of recovered axis " +
                std::to_string(axisIndex));
        }
    }
    if (!radiusObservations.empty()) {
        result.radius = std::accumulate(radiusObservations.begin(), radiusObservations.end(), 0.0) /
                        static_cast<double>(radiusObservations.size());
    }
    return result;
}

double meshSurfaceArea(const Mesh& mesh) {
    return std::accumulate(mesh.features.begin(), mesh.features.end(), 0.0,
        [](double sum, const TriangleFeatures& feature) { return sum + feature.area; });
}

double meshSignedVolume(const Mesh& mesh) {
    double sixVolume = 0.0;
    for (const Triangle& triangle : mesh.triangles) {
        const Vec3d& a = mesh.vertices[triangle.vertices[0]];
        const Vec3d& b = mesh.vertices[triangle.vertices[1]];
        const Vec3d& c = mesh.vertices[triangle.vertices[2]];
        sixVolume += dot(a, cross(b, c));
    }
    return sixVolume / 6.0;
}

json vectorJson(const Vec3d& value) {
    return json::array({value.x, value.y, value.z});
}

json distanceJson(const DistanceStatistics& value) {
    return {{"rms", value.rms}, {"max", value.maximum}, {"mean", value.mean},
            {"samples", value.samples}, {"weight", value.weight}};
}

std::string compilerVersion() {
#ifdef _MSC_FULL_VER
    return std::to_string(_MSC_FULL_VER);
#else
    return "not MSVC";
#endif
}

std::string buildConfiguration() {
#ifdef _DEBUG
    return "x64 Debug";
#else
    return "x64 Release";
#endif
}

std::string cudaVersionString(int version) {
    if (version <= 0) return "unknown";
    const int major = version / 1000;
    const int minor = (version % 1000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
}

void writeSegmentationPly(
    const fs::path& path, const Mesh& mesh, const PlaneSegmentationResult& segmentation) {
    static constexpr std::array<std::array<int, 3>, 12> palette{{
        {{230, 57, 70}}, {{29, 53, 87}}, {{69, 123, 157}}, {{42, 157, 143}},
        {{233, 196, 106}}, {{244, 162, 97}}, {{231, 111, 81}}, {{142, 202, 230}},
        {{255, 183, 3}}, {{131, 56, 236}}, {{6, 214, 160}}, {{251, 86, 7}}
    }};
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("Cannot write " + path.string());
    stream << "ply\nformat ascii 1.0\n"
           << "element vertex " << mesh.vertices.size() << "\n"
           << "property double x\nproperty double y\nproperty double z\n"
           << "element face " << mesh.triangles.size() << "\n"
           << "property list uchar uint vertex_indices\n"
           << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    stream << std::setprecision(17);
    for (const Vec3d& vertex : mesh.vertices) {
        stream << vertex.x << ' ' << vertex.y << ' ' << vertex.z << '\n';
    }
    for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
        const Triangle& triangle = mesh.triangles[i];
        const std::int32_t label = segmentation.triangleLabels[i];
        const std::array<int, 3> color = label < 0
            ? std::array<int, 3>{165, 165, 165}
            : palette[static_cast<std::size_t>(label) % palette.size()];
        stream << "3 " << triangle.vertices[0] << ' ' << triangle.vertices[1] << ' '
               << triangle.vertices[2] << ' ' << color[0] << ' ' << color[1] << ' '
               << color[2] << '\n';
    }
}

struct SatStatus {
    bool found{};
    fs::path path;
    std::string status;
    std::string detail;
};

// Deliberately called only after final candidate selection and BREP/STEP export.
SatStatus inspectSatCapabilityAfterReconstruction(const fs::path& inputDirectory) {
    SatStatus result;
    std::error_code error;
    for (const fs::directory_entry& entry : fs::directory_iterator(inputDirectory, error)) {
        if (error) break;
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (entry.is_regular_file() && extension == ".sat") {
            result.found = true;
            result.path = entry.path();
            break;
        }
    }
    result.status = result.found ? "UNSUPPORTED" : "NOT_FOUND";
    result.detail = result.found
        ? "OCCT has no native ACIS SAT reader and no licensed SAT translator is installed; "
          "the SAT contents were not parsed and were not used by reconstruction or selection."
        : "No SAT reference file was found beside the GLB.";
    return result;
}

} // namespace
} // namespace recad

int main(int argc, char** argv) {
    using namespace recad;
    try {
        const CommandLine commandLine = parseCommandLine(argc, argv);
        const fs::path inputPath = fs::absolute(commandLine.input).lexically_normal();
        const fs::path outputDirectory = fs::absolute(commandLine.output).lexically_normal();
        fs::create_directories(outputDirectory);

        std::cout << "mesh2brep: GLB-only analytic reconstruction\n"
                  << "Input: " << inputPath.string() << "\n";
        MeshLoadResult loaded = loadGlbMesh(inputPath);
        Mesh& mesh = loaded.mesh;

        const CudaTriangleFeatureResult gpuFeatures = computeTriangleFeaturesCuda(
            mesh.vertices, mesh.triangles, 0.0);
        mesh.features = gpuFeatures.features;
        rebuildTopology(mesh);

        PlaneSegmentationOptions planeOptions;
        planeOptions.distanceTolerance = std::max(mesh.bounds.diagonal() * 1.0e-6,
                                                   mesh.weldTolerance * 8.0);
        planeOptions.angleToleranceRadians = 0.01308996938995747; // 0.75 degrees.
        planeOptions.minimumSupportAreaFraction = 5.0e-4;
        const PlaneSegmentationResult planes = segmentPlanarRegions(mesh, planeOptions);
        if (planes.segments.empty()) throw std::runtime_error("No significant planes were detected");
        if (planes.nonPlanarTriangles.empty()) {
            throw std::runtime_error("No curved region remains after plane segmentation");
        }

        const SphereFitResult rawSphere = fitSphere(mesh, planes.nonPlanarTriangles);
        if (!rawSphere.valid) throw std::runtime_error("Sphere fit failed: " + rawSphere.diagnostic);
        const OrientedBoxFitResult box = inferOrientedBox(mesh, planes.segments);
        if (!box.valid) throw std::runtime_error("Oriented box inference failed: " + box.diagnostic);

        std::vector<std::string> constraints;
        SphereFitResult sphere = refineAnalyticConstraints(
            rawSphere, box, mesh.bounds.diagonal(), constraints);
        std::vector<double> curvedWeights;
        const std::vector<Vec3d> curvedPoints = uniqueSurfacePoints(
            mesh, planes.nonPlanarTriangles, &curvedWeights);
        const double sphereTolerance = std::max(mesh.bounds.diagonal() * 1.0e-5,
                                                 rawSphere.maxResidual * 8.0);
        const CudaResidualResult sphereGpu = evaluateSphereResidualsCuda(
            curvedPoints, sphere.center, sphere.radius, sphereTolerance, curvedWeights,
            gpuFeatures.device.ordinal);
        sphere.rmsResidual = sphereGpu.statistics.weightedRmsResidual;
        sphere.maxResidual = sphereGpu.statistics.maxResidual;
        sphere.inlierFraction = sphereGpu.statistics.inlierWeightFraction;
        sphere.pointCount = curvedPoints.size();

        std::vector<CudaResidualStatistics> planeGpuStatistics;
        planeGpuStatistics.reserve(planes.segments.size());
        for (const PlaneSegment& segment : planes.segments) {
            std::vector<double> weights;
            const std::vector<Vec3d> points = uniqueSurfacePoints(mesh, segment.triangleIndices, &weights);
            planeGpuStatistics.push_back(evaluatePlaneResidualsCuda(
                points, segment.plane.normal, segment.plane.offset,
                planes.distanceTolerance, weights, gpuFeatures.device.ordinal).statistics);
        }

        const double modelingTolerance = std::max(mesh.bounds.diagonal() * 1.0e-8,
                                                   mesh.weldTolerance * 4.0);
        const double comparisonDeflection = mesh.bounds.diagonal() * 2.0e-4;
        std::cout << "Building and scoring CUT, FUSE, and COMMON...\n";
        CadReconstructionResult cad = buildAndSelectBoolean(
            box, sphere, mesh, modelingTolerance, comparisonDeflection);
        const BooleanCandidate& selected = cad.candidates[cad.selectedCandidate];
        if (!selected.properties.valid || !selected.properties.closedSolid ||
            selected.properties.solidCount != 1 ||
            selected.properties.planarFaceCount == 0 ||
            selected.properties.sphericalFaceCount == 0 ||
            selected.properties.otherSurfaceCount != 0 ||
            selected.properties.circleEdgeCount == 0) {
            throw std::runtime_error(
                "Selected Boolean did not produce a closed, valid analytic plane/sphere solid");
        }

        const fs::path brepPath = outputDirectory / "reconstructed.brep";
        const fs::path stepPath = outputDirectory / "reconstructed.step";
        writeBrepAndStep(cad.selectedShape, brepPath, stepPath);
        writeSegmentationPly(outputDirectory / "segmentation.ply", mesh, planes);

        // The reconstruction is now fixed and exported. Only at this point do
        // we look for the SAT reference, and then only to report translator
        // capability; no SAT geometry is fed back into the pipeline.
        const SatStatus sat = inspectSatCapabilityAfterReconstruction(inputPath.parent_path());

        json root;
        root["schema_version"] = 1;
        root["input"] = inputPath.string();
        root["environment"] = {
            {"build_configuration", buildConfiguration()},
            {"msvc_full_version", compilerVersion()},
            {"cmake_version", RECAD_CMAKE_VERSION},
            {"cuda_runtime_version", cudaVersionString(gpuFeatures.device.runtimeVersion)},
            {"cuda_driver_version", cudaVersionString(gpuFeatures.device.driverVersion)},
            {"cuda_compile_version", cudaVersionString(CUDART_VERSION)},
            {"gpu", gpuFeatures.device.name},
            {"compute_capability", std::to_string(gpuFeatures.device.computeCapabilityMajor) + "." +
                                   std::to_string(gpuFeatures.device.computeCapabilityMinor)},
            {"occt_version", OCC_VERSION_COMPLETE}
        };
        root["mesh"] = {
            {"raw_vertices", mesh.stats.rawVertexCount},
            {"welded_vertices", mesh.stats.weldedVertexCount},
            {"triangles", mesh.stats.triangleCount},
            {"bounds_min", vectorJson(mesh.bounds.minimum)},
            {"bounds_max", vectorJson(mesh.bounds.maximum)},
            {"diagonal", mesh.bounds.diagonal()},
            {"weld_tolerance", mesh.weldTolerance},
            {"components", mesh.stats.connectedComponentCount},
            {"boundary_edges", mesh.stats.boundaryEdgeCount},
            {"non_manifold_edges", mesh.stats.nonManifoldEdgeCount},
            {"degenerate_triangles", mesh.stats.discardedDegenerateTriangles + gpuFeatures.degenerateCount},
            {"watertight", mesh.stats.watertight},
            {"surface_area", meshSurfaceArea(mesh)},
            {"signed_volume", meshSignedVolume(mesh)}
        };
        root["cuda_timing_ms"] = {
            {"features_h2d", gpuFeatures.timings.hostToDeviceMs},
            {"features_kernel", gpuFeatures.timings.kernelMs},
            {"features_d2h", gpuFeatures.timings.deviceToHostMs},
            {"features_total_gpu", gpuFeatures.timings.totalGpuMs},
            {"sphere_kernel", sphereGpu.timings.kernelMs}
        };
        root["planes"] = json::array();
        for (std::size_t i = 0; i < planes.segments.size(); ++i) {
            const PlaneSegment& segment = planes.segments[i];
            root["planes"].push_back({
                {"normal", vectorJson(segment.plane.normal)},
                {"d", segment.plane.offset},
                {"triangles", segment.triangleIndices.size()},
                {"connected_components", segment.connectedComponents},
                {"area", segment.area},
                {"rms", segment.rmsResidual},
                {"max", segment.maxResidual},
                {"cuda_weighted_rms", planeGpuStatistics[i].weightedRmsResidual},
                {"cuda_inliers", planeGpuStatistics[i].inlierCount}
            });
        }
        root["sphere"] = {
            {"raw_center", vectorJson(rawSphere.center)},
            {"raw_radius", rawSphere.radius},
            {"center", vectorJson(sphere.center)},
            {"radius", sphere.radius},
            {"points", sphere.pointCount},
            {"inliers", sphereGpu.statistics.inlierCount},
            {"rms", sphere.rmsResidual},
            {"max", sphere.maxResidual},
            {"inlier_fraction", sphere.inlierFraction},
            {"radial_orientation", sphere.radialOrientation},
            {"inferred_constraints", constraints}
        };
        root["box"] = {
            {"center", vectorJson(box.center)},
            {"dimensions", vectorJson(box.dimensions)},
            {"axes", json::array({vectorJson(box.axes[0]), vectorJson(box.axes[1]), vectorJson(box.axes[2])})},
            {"low_coordinates", vectorJson(box.lowCoordinates)},
            {"high_coordinates", vectorJson(box.highCoordinates)},
            {"rms", box.rmsResidual},
            {"max", box.maxResidual},
            {"orthogonality_error", box.orthogonalityError}
        };
        root["box"]["supporting_planes"] = json::array();
        for (std::size_t i = 0; i < 3; ++i) {
            root["box"]["supporting_planes"].push_back({
                {"normal", vectorJson(-box.axes[i])}, {"d", box.lowCoordinates[i]}});
            root["box"]["supporting_planes"].push_back({
                {"normal", vectorJson(box.axes[i])}, {"d", -box.highCoordinates[i]}});
        }
        root["box"]["corners"] = json::array();
        for (unsigned mask = 0; mask < 8; ++mask) {
            Vec3d corner{};
            for (std::size_t i = 0; i < 3; ++i) {
                const double coordinate = (mask & (1u << i)) != 0
                    ? box.highCoordinates[i] : box.lowCoordinates[i];
                corner += box.axes[i] * coordinate;
            }
            root["box"]["corners"].push_back(vectorJson(corner));
        }
        root["boolean_candidates"] = json::array();
        for (const BooleanCandidate& candidate : cad.candidates) {
            root["boolean_candidates"].push_back({
                {"operation", toString(candidate.operation)},
                {"build_succeeded", candidate.buildSucceeded},
                {"valid", candidate.properties.valid},
                {"solid_count", candidate.properties.solidCount},
                {"score", candidate.score},
                {"bbox_error", candidate.boundingBoxError},
                {"mesh_to_brep", distanceJson(candidate.distances.meshToBrep)},
                {"brep_to_mesh", distanceJson(candidate.distances.brepToMesh)},
                {"diagnostic", candidate.diagnostic}
            });
        }
        root["selected_boolean"] = toString(selected.operation);
        root["brep"] = {
            {"valid", selected.properties.valid},
            {"closed_solid", selected.properties.closedSolid},
            {"solid_count", selected.properties.solidCount},
            {"face_count", selected.properties.faceCount},
            {"edge_count", selected.properties.edgeCount},
            {"vertex_count", selected.properties.vertexCount},
            {"planar_faces", selected.properties.planarFaceCount},
            {"spherical_faces", selected.properties.sphericalFaceCount},
            {"other_surfaces", selected.properties.otherSurfaceCount},
            {"circle_edges", selected.properties.circleEdgeCount},
            {"volume", selected.properties.volume},
            {"surface_area", selected.properties.surfaceArea},
            {"mesh_to_brep", distanceJson(selected.distances.meshToBrep)},
            {"brep_to_mesh", distanceJson(selected.distances.brepToMesh)}
        };
        root["sat_verification"] = {
            {"found", sat.found}, {"path", sat.path.string()},
            {"status", sat.status}, {"detail", sat.detail}
        };

        std::ostringstream report;
        report << std::setprecision(12) << std::fixed;
        report << "Build status: SUCCESS (" << buildConfiguration() << ")\n"
               << "MSVC version: " << compilerVersion() << "\n"
               << "CMake version: " << RECAD_CMAKE_VERSION << "\n"
               << "CUDA version: compile " << cudaVersionString(CUDART_VERSION)
               << ", runtime " << cudaVersionString(gpuFeatures.device.runtimeVersion) << "\n"
               << "GPU used: " << gpuFeatures.device.name << " (sm_"
               << gpuFeatures.device.computeCapabilityMajor << gpuFeatures.device.computeCapabilityMinor << ")\n"
               << "OpenCascade version: " << OCC_VERSION_COMPLETE << "\n\n"
               << "GLB raw vertices/triangles: " << mesh.stats.rawVertexCount << " / "
               << mesh.stats.triangleCount << "\n"
               << "Welded vertices: " << mesh.stats.weldedVertexCount << "\n"
               << "Bounds: [" << mesh.bounds.minimum.x << ", " << mesh.bounds.minimum.y << ", "
               << mesh.bounds.minimum.z << "] - [" << mesh.bounds.maximum.x << ", "
               << mesh.bounds.maximum.y << ", " << mesh.bounds.maximum.z << "]\n"
               << "Topology: components=" << mesh.stats.connectedComponentCount
               << ", boundary edges=" << mesh.stats.boundaryEdgeCount
               << ", non-manifold edges=" << mesh.stats.nonManifoldEdgeCount
               << ", degenerate triangles="
               << (mesh.stats.discardedDegenerateTriangles + gpuFeatures.degenerateCount)
               << ", watertight=" << (mesh.stats.watertight ? "YES" : "NO") << "\n"
               << "CUDA feature timing: H2D=" << gpuFeatures.timings.hostToDeviceMs
               << " ms, kernel=" << gpuFeatures.timings.kernelMs
               << " ms, D2H=" << gpuFeatures.timings.deviceToHostMs << " ms\n\n"
               << "Detected planes: " << planes.segments.size() << "\n";
        for (std::size_t i = 0; i < planes.segments.size(); ++i) {
            const PlaneSegment& plane = planes.segments[i];
            report << "  [" << i << "] n=(" << plane.plane.normal.x << ", "
                   << plane.plane.normal.y << ", " << plane.plane.normal.z << "), d="
                   << plane.plane.offset << ", triangles=" << plane.triangleIndices.size()
                   << ", components=" << plane.connectedComponents << ", RMS="
                   << plane.rmsResidual << ", max=" << plane.maxResidual << "\n";
        }
        report << "\nRecovered box center: (" << box.center.x << ", " << box.center.y << ", "
               << box.center.z << ")\n"
               << "Recovered box dimensions: (" << box.dimensions.x << ", "
               << box.dimensions.y << ", " << box.dimensions.z << ")\n";
        for (std::size_t i = 0; i < 3; ++i) {
            report << "  axis " << i << ": (" << box.axes[i].x << ", " << box.axes[i].y
                   << ", " << box.axes[i].z << ")\n";
        }
        report << "Recovered sphere center/radius: (" << sphere.center.x << ", "
               << sphere.center.y << ", " << sphere.center.z << ") / " << sphere.radius << "\n"
               << "Sphere fitting error RMS/max: " << sphere.rmsResidual << " / "
               << sphere.maxResidual << " (" << sphereGpu.statistics.inlierCount << "/"
               << sphere.pointCount << " inliers, weighted fraction="
               << sphere.inlierFraction << ")\n\n"
               << "Boolean candidate scores (lower is better):\n";
        for (const BooleanCandidate& candidate : cad.candidates) {
            report << "  " << toString(candidate.operation) << ": score=" << candidate.score
                   << ", GLB->BRep RMS/max=" << candidate.distances.meshToBrep.rms << "/"
                   << candidate.distances.meshToBrep.maximum << ", BRep->GLB RMS/max="
                   << candidate.distances.brepToMesh.rms << "/"
                   << candidate.distances.brepToMesh.maximum << ", valid="
                   << (candidate.properties.valid ? "YES" : "NO") << "\n";
        }
        report << "Selected Boolean: " << toString(selected.operation) << "\n\n"
               << "B-Rep valid: " << (selected.properties.valid ? "YES" : "NO") << "\n"
               << "Closed solid: " << (selected.properties.closedSolid ? "YES" : "NO") << "\n"
               << "Solid/face/edge/vertex counts: " << selected.properties.solidCount << "/"
               << selected.properties.faceCount << "/" << selected.properties.edgeCount << "/"
               << selected.properties.vertexCount << "\n"
               << "Analytic planar/spherical/other faces: " << selected.properties.planarFaceCount
               << "/" << selected.properties.sphericalFaceCount << "/"
               << selected.properties.otherSurfaceCount << "\n"
               << "Line/circle/other edges: " << selected.properties.lineEdgeCount << "/"
               << selected.properties.circleEdgeCount << "/" << selected.properties.otherCurveCount << "\n"
               << "Volume: " << selected.properties.volume << "\n"
               << "Surface area: " << selected.properties.surfaceArea << "\n"
               << "GLB -> BRep RMS/max: " << selected.distances.meshToBrep.rms << " / "
               << selected.distances.meshToBrep.maximum << "\n"
               << "BRep -> GLB RMS/max: " << selected.distances.brepToMesh.rms << " / "
               << selected.distances.brepToMesh.maximum << "\n\n"
               << "SAT verification status: " << sat.status << "\n"
               << sat.detail << "\n"
               << "Known limitations: unsigned distance scoring; STEP numeric coordinates are written as mm "
                  "because GLB contains no authoritative unit metadata; SAT verification requires an external "
                  "licensed ACIS translator.\n";

        {
            std::ofstream jsonStream(outputDirectory / "reconstruction.json");
            if (!jsonStream) throw std::runtime_error("Cannot write reconstruction.json");
            jsonStream << std::setw(2) << root << '\n';
        }
        {
            std::ofstream reportStream(outputDirectory / "report.txt");
            if (!reportStream) throw std::runtime_error("Cannot write report.txt");
            reportStream << report.str();
        }
        std::cout << report.str();

        if (commandLine.verifyExpectedStructure) {
            const double scale = std::max(
                mesh.bounds.diagonal(), std::numeric_limits<double>::min());
            const bool acceptable = selected.properties.valid && selected.properties.closedSolid &&
                selected.properties.planarFaceCount > 0 && selected.properties.sphericalFaceCount > 0 &&
                selected.properties.otherSurfaceCount == 0 &&
                selected.distances.symmetricRms < scale * 1.0e-3;
            if (!acceptable) throw std::runtime_error("Structural verification failed");
        }
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "mesh2brep error: " << exception.what() << '\n';
        return 1;
    }
}
