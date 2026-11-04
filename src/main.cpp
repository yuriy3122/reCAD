#include "CadReconstruction.h"
#include "CudaGeometry.h"
#include "MeshLoader.h"
#include "PatchReconstruction.h"
#include "SurfaceFitting.h"
#include "ReferenceValidation.h"

#include <Standard_Version.hxx>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace fs = std::filesystem;
using nlohmann::json;

namespace recad {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct CommandLine {
    fs::path input{"testData/box_sphere_test-01.glb"};
    fs::path output{"output"};
    // glTF defines linear coordinates in metres. The OCCT STEP writer is
    // configured for millimetres, so 1000 is the interoperable default.
    double inputScale{1000.0};
    double smoothAngleDegrees{30.0};
    double fitTolerance{};
    bool verify{};
    fs::path referenceStep;
    bool referenceZUp{};
    std::size_t referenceSamples{512};
    bool verifyAnalytic{};
    bool ignoreSourcePatches{};
};

double parseNumber(std::string_view text, const char* option) {
    std::size_t consumed = 0;
    const std::string value(text);
    double result = 0.0;
    try {
        result = std::stod(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) + " requires a number");
    }
    if (consumed != value.size() || !std::isfinite(result)) {
        throw std::invalid_argument(std::string(option) + " requires a finite number");
    }
    return result;
}

CommandLine parseCommandLine(int argc, char** argv) {
    CommandLine result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto requireValue = [&](const char* option) -> std::string_view {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string("Missing value for ") + option);
            }
            return argv[++index];
        };

        if (argument == "--input" || argument == "-i") {
            result.input = fs::path(requireValue("--input"));
        } else if (argument == "--output" || argument == "-o") {
            result.output = fs::path(requireValue("--output"));
        } else if (argument == "--input-scale") {
            result.inputScale = parseNumber(requireValue("--input-scale"), "--input-scale");
            if (!(result.inputScale > 0.0)) {
                throw std::invalid_argument("--input-scale must be positive");
            }
        } else if (argument == "--smooth-angle") {
            result.smoothAngleDegrees = parseNumber(
                requireValue("--smooth-angle"), "--smooth-angle");
            if (!(result.smoothAngleDegrees > 0.0 && result.smoothAngleDegrees < 180.0)) {
                throw std::invalid_argument("--smooth-angle must be between 0 and 180 degrees");
            }
        } else if (argument == "--fit-tolerance") {
            result.fitTolerance = parseNumber(
                requireValue("--fit-tolerance"), "--fit-tolerance");
            if (result.fitTolerance < 0.0) {
                throw std::invalid_argument("--fit-tolerance must be non-negative");
            }

        } else if (argument == "--ignore-source-patches") {
            result.ignoreSourcePatches = true;
        } else if (argument == "--reference-step") {
            result.referenceStep = fs::path(requireValue("--reference-step"));
        } else if (argument == "--reference-z-up") {
            result.referenceZUp = true;
        } else if (argument == "--reference-samples") {
            const double value = parseNumber(requireValue("--reference-samples"), "--reference-samples");
            if (value < 1 || value > 1000000 || std::floor(value) != value) {
                throw std::invalid_argument("--reference-samples must be an integer in [1, 1000000]");
            }
            result.referenceSamples = static_cast<std::size_t>(value);
        } else if (argument == "--verify-analytic") {
            result.verify = true;
            result.verifyAnalytic = true;
        } else if (argument == "--verify" || argument == "--verify-expected-structure") {
            result.verify = true;
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "mesh2brep --input model.glb --output directory [options]\n\n"
                << "Options:\n"
                << "  --input-scale value    GLB-to-STEP coordinate scale (default 1000)\n"
                << "  --smooth-angle deg    CUDA adjacency threshold (default 30)\n"
                << "  --fit-tolerance value Analytic surface tolerance; 0 is automatic\n"
                << "  --ignore-source-patches Use geometric CUDA segmentation only\n"
                << "  --reference-step path Compare against STEP after export\n"
                << "  --reference-z-up      Rotate CAD reference into glTF Y-up frame\n"
                << "  --reference-samples n Surface/edge sample limit per direction (default 512)\n"
                << "  --verify-analytic     Also require all output faces to be analytic\n"
                << "  --verify               Fail unless a valid close approximation is built\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + std::string(argument));
        }
    }
    return result;
}

double meshSurfaceArea(const Mesh& mesh) {
    return std::accumulate(
        mesh.features.begin(), mesh.features.end(), 0.0,
        [](double sum, const TriangleFeatures& feature) {
            return sum + (feature.degenerate ? 0.0 : feature.area);
        });
}

double meshSignedVolume(const Mesh& mesh) {
    long double sixVolume = 0.0L;
    const Vec3d reference = mesh.bounds.center();
    for (const Triangle& triangle : mesh.triangles) {
        const Vec3d a = mesh.vertices[triangle[0]] - reference;
        const Vec3d b = mesh.vertices[triangle[1]] - reference;
        const Vec3d c = mesh.vertices[triangle[2]] - reference;
        sixVolume += static_cast<long double>(dot(a, cross(b, c)));
    }
    return static_cast<double>(sixVolume / 6.0L);
}

json vectorJson(const Vec3d& value) {
    return json::array({value.x, value.y, value.z});
}

json distanceJson(const DistanceStatistics& value) {
    return {{"rms", value.rms},
            {"max", value.maximum},
            {"mean", value.mean},
            {"samples", value.samples},
            {"weight", value.weight}};
}

json shapePropertiesJson(const ShapeProperties& value) {
    return {
        {"valid", value.valid},
        {"closed_solid", value.closedSolid},
        {"solids", value.solidCount},
        {"shells", value.shellCount},
        {"faces", value.faceCount},
        {"edges", value.edgeCount},
        {"vertices", value.vertexCount},
        {"planes", value.planarFaceCount},
        {"cylinders", value.cylindricalFaceCount},
        {"cones", value.conicalFaceCount},
        {"spheres", value.sphericalFaceCount},
        {"tori", value.toroidalFaceCount},
        {"bspline_surfaces", value.bsplineFaceCount},
        {"other_surfaces", value.otherSurfaceCount},
        {"lines", value.lineEdgeCount},
        {"circles", value.circleEdgeCount},
        {"bspline_curves", value.bsplineEdgeCount},
        {"other_curves", value.otherCurveCount},
        {"volume", value.volume},
        {"surface_area", value.surfaceArea},
        {"bounds_min", vectorJson(value.bounds.minimum)},
        {"bounds_max", vectorJson(value.bounds.maximum)}
    };
}

json surfacePatchJson(const SurfacePatchFit& patch) {
    json result = {
        {"kind", toString(patch.kind)},
        {"accepted", patch.accepted},
        {"triangles", patch.triangleIndices.size()},
        {"score", patch.score},
        {"diagnostic", patch.diagnostic},
        {"area", patch.residuals.area},
        {"position_rms", patch.residuals.positionRms},
        {"position_max", patch.residuals.positionMax},
        {"normal_rms_radians", patch.residuals.normalRmsRadians},
        {"normal_max_radians", patch.residuals.normalMaxRadians},
        {"normal_excess_rms_radians", patch.residuals.normalExcessRmsRadians},
        {"normal_excess_max_radians", patch.residuals.normalExcessMaxRadians},
        {"inlier_area_fraction", patch.residuals.inlierAreaFraction}
    };
    if (const Plane* plane = std::get_if<Plane>(&patch.parameters)) {
        result["parameters"] = {
            {"normal", vectorJson(plane->normal)}, {"offset", plane->offset}};
    } else if (const CylinderSurface* cylinder =
                   std::get_if<CylinderSurface>(&patch.parameters)) {
        result["parameters"] = {
            {"axis_point", vectorJson(cylinder->axisPoint)},
            {"axis_direction", vectorJson(cylinder->axisDirection)},
            {"radius", cylinder->radius}};
    } else if (const ConeSurface* cone = std::get_if<ConeSurface>(&patch.parameters)) {
        result["parameters"] = {{"apex", vectorJson(cone->apex)},
                                {"axis_direction", vectorJson(cone->axisDirection)},
                                {"semi_angle_radians", cone->semiAngle}};
    } else if (const SphereSurface* sphere =
                   std::get_if<SphereSurface>(&patch.parameters)) {
        result["parameters"] = {
            {"center", vectorJson(sphere->center)}, {"radius", sphere->radius}};
    } else if (const TorusSurface* torus =
                   std::get_if<TorusSurface>(&patch.parameters)) {
        result["parameters"] = {
            {"center", vectorJson(torus->center)},
            {"axis_direction", vectorJson(torus->axisDirection)},
            {"major_radius", torus->majorRadius},
            {"minor_radius", torus->minorRadius}};
    }
    return result;
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
    return std::to_string(version / 1000) + "." +
           std::to_string((version % 1000) / 10);
}

void writeSegmentationPly(const fs::path& path,
                          const Mesh& mesh,
                          const std::vector<std::int32_t>& labels) {
    if (labels.size() != mesh.triangles.size()) {
        throw std::invalid_argument("Segmentation label count does not match the mesh");
    }
    static constexpr std::array<std::array<int, 3>, 16> palette{{
        {{230, 57, 70}}, {{29, 53, 87}}, {{69, 123, 157}}, {{42, 157, 143}},
        {{233, 196, 106}}, {{244, 162, 97}}, {{231, 111, 81}}, {{142, 202, 230}},
        {{255, 183, 3}}, {{131, 56, 236}}, {{6, 214, 160}}, {{251, 86, 7}},
        {{0, 150, 199}}, {{144, 190, 109}}, {{181, 101, 167}}, {{255, 127, 14}}
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
    for (std::size_t index = 0; index < mesh.triangles.size(); ++index) {
        const Triangle& triangle = mesh.triangles[index];
        const std::int32_t label = labels[index];
        const std::array<int, 3> color = label < 0
            ? std::array<int, 3>{165, 165, 165}
            : palette[static_cast<std::size_t>(label) % palette.size()];
        stream << "3 " << triangle[0] << ' ' << triangle[1] << ' ' << triangle[2]
               << ' ' << color[0] << ' ' << color[1] << ' ' << color[2] << '\n';
    }
}

std::vector<std::int32_t> labelFittedPatches(
    const Mesh& mesh, std::vector<SurfacePatchFit>& patches) {
    std::vector<std::int32_t> labels(mesh.triangles.size(), -1);
    for (std::size_t patchIndex = 0; patchIndex < patches.size(); ++patchIndex) {
        for (const std::uint32_t triangle : patches[patchIndex].triangleIndices) {
            if (triangle >= labels.size()) {
                throw std::runtime_error("Surface fitter returned an out-of-range triangle");
            }
            if (labels[triangle] >= 0) {
                throw std::runtime_error("Surface fitter returned overlapping patches");
            }
            labels[triangle] = static_cast<std::int32_t>(patchIndex);
        }
    }

    if (std::find(labels.begin(), labels.end(), -1) != labels.end()) {
        throw std::runtime_error("Surface fitting left triangles unassigned");
    }
    return labels;
}

} // namespace
} // namespace recad

int main(int argc, char** argv) {
    using namespace recad;
    fs::path statusPath;
    try {
        const CommandLine command = parseCommandLine(argc, argv);
        const fs::path inputPath = fs::absolute(command.input).lexically_normal();
        const fs::path outputDirectory = fs::absolute(command.output).lexically_normal();
        fs::create_directories(outputDirectory);
        statusPath = outputDirectory / "run_status.json";
        { std::ofstream status(statusPath); status << json{{"status", "running"}} << '\n'; }

        std::cout << "mesh2brep: CUDA-segmented analytic/NURBS reconstruction\n"
                  << "Input: " << inputPath.string() << "\n"
                  << "Input coordinate scale: " << command.inputScale << "\n";

        MeshLoadOptions loadOptions;
        loadOptions.linearUnitScale = command.inputScale;
        MeshLoadResult loaded = loadGlbMesh(inputPath, loadOptions);
        Mesh& mesh = loaded.mesh;

        const CudaTriangleFeatureResult gpuFeatures = computeTriangleFeaturesCuda(
            mesh.vertices, mesh.triangles, 0.0);
        mesh.features = gpuFeatures.features;
        rebuildTopology(mesh);

        CudaTriangleSegmentationOptions segmentationOptions;
        // When pre-weld GLB connectivity is present it is a stronger face
        // boundary signal than a tessellation-dependent dihedral threshold.
        // Pi keeps every source face connected while the CUDA stage still
        // classifies all adjacency and enforces source boundaries.
        const bool usefulSourcePatches = !command.ignoreSourcePatches &&
            mesh.sourcePatchCount > 1U &&
            mesh.sourcePatchCount <= mesh.triangles.size() / 2U;
        const double effectiveSmoothAngleDegrees = usefulSourcePatches
            ? 180.0
            : command.smoothAngleDegrees;
        segmentationOptions.maximumDihedralAngleRadians =
            effectiveSmoothAngleDegrees * kPi / 180.0;
        segmentationOptions.respectSourcePatchBoundaries = usefulSourcePatches;
        const CudaTriangleSegmentationResult segmentation =
            segmentTriangleRegionsCuda(mesh, segmentationOptions, gpuFeatures.device.ordinal);
        std::cout << "Initial segmentation: " << mesh.sourcePatchCount
                  << " source components, " << segmentation.regions.size()
                  << " CUDA regions\n" << std::flush;

        SurfaceFitOptions fitOptions;
        fitOptions.positionTolerance = command.fitTolerance;
        fitOptions.minimumPatchAreaFraction = 1.0e-8;
        fitOptions.splitMixedRegions = true;
        fitOptions.respectSourcePatchBoundaries = usefulSourcePatches;
        const auto fitStarted = std::chrono::steady_clock::now();
        std::vector<SurfacePatchFit> patches = fitSurfacePatches(
            mesh, segmentation.triangleLabels, fitOptions);
        const double fitMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - fitStarted).count();
        const std::vector<std::int32_t> fittedLabels = labelFittedPatches(mesh, patches);

        double analyticArea = 0.0;
        std::size_t acceptedAnalyticPatches = 0;
        for (const SurfacePatchFit& patch : patches) {
            if (patch.accepted && patch.kind != SurfaceKind::Unknown &&
                patch.kind != SurfaceKind::BSpline) {
                analyticArea += patch.residuals.area;
                ++acceptedAnalyticPatches;
            }
        }
        const double totalArea = meshSurfaceArea(mesh);
        const double analyticCoverage = totalArea > 0.0 ? analyticArea / totalArea : 0.0;
        std::cout << "Mesh: " << mesh.vertices.size() << " vertices, "
                  << mesh.triangles.size() << " triangles, " << mesh.stats.boundaryEdgeCount
                  << " boundary edges; fitted " << patches.size() << " patches, "
                  << acceptedAnalyticPatches << " analytic\n" << std::flush;
        {
            json fitting = {{"patches", json::array()},
                            {"warnings", loaded.warnings},
                            {"bounds_min", vectorJson(mesh.bounds.minimum)},
                            {"bounds_max", vectorJson(mesh.bounds.maximum)},
                            {"boundary_edges", mesh.stats.boundaryEdgeCount},
                            {"non_manifold_edges", mesh.stats.nonManifoldEdgeCount}};
            for (const auto& patch : patches) fitting["patches"].push_back(surfacePatchJson(patch));
            std::ofstream stream(outputDirectory / "surface_fitting.json");
            stream << std::setw(2) << fitting << '\n';
        }
        writeSegmentationPly(outputDirectory / "segmentation.ply", mesh, fittedLabels);

        const double modelScale = std::max(
            mesh.bounds.diagonal(), std::numeric_limits<double>::min());
        const double comparisonDeflection = modelScale * 2.0e-4;

        TopoDS_Shape selectedShape;
        std::string reconstructionMethod;
        std::vector<std::string> reconstructionDiagnostics;
        PatchReconstructionResult patchReconstruction;
        BidirectionalDistance distances;

        {
            const auto reconstructionStarted = std::chrono::steady_clock::now();
            PatchReconstructionOptions reconstructionOptions;
            // Let the patch builder include the tessellation chord sag when
            // sewing analytic curves to neighboring fitted surfaces.
            reconstructionOptions.sewingTolerance = 0.0;
            patchReconstruction = buildPatchReconstruction(mesh, patches, reconstructionOptions);
            const double reconstructionMilliseconds =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - reconstructionStarted).count();
            std::ostringstream timing;
            timing << "Hybrid patch reconstruction took " << reconstructionMilliseconds << " ms";
            reconstructionDiagnostics.push_back(timing.str());
            reconstructionDiagnostics.insert(reconstructionDiagnostics.end(),
                                             patchReconstruction.diagnostics.begin(),
                                             patchReconstruction.diagnostics.end());

            if (patchReconstruction.succeeded) {
                try {
                    distances = compareMeshAndBrep(
                        mesh, patchReconstruction.shape, comparisonDeflection);
                    const double maximum = std::max(
                        distances.meshToBrep.maximum, distances.brepToMesh.maximum);
                    const bool closeEnough =
                        distances.symmetricRms <= modelScale * 2.0e-3 &&
                        maximum <= modelScale * 1.0e-2;
                    if (closeEnough) {
                        selectedShape = patchReconstruction.shape;
                        reconstructionMethod = patchReconstruction.fittedNurbsFaceCount == 0U
                            ? "analytic_patches"
                            : "hybrid_analytic_nurbs";
                    } else {
                        std::ostringstream message;
                        message << "Hybrid B-Rep rejected by mesh distance: RMS="
                                << distances.symmetricRms << ", max=" << maximum;
                        reconstructionDiagnostics.push_back(message.str());
                    }
                } catch (const std::exception& error) {
                    reconstructionDiagnostics.push_back(
                        std::string("Hybrid B-Rep comparison failed: ") + error.what());
                }
            }
        }

        if (selectedShape.IsNull()) {
            for (const auto& diagnostic : reconstructionDiagnostics) {
                std::cerr << diagnostic << '\n';
            }
            throw std::runtime_error("Surface reconstruction failed; no substitute geometry exported");
        }

        const ShapeProperties properties = inspectShape(selectedShape);
        if (!properties.valid || !properties.closedSolid || properties.solidCount == 0U) {
            throw std::runtime_error("Selected reconstruction is not a valid closed solid");
        }

        const fs::path brepPath = outputDirectory / "reconstructed.brep";
        const fs::path stepPath = outputDirectory / "reconstructed.step";
        writeBrepAndStep(selectedShape, brepPath, stepPath);
        writeSegmentationPly(outputDirectory / "segmentation.ply", mesh, fittedLabels);

        // Geometry is finalized and exported before a reference is opened.
        std::optional<ReferenceValidationResult> referenceValidation;
        if (!command.referenceStep.empty()) {
            ReferenceValidationOptions referenceOptions;
            referenceOptions.referenceZUpToGltf = command.referenceZUp;
            referenceOptions.maximumSamplesPerDirection = command.referenceSamples;
            referenceValidation = validateAgainstStepReference(
                selectedShape, command.referenceStep, referenceOptions);
        }

        json root;
        root["schema_version"] = 2;
        root["input"] = {
            {"path", inputPath.string()},
            {"coordinate_scale_to_millimetres", command.inputScale},
            {"warnings", loaded.warnings}
        };
        root["environment"] = {
            {"build_configuration", buildConfiguration()},
            {"msvc_full_version", compilerVersion()},
            {"build_system", "Visual Studio 2026 / MSBuild"},
            {"cuda_compile_version", cudaVersionString(CUDART_VERSION)},
            {"cuda_runtime_version", cudaVersionString(gpuFeatures.device.runtimeVersion)},
            {"cuda_driver_version", cudaVersionString(gpuFeatures.device.driverVersion)},
            {"gpu", gpuFeatures.device.name},
            {"compute_capability", std::to_string(gpuFeatures.device.computeCapabilityMajor) +
                                   "." + std::to_string(gpuFeatures.device.computeCapabilityMinor)},
            {"occt_version", OCC_VERSION_COMPLETE}
        };
        root["mesh"] = {
            {"raw_vertices", mesh.stats.rawVertexCount},
            {"welded_vertices", mesh.stats.weldedVertexCount},
            {"input_triangles", mesh.stats.inputTriangleCount},
            {"triangles", mesh.stats.triangleCount},
            {"source_patches", mesh.sourcePatchCount},
            {"bounds_min_mm", vectorJson(mesh.bounds.minimum)},
            {"bounds_max_mm", vectorJson(mesh.bounds.maximum)},
            {"diagonal_mm", modelScale},
            {"weld_tolerance_mm", mesh.weldTolerance},
            {"components", mesh.stats.connectedComponentCount},
            {"boundary_edges", mesh.stats.boundaryEdgeCount},
            {"non_manifold_edges", mesh.stats.nonManifoldEdgeCount},
            {"watertight", mesh.stats.watertight},
            {"surface_area_mm2", totalArea},
            {"signed_volume_mm3", meshSignedVolume(mesh)}
        };
        root["cuda_segmentation"] = {
            {"requested_smooth_angle_degrees", command.smoothAngleDegrees},
            {"effective_smooth_angle_degrees", effectiveSmoothAngleDegrees},
            {"regions", segmentation.regions.size()},
            {"adjacency_pairs", segmentation.adjacencyPairCount},
            {"merged_pairs", segmentation.mergedAdjacencyPairCount},
            {"source_boundary_pairs", segmentation.sourceBoundaryPairCount},
            {"source_prior_used", usefulSourcePatches},
            {"feature_kernel_ms", gpuFeatures.timings.kernelMs},
            {"segmentation_kernel_ms", segmentation.timings.kernelMs},
            {"segmentation_total_gpu_ms", segmentation.timings.totalGpuMs},
            {"host_union_find_ms", segmentation.hostUnionFindMs}
        };
        root["surface_fitting"] = {
            {"patch_count", patches.size()},
            {"accepted_analytic_patches", acceptedAnalyticPatches},
            {"analytic_area_fraction", analyticCoverage},
            {"fit_ms", fitMilliseconds},
            {"patches", json::array()}
        };
        for (const SurfacePatchFit& patch : patches) {
            root["surface_fitting"]["patches"].push_back(surfacePatchJson(patch));
        }
        root["reconstruction"] = {
            {"method", reconstructionMethod},
            {"properties", shapePropertiesJson(properties)},
            {"mesh_to_brep", distanceJson(distances.meshToBrep)},
            {"brep_to_mesh", distanceJson(distances.brepToMesh)},
            {"symmetric_rms", distances.symmetricRms},
            {"diagnostics", reconstructionDiagnostics},
            {"hybrid", {
                {"succeeded", patchReconstruction.succeeded},
                {"analytic_faces", patchReconstruction.analyticFaceCount},
                {"fitted_nurbs_faces", patchReconstruction.fittedNurbsFaceCount},
                {"analytic_patch_failures", patchReconstruction.analyticPatchFailureCount}
            }},
            {"brep_path", brepPath.string()},
            {"step_path", stepPath.string()}
        };
        if (referenceValidation) {
            const auto& reference = *referenceValidation;
            root["reference_validation"] = {
                {"path", reference.referencePath.string()},
                {"reference_z_up_to_gltf", command.referenceZUp},
                {"sample_limit_per_direction", command.referenceSamples},
                {"reference_loaded", reference.referenceLoaded},
                {"properties_compared", reference.propertiesCompared},
                {"distances_compared", reference.distancesCompared},
                {"reference_valid", reference.reference.valid},
                {"reference_closed", reference.reference.closed},
                {"reference_faces", reference.reference.faceCount},
                {"reference_planes", reference.reference.surfaces.planes},
                {"reference_cylinders", reference.reference.surfaces.cylinders},
                {"reference_cones", reference.reference.surfaces.cones},
                {"reference_spheres", reference.reference.surfaces.spheres},
                {"reference_tori", reference.reference.surfaces.tori},
                {"topology_counts_match", reference.topology.exactMatch},
                {"volume_relative_error", reference.properties.volumeRelativeError},
                {"area_relative_error", reference.properties.surfaceAreaRelativeError},
                {"bounds_max_coordinate_error_mm",
                    reference.properties.boundsMaximumCoordinateError},
                {"symmetric_rms_mm", reference.distances.symmetricRms},
                {"symmetric_max_mm", reference.distances.symmetricMaximum},
                {"reconstructed_to_reference_max_point", vectorJson(reference.distances.reconstructedToReference.maximumPoint)},
                {"reference_to_reconstructed_max_point", vectorJson(reference.distances.referenceToReconstructed.maximumPoint)},
                {"reconstructed_max_on_boundary", reference.distances.reconstructedToReference.maximumOnBoundary},
                {"reference_max_on_boundary", reference.distances.referenceToReconstructed.maximumOnBoundary},
                {"normalized_rms", reference.distances.normalizedSymmetricRms},
                {"normalized_max", reference.distances.normalizedSymmetricMaximum},
                {"reconstructed_to_reference_samples",
                    reference.distances.reconstructedToReference.sampleCount},
                {"reference_to_reconstructed_samples",
                    reference.distances.referenceToReconstructed.sampleCount},
                {"failed_samples", reference.distances.reconstructedToReference.failedSampleCount +
                                    reference.distances.referenceToReconstructed.failedSampleCount},
                {"diagnostics", reference.diagnostics}
            };
        }

        {
            std::ofstream stream(outputDirectory / "reconstruction.json");
            if (!stream) throw std::runtime_error("Cannot write reconstruction.json");
            stream << std::setw(2) << root << '\n';
        }

        std::ostringstream report;
        report << std::setprecision(12) << std::fixed
               << "Build status: SUCCESS (" << buildConfiguration() << ")\n"
               << "Input: " << inputPath.string() << "\n"
               << "GLB-to-mm scale: " << command.inputScale << "\n"
               << "GPU: " << gpuFeatures.device.name << " (sm_"
               << gpuFeatures.device.computeCapabilityMajor
               << gpuFeatures.device.computeCapabilityMinor << ")\n"
               << "OpenCascade: " << OCC_VERSION_COMPLETE << "\n\n"
               << "Mesh vertices/triangles: " << mesh.stats.weldedVertexCount << "/"
               << mesh.stats.triangleCount << "\n"
               << "Watertight: " << (mesh.stats.watertight ? "YES" : "NO") << "\n"
               << "Source patches: " << mesh.sourcePatchCount << "\n"
               << "CUDA regions: " << segmentation.regions.size() << "\n"
               << "Fitted patches: " << patches.size() << "\n"
               << "Accepted analytic patches: " << acceptedAnalyticPatches
               << " (area coverage " << analyticCoverage * 100.0 << "%)\n"
               << "Reconstruction method: " << reconstructionMethod << "\n"
               << "Valid closed solids: " << properties.solidCount << "\n"
               << "Faces (plane/cylinder/cone/sphere/torus/B-spline/other): "
               << properties.planarFaceCount << "/" << properties.cylindricalFaceCount
               << "/" << properties.conicalFaceCount << "/" << properties.sphericalFaceCount << "/"
               << properties.toroidalFaceCount << "/" << properties.bsplineFaceCount
               << "/" << properties.otherSurfaceCount << "\n"
               << "Mesh -> B-Rep RMS/max: " << distances.meshToBrep.rms << "/"
               << distances.meshToBrep.maximum << " mm\n"
               << "B-Rep -> mesh RMS/max: " << distances.brepToMesh.rms << "/"
               << distances.brepToMesh.maximum << " mm\n"
               << "STEP: " << stepPath.string() << "\n\n"
               << "Exact feature history cannot be guaranteed from a tessellated GLB. "
                  "Analytic surfaces are accepted only when their measured residuals pass; "
                  "unresolved regions require fitted OpenCascade NURBS. "
                  "Failed fitting or solid assembly is reported as a failure.\n";
        if (referenceValidation) {
            const auto& reference = *referenceValidation;
            report << "STEP reference loaded: " << (reference.referenceLoaded ? "YES" : "NO")
                   << "\nSampled reference RMS/max: " << reference.distances.symmetricRms << "/"
                   << reference.distances.symmetricMaximum << " mm\n"
                   << "Reference volume/area relative error: "
                   << reference.properties.volumeRelativeError << "/"
                   << reference.properties.surfaceAreaRelativeError << "\n";
        }
        {
            std::ofstream stream(outputDirectory / "report.txt");
            if (!stream) throw std::runtime_error("Cannot write report.txt");
            stream << report.str();
        }
        std::cout << report.str();

        if (command.verify) {
            const double maximum = std::max(
                distances.meshToBrep.maximum, distances.brepToMesh.maximum);
            const bool acceptable = properties.valid && properties.closedSolid &&
                properties.solidCount > 0U &&
                distances.symmetricRms <= modelScale * 2.0e-3 &&
                maximum <= modelScale * 1.0e-2;
            if (!acceptable) throw std::runtime_error("Reconstruction verification failed");
            if (command.verifyAnalytic &&
                (properties.bsplineFaceCount > 0U || properties.otherSurfaceCount > 0U ||
                 reconstructionMethod != "analytic_patches")) {
                throw std::runtime_error("Analytic reconstruction verification failed");
            }
            if (referenceValidation &&
                (!referenceValidation->referenceLoaded ||
                 !referenceValidation->distancesCompared ||
                 referenceValidation->distances.reconstructedToReference.failedSampleCount > 0U ||
                 referenceValidation->distances.referenceToReconstructed.failedSampleCount > 0U)) {
                throw std::runtime_error("Reference validation did not complete");
            }
        }
        { std::ofstream status(statusPath); status << json{{"status", "completed"}} << '\n'; }
        return 0;
    } catch (const std::exception& exception) {
        if (!statusPath.empty()) {
            std::ofstream status(statusPath);
            status << json{{"status", "failed"}, {"error", exception.what()}} << '\n';
        }
        std::cerr << "mesh2brep error: " << exception.what() << '\n';
        return 1;
    }
}
