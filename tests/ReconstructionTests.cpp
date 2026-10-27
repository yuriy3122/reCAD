#include "Geometry.h"
#include "SurfaceFitting.h"
#include "PatchReconstruction.h"
#include "GeneralReconstruction.h"
#include "ReferenceValidation.h"

#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRep_Tool.hxx>
#include <STEPControl_Reader.hxx>
#include <TopExp_Explorer.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <gp_Pnt.hxx>

#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
using namespace recad;
constexpr double pi = 3.14159265358979323846;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void finish(Mesh& mesh) {
    for (const auto& p : mesh.vertices) mesh.bounds.expand(p);
    mesh.weldTolerance = mesh.bounds.diagonal() * 1.0e-9;
    computeTriangleFeatures(mesh);
    rebuildTopology(mesh);
}

std::vector<std::uint32_t> allTriangles(const Mesh& mesh) {
    std::vector<std::uint32_t> indices(mesh.triangles.size());
    std::iota(indices.begin(), indices.end(), 0U);
    return indices;
}

void testRotatedPrimitives() {
    const Vec3d axis = normalized(Vec3d{1.0, 2.0, 3.0});
    const Vec3d x = normalized(cross(axis, Vec3d{1.0, 0.0, 0.0}));
    const Vec3d y = cross(axis, x);
    const Vec3d origin{41.0, -27.0, 12.0};
    for (const bool torus : {false, true}) {
        Mesh mesh;
        constexpr unsigned count = 16;
        for (unsigned i = 0; i <= count; ++i) {
            const double u = -0.6 + 1.2 * i / count;
            for (unsigned j = 0; j <= count; ++j) {
                const double v = -0.7 + 1.4 * j / count;
                const Vec3d radial = x * std::cos(u) + y * std::sin(u);
                mesh.vertices.push_back(torus
                    ? origin + radial * (8.0 + 2.0 * std::cos(v)) + axis * (2.0 * std::sin(v))
                    : origin + radial * 3.0 + axis * (4.0 * v));
            }
        }
        for (unsigned i = 0; i < count; ++i) {
            for (unsigned j = 0; j < count; ++j) {
                const unsigned a = i * (count + 1) + j;
                const unsigned b = a + count + 1;
                mesh.triangles.push_back({{a, b, b + 1}});
                mesh.triangles.push_back({{a, b + 1, a + 1}});
            }
        }
        finish(mesh);
        const auto indices = allTriangles(mesh);
        const auto fit = fitSurfacePatch(mesh, indices);
        require(fit.accepted, "Rotated primitive must fit");
        require(fit.kind == (torus ? SurfaceKind::Torus : SurfaceKind::Cylinder),
                "Rotated primitive must select the correct family");
        require(fit.residuals.positionMax < mesh.bounds.diagonal() * 1.0e-4,
                "Rotated primitive residual exceeds accuracy target");
    }
}

struct GraphFixture {
    Mesh mesh;
    std::vector<SurfacePatchFit> patches;
};

GraphFixture graphSolid() {
    GraphFixture fixture;
    auto& mesh = fixture.mesh;
    fixture.patches.resize(6);
    constexpr unsigned count = 8;
    constexpr unsigned layer = (count + 1) * (count + 1);
    for (unsigned side = 0; side < 2; ++side) {
        for (unsigned i = 0; i <= count; ++i) {
            for (unsigned j = 0; j <= count; ++j) {
                const double x = -1.0 + 2.0 * i / count;
                const double y = -1.0 + 2.0 * j / count;
                const double z = side == 0 ? 0.0 :
                    1.0 + 0.12 * std::sin(pi * x) * std::sin(pi * y);
                mesh.vertices.push_back({x, y, z});
            }
        }
    }
    const auto add = [&](unsigned patch, Triangle triangle) {
        fixture.patches[patch].triangleIndices.push_back(
            static_cast<std::uint32_t>(mesh.triangles.size()));
        mesh.triangles.push_back(triangle);
    };
    for (unsigned i = 0; i < count; ++i) {
        for (unsigned j = 0; j < count; ++j) {
            const unsigned a = i * (count + 1) + j;
            const unsigned b = a + count + 1;
            add(0, {{a, b + 1, b}});
            add(0, {{a, a + 1, b + 1}});
            add(1, {{a + layer, b + layer, b + 1 + layer}});
            add(1, {{a + layer, b + 1 + layer, a + 1 + layer}});
        }
    }
    std::vector<unsigned> boundary;
    for (unsigned i = 0; i < count; ++i) boundary.push_back(i * (count + 1));
    for (unsigned j = 0; j < count; ++j) boundary.push_back(count * (count + 1) + j);
    for (unsigned i = count; i > 0; --i) boundary.push_back(i * (count + 1) + count);
    for (unsigned j = count; j > 0; --j) boundary.push_back(j);
    for (unsigned i = 0; i < boundary.size(); ++i) {
        const unsigned a = boundary[i];
        const unsigned b = boundary[(i + 1) % boundary.size()];
        const unsigned patch = 2 + i / count;
        add(patch, {{a, b, b + layer}});
        add(patch, {{a, b + layer, a + layer}});
    }
    finish(mesh);
    require(mesh.stats.watertight, "Graph fixture must be watertight");
    for (unsigned i = 0; i < fixture.patches.size(); ++i) {
        if (i == 1) continue; // Intentionally exercise free-form NURBS fitting.
        fixture.patches[i] = fitSurfacePatch(mesh, fixture.patches[i].triangleIndices);
        require(fixture.patches[i].kind == SurfaceKind::Plane,
                "Graph fixture sides must fit planes");
    }
    return fixture;
}

void testNurbsFittingAndFallback() {
    const auto fixture = graphSolid();
    PatchReconstructionOptions options;
    options.nurbsFitTolerance = 0.001;
    options.sewingTolerance = 0.004;
    const auto fitted = buildPatchReconstruction(fixture.mesh, fixture.patches, options);
    for (const auto& diagnostic : fitted.diagnostics) std::cout << diagnostic << '\n';
    require(fitted.fittedNurbsFaceCount == 1U,
            "Unrecognized graph patch must be fitted as one OCCT NURBS face");
    require(fitted.succeeded && fitted.closed && fitted.valid,
            "Fitted NURBS graph solid must validate and close");
    require(fitted.fallbackFaceCount == 0U, "Smooth NURBS test must avoid triangle fallback");
    const auto preserved = buildNurbsFallback(fixture.mesh);
    require(preserved.succeeded && preserved.solidCount == 1U,
            "Triangle NURBS fallback must preserve a closed graph solid");
    require(preserved.createdFaceCount == fixture.mesh.triangles.size(),
            "Triangle fallback must preserve every input triangle");
}

void testReferenceDistances() {
    const auto box = BRepPrimAPI_MakeBox(2.0, 3.0, 4.0).Shape();
    const auto moved = BRepPrimAPI_MakeBox(gp_Pnt(0.5, 0.0, 0.0), 2.0, 3.0, 4.0).Shape();
    ReferenceValidationOptions options;
    options.maximumSamplesPerDirection = 64;
    const auto identical = compareShapesForValidation(box, box, options);
    require(identical.distancesCompared && identical.distances.symmetricMaximum < 1.0e-6,
            "Identical shapes must have zero reference distance");
    const auto shifted = compareShapesForValidation(box, moved, options);
    require(shifted.distancesCompared && shifted.distances.symmetricMaximum > 0.49 &&
            shifted.distances.symmetricMaximum < 0.51,
            "Boundary validation must detect a known translation");
}
} // namespace

#ifdef RECAD_COMBINED_TESTS
int runCudaSegmentationTests();
#endif

int main(int argc, char** argv) {
    try {
#ifdef RECAD_COMBINED_TESTS
        if (runCudaSegmentationTests() != 0) return 1;
#endif
        testRotatedPrimitives();
        testNurbsFittingAndFallback();
        testReferenceDistances();
        for (int i = 1; i < argc; ++i) {
            STEPControl_Reader reader;
            require(reader.ReadFile(argv[i]) == IFSelect_RetDone,
                    "Exported STEP must be readable");
            require(reader.TransferRoots() > 0, "Exported STEP must transfer roots");
            const auto shape = reader.OneShape();
            require(!shape.IsNull() && BRepCheck_Analyzer(shape, true).IsValid(),
                    "Exported STEP must remain valid after reimport");
            TopExp_Explorer solids(shape, TopAbs_SOLID);
            require(solids.More(), "Exported STEP must contain a solid after reimport");
            for (TopExp_Explorer shells(shape, TopAbs_SHELL); shells.More(); shells.Next()) {
                require(BRep_Tool::IsClosed(shells.Current()),
                        "Exported STEP shells must remain closed after reimport");
            }
            std::cout << "STEP round-trip passed: " << argv[i] << '\n';
        }
        std::cout << "Primitive, NURBS, and reference validation tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Reconstruction test failure: " << error.what() << '\n';
        return 1;
    }
}
