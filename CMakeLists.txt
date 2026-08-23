cmake_minimum_required(VERSION 3.31)

project(reCAD_mesh2brep VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

find_package(OpenCASCADE CONFIG REQUIRED COMPONENTS
  FoundationClasses
  ModelingData
  ModelingAlgorithms
  Visualization
  ApplicationFramework
  DataExchange
)
find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(mesh2brep
  src/main.cpp
  src/MeshLoader.cpp
  src/Geometry.cpp
  src/CudaGeometry.cu
  src/CadReconstruction.cpp
  src/Distance.cpp
)

set(RECAD_DOCUMENTATION_FILES
  README.md
  docs/pipeline.svg
  docs/pipeline.png
  docs/welding-topology.svg
  docs/welding-topology.png
  docs/analytic-recovery.svg
  docs/analytic-recovery.png
  docs/boolean-scoring.svg
  docs/boolean-scoring.png
)
set_source_files_properties(${RECAD_DOCUMENTATION_FILES} PROPERTIES
  HEADER_FILE_ONLY TRUE
)
target_sources(mesh2brep PRIVATE ${RECAD_DOCUMENTATION_FILES})
source_group("Documentation" FILES ${RECAD_DOCUMENTATION_FILES})

target_include_directories(mesh2brep PRIVATE
  include
  ${OpenCASCADE_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/.deps/tinygltf/include
  ${CMAKE_SOURCE_DIR}/.deps/installed/x64-windows/include
)

target_link_libraries(mesh2brep PRIVATE
  Eigen3::Eigen
  nlohmann_json::nlohmann_json
  ${OpenCASCADE_FoundationClasses_LIBRARIES}
  ${OpenCASCADE_ModelingData_LIBRARIES}
  ${OpenCASCADE_ModelingAlgorithms_LIBRARIES}
  ${OpenCASCADE_Visualization_LIBRARIES}
  ${OpenCASCADE_ApplicationFramework_LIBRARIES}
  ${OpenCASCADE_DataExchange_LIBRARIES}
)

target_compile_definitions(mesh2brep PRIVATE
  NOMINMAX
  WIN32_LEAN_AND_MEAN
  _USE_MATH_DEFINES
  RECAD_CMAKE_VERSION="${CMAKE_VERSION}"
  RECAD_PROJECT_VERSION="${PROJECT_VERSION}"
)

target_compile_options(mesh2brep PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /EHsc /Zc:__cplusplus>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr -lineinfo>
)

set_target_properties(mesh2brep PROPERTIES
  CUDA_ARCHITECTURES 120
  CUDA_RUNTIME_LIBRARY Shared
  CUDA_SEPARABLE_COMPILATION ON
  VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  VS_DEBUGGER_COMMAND_ARGUMENTS "--input ${CMAKE_SOURCE_DIR}/testData/box_sphere_test-01.glb --output ${CMAKE_SOURCE_DIR}/output"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(WIN32 AND DEFINED ENV{CUDA_PATH})
  find_file(RECAD_CUDART_DLL
    NAMES cudart64_13.dll
    PATHS "$ENV{CUDA_PATH}/bin/x64" "$ENV{CUDA_PATH}/bin"
    NO_DEFAULT_PATH
  )
  if(RECAD_CUDART_DLL)
    add_custom_command(TARGET mesh2brep POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${RECAD_CUDART_DLL}"
        "$<TARGET_FILE_DIR:mesh2brep>"
      VERBATIM
    )
  endif()
endif()

if(MSVC AND DEFINED RECAD_VCPKG_ROOT)
  add_custom_command(TARGET mesh2brep POST_BUILD
    COMMAND ${CMAKE_COMMAND}
      -DRECAD_CONFIG=$<CONFIG>
      -DRECAD_VCPKG_ROOT=${RECAD_VCPKG_ROOT}
      -DRECAD_DEST=$<TARGET_FILE_DIR:mesh2brep>
      -P ${CMAKE_SOURCE_DIR}/cmake/CopyRuntime.cmake
    VERBATIM
  )
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME reconstruct_glb
    COMMAND mesh2brep
      --input ${CMAKE_SOURCE_DIR}/testData/box_sphere_test-01.glb
      --output ${CMAKE_BINARY_DIR}/test-output
      --verify-expected-structure
  )
  set_tests_properties(reconstruct_glb PROPERTIES TIMEOUT 300)
endif()
