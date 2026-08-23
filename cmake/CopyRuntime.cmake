if(RECAD_CONFIG STREQUAL "Debug")
  set(_runtime_dir "${RECAD_VCPKG_ROOT}/debug/bin")
else()
  set(_runtime_dir "${RECAD_VCPKG_ROOT}/bin")
endif()

if(EXISTS "${_runtime_dir}")
  file(GLOB _runtime_dlls "${_runtime_dir}/*.dll")
  foreach(_dll IN LISTS _runtime_dlls)
    file(COPY "${_dll}" DESTINATION "${RECAD_DEST}")
  endforeach()
endif()

