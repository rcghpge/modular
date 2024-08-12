##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(RTTIExceptions)

add_custom_target(max-engine)

pybind11_add_module(max-core
  ${CMAKE_CURRENT_SOURCE_DIR}/core.cpp
)
target_link_libraries(max-core
  PRIVATE
  ModularFrameworkPython
)
add_dependencies(max-engine max-core)

# The extension itself must be compiled with RTTI and exceptions enabled.
set_rtti_exceptions(max-core)

# Set the relative RPATH for the `core` library.
# This assumes that in the Python wheel, shared libraries needed by `core`
# reside at modular/lib, relative to `core`'s location at max/_engine.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(relative_rpath "$ORIGIN/../lib")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(relative_rpath "@loader_path/../lib")
endif()

set_target_properties(max-core PROPERTIES
  # Ultimately we want to set the RPATH at build time since these are relative
  # paths in the package.
  # The bindings should always be loaded from the package directory structure.
  # At a future point we will set `BUILD_WITH_INSTALL_RPATH` to `TRUE`, but for
  # the time being set it to `FALSE` so that the path of each library that
  # `core` links to is baked into the `core` library itself.
  # This is to ensure stability until we have a fully contained wheel.
  OUTPUT_NAME "core"
  BUILD_WITH_INSTALL_RPATH FALSE
  BUILD_RPATH "${relative_rpath}"
  INSTALL_RPATH "${relative_rpath}"
)
