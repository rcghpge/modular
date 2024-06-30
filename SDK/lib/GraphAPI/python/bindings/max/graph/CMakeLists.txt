##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddMLIRPython)
include(RTTIExceptions)

add_custom_target(max-graph)

pybind11_add_module(max-graph-core
  ${CMAKE_CURRENT_SOURCE_DIR}/core.cpp
)
add_dependencies(max-graph max-graph-core)

# Link dep and include directories for MAX C API functions such as M_version().
target_link_libraries(max-graph-core PRIVATE modular-framework-common)
target_include_directories(max-graph-core PUBLIC
  ${MODULAR_SOURCE_DIR}/GenericML/include/GenericML/API)

# The extension itself must be compiled with RTTI and exceptions enabled.
set_rtti_exceptions(max-graph-core)

# Set the relative RPATH for the `max-graph-core` library.
# This assumes that in the Python wheel, shared libraries needed by
# `max-graph-core` reside at modular/lib, relative to `max-graph-core`'s
# location at max/graph.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(relative_rpath "$ORIGIN/../lib")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(relative_rpath "@loader_path/../lib")
endif()

set_target_properties(max-graph-core PROPERTIES
  # Ultimately we want to set the RPATH at build time since these are relative
  # paths in the package.
  # The bindings should always be loaded from the package directory structure.
  # At a future point we will set `BUILD_WITH_INSTALL_RPATH` to `TRUE`, but for
  # the time being set it to `FALSE` so that the path of each library that
  # `max-graph-core` links to is baked into the `max-graph-core` library
  # itself.
  # This is to ensure stability until we have a fully contained wheel.
  OUTPUT_NAME "max-graph-core"
  BUILD_WITH_INSTALL_RPATH FALSE
  BUILD_RPATH "${relative_rpath}"
  INSTALL_RPATH "${relative_rpath}"
)
