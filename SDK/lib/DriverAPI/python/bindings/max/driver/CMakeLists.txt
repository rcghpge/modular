##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(RTTIExceptions)

add_custom_target(max-driver)

pybind11_add_module(max-driver-core
  ${CMAKE_CURRENT_SOURCE_DIR}/core.cpp
)

add_dependencies(max-driver max-driver-core)

target_link_libraries(max-driver-core PRIVATE DeviceDriver)

# The extension itself must be compiled with RTTI and exceptions enabled.
set_rtti_exceptions(max-driver-core)

# Set the relative RPATH for the `max-driver-core` library.
# This assumes that in the Python wheel, shared libraries needed by
# `max-driver-core` reside at modular/lib, relative to `max-driver-core`'s
# location at max/driver.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(relative_rpath "$ORIGIN/../lib")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(relative_rpath "@loader_path/../lib")
endif()

set_target_properties(max-driver-core PROPERTIES
  # Ultimately we want to set the RPATH at build time since these are relative
  # paths in the package.
  # The bindings should always be loaded from the package directory structure.
  # At a future point we will set `BUILD_WITH_INSTALL_RPATH` to `TRUE`, but for
  # the time being set it to `FALSE` so that the path of each library that
  # `driver_core` links to is baked into the `max-driver-core` library
  # itself.
  # This is to ensure stability until we have a fully contained wheel.
  OUTPUT_NAME "driver_core"
  BUILD_WITH_INSTALL_RPATH FALSE
  BUILD_RPATH "${relative_rpath}"
  INSTALL_RPATH "${relative_rpath}"
)
