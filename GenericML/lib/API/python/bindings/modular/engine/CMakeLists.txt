##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_custom_target(modular-engine)

pybind11_add_module(core
  ${CMAKE_CURRENT_SOURCE_DIR}/core.cpp
)
target_link_libraries(core PUBLIC modular-framework-common modular-framework-tf-ext modular-framework-torch-ext)
add_dependencies(modular-engine core)

# The extension itself must be compiled with RTTI and exceptions enabled.
target_compile_options(core PRIVATE
 $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
   # Enable RTTI and exceptions.
   -frtti -fexceptions
 >
 $<$<CXX_COMPILER_ID:MSVC>:
   # Enable RTTI and exceptions.
   /EHsc /GR>
)

# Set the relative RPATH for the `core` library.
# This assumes that in the Python wheel, shared libraries needed by `core`
# reside at modular/lib, relative to `core`'s location at modular/engine.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(relative_rpath "$ORIGIN/../lib")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(relative_rpath "@loader_path/../lib")
endif()

set_target_properties(core PROPERTIES
  # Ultimately we want to set the RPATH at build time since these are relative
  # paths in the package.
  # The bindings should always be loaded from the package directory structure.
  # At a future point we will set `BUILD_WITH_INSTALL_RPATH` to `TRUE`, but for
  # the time being set it to `FALSE` so that the path of each library that
  # `core` links to is baked into the `core` library itself.
  # This is to ensure stability until we have a fully contained wheel.
  BUILD_WITH_INSTALL_RPATH FALSE
  BUILD_RPATH "${relative_rpath}"
  INSTALL_RPATH "${relative_rpath}"
)
