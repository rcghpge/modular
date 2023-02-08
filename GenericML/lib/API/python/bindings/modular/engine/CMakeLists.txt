##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_custom_target(modular-engine)

pybind11_add_module(core
  ${CMAKE_CURRENT_SOURCE_DIR}/core.cpp
)
target_link_libraries(core PUBLIC modular-framework-common)
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
