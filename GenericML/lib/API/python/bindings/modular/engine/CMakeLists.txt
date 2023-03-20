##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_custom_target(modular-engine)

pybind11_add_module(core
  ${CMAKE_CURRENT_SOURCE_DIR}/core.cpp
)
# NOTE: Currently there is an issue that global variables originating from
# LLCLRuntime get defined in multiple shared libraries and executables.
# For example, `ConcreteAsyncValue<T>::staticTypeID` is defined in all of the
# following:
# - modular-framework-common
# - tf
# - tfl
# Therefore although core really depends on modular-framework-common, it has to
# link only to tfl so that the LLCL symbols are defined only in tfl.
# Otherwise, if core links to modular-framework-common, then the global LLCL
# state get defined both in core-cpython.so and libtfl.so, causing tricky bugs.
target_link_libraries(core PRIVATE tf)
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
