##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(RTTIExceptions)

add_modular_library(ModularFrameworkPython
  Types.cpp
  TensorSpec.cpp
  TorchInputSpec.cpp
  InferenceSession.cpp
  InputProcessing.cpp
  ResultProcessing.cpp
  Utils.cpp

  LINK_LIBS PRIVATE
  modular-framework-common
  modular-framework-torch-ext
)
# Set `MOJO_CONFIG_SECTION` so that `InferenceSession` can override
# KGENCompilerRTShared.so's location.
target_compile_definitions(ModularFrameworkPython
  PRIVATE -DMOJO_CONFIG_SECTION=${MOJO_CONFIG_SECTION}
)

# The library must be compiled with RTTI and exceptions enabled.
set_rtti_exceptions(ModularFrameworkPython)

target_include_directories(ModularFrameworkPython PUBLIC
  ${MODULAR_SOURCE_DIR}/GenericML/include/GenericML/API)

target_include_directories(ModularFrameworkPython PRIVATE ${pybind11_INCLUDE_DIRS})
