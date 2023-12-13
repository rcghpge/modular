##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(RTTIExceptions)

add_modular_library(ModularFrameworkPython
  Types.cpp
  TensorSpec.cpp
  InferenceSession.cpp
  InputProcessing.cpp
  ResultProcessing.cpp
)

# The library must be compiled with RTTI and exceptions enabled.
set_rtti_exceptions(ModularFrameworkPython)

target_include_directories(ModularFrameworkPython PUBLIC
  ${MODULAR_SOURCE_DIR}/GenericML/include/GenericML/API)

target_include_directories(ModularFrameworkPython PRIVATE ${pybind11_INCLUDE_DIRS})
