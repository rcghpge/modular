##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_genericml_library(EngineCLICommon
  APIBridge.cpp
  InputGenerationSchema.cpp
  TorchSupport.cpp

  LINK_LIBS
  PUBLIC
  MSupport
  modular-framework-common

  PRIVATE
  MSupportML
)

# Can't go into LINK_LIBS above because of a layering violation.
# add_genericml_library checks that all targets listed in LINK_LIBS exist.  But
# at the time we're running this, we're in GenericML, and PyTorch's CMakeLists
# have not run yet, so this target doesn't exist.  target_link_libraries is a
# little more lax -- it resolves target names at generation time, after all
# CMakeLists have run.  modular-framework-torch-ext should move back up into
# LINK_LIBS once the MAX CLI is moved out of GenericML.
target_link_libraries(EngineCLICommon PRIVATE modular-framework-torch-ext)
