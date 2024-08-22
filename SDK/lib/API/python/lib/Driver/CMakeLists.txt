##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##


include(RTTIExceptions)
add_modular_library(DriverPython STATIC
    Device.cpp
    Tensor.cpp

    LINK_LIBS PUBLIC
    DeviceDriver
    modular-framework-common
)

set_rtti_exceptions(DriverPython)
target_include_directories(DriverPython PRIVATE ${pybind11_INCLUDE_DIRS})
