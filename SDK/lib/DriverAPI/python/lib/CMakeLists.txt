##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##


add_modular_library(DriverPython STATIC
    Device.cpp

    LINK_LIBS PUBLIC
    DeviceDriver
)
