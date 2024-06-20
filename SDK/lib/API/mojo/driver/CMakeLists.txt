##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set (DRIVER_SRCS
  anytensor
  device
  device_memory
  tensor
  tensor_slice
  cuda
  graph
)



add_mojo_package(max-driver PRIVATE
  ${DRIVER_SRCS}

  DEPENDS
  mof
  mojo-tensor
  DeviceDriver
  max-mojo
)
