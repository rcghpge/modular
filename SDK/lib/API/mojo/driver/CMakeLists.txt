##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set (DRIVER_SRCS
  anytensor
  device
  device_tensor
  tensor
  tensor_slice
)



add_mojo_package(max-driver PRIVATE
  ${DRIVER_SRCS}

  DEPENDS
  mof
  mojo-tensor
  MAXDriver
  max-mojo
)
