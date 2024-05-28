##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set (DRIVER_SRCS
  anymemory
  device
  device_memory
  tensor
  tensor_slice
  cuda
)



add_mojo_package(max-driver PRIVATE
  ${DRIVER_SRCS}

  DEPENDS
  mof
  mojo-tensor
  MAXDriver
  max-mojo
)
