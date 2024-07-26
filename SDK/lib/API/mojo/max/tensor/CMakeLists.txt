##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mojo_package(max-tensor PRIVATE
  __init__
  tensor_shape
  tensor_spec
  tensor

  DOCS_DIR
  SDK/docs

  DEPENDS
  MojoStdlib
  mojo-algorithm
)
