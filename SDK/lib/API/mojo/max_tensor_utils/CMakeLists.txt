##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mojo_package(max-tensor-utils PRIVATE
  __init__
  indexing
  tensor_like
  unsafe_tensor_slice

  DOCS_DIR
  SDK/docs

  DEPENDS
  mojo-buffer
  mojo-algorithm
  max-tensor
)
