##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mojo_package(max-compiler PRIVATE
  __init__

  DOCS_DIR
  SDK/docs

  DEPENDS
  max-tensor-utils
  extensibility-compiler-internal
  mojo-buffer
  mojo-algorithm
)
