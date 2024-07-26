##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mojo_package(max-compiler PRIVATE
  __init__
  directives

  DOCS_DIR
  SDK/docs

  DEPENDS
  max-tensor-utils
  mojo-buffer
  mojo-algorithm
)
