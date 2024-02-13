##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(GRAPH_SRCS
  graph/attr
  graph/graph
  graph/module
  graph/symbol
  graph/type

  graph/ops/casting
  graph/ops/complex
  graph/ops/elementwise
  graph/ops/linalg
  graph/ops/reduction
  graph/ops/slicing

  graph/_capi
  graph/_testing)

set(ENGINE_SRCS
  engine/engine
  engine/model
  engine/session
  engine/tensor
  engine/tensor_map
  engine/tensor_spec

  engine/_compilation
  engine/_context
  engine/_dtypes
  engine/_engine_impl
  engine/_model_impl
  engine/_model_specs
  engine/_status
  engine/_tensor_impl
  engine/_tensor_map_impl
  engine/_tensor_spec_impl
  engine/_utils)

# TODO: Enable priming the compilation cache.
add_mojo_package(max-mojo PRIVATE DISABLE_PRIME_CACHE
  ${GRAPH_SRCS}
  ${ENGINE_SRCS}

  DOCS_DIR
  SDK/docs
)

# Dependencies aren't automatically included in releases!
# They need to be manually added to utils/build/engine_packaging/c/CMakeLists.txt
# See the install-max-graph-c target.

add_dependencies(max-mojo mlir-mojo-api)
