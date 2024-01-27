##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(GRAPH_SRCS
  graph/attr
  graph/graph
  graph/capi
  graph/module
  graph/symbol
  graph/testing
  graph/type

  graph/ops/casting
  graph/ops/complex
  graph/ops/elementwise
  graph/ops/linalg
  graph/ops/reduction
  graph/ops/slicing)

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

add_mojo_package(max-mojo PRIVATE
  ${GRAPH_SRCS}
  ${ENGINE_SRCS}

  DOCS_DIR
  SDK/docs
)

add_dependencies(max-mojo mlir-mojo-api)

# This install rule is consumed by the install-max-graph-c target defined in
# ModularFramework/lib/API/CMakeLists.txt which describes the installation
# process in more detail.
install(
  FILES $<TARGET_PROPERTY:max-mojo,PACKAGE_LOCATION>
  DESTINATION "pkg"
  COMPONENT max-graph
)
