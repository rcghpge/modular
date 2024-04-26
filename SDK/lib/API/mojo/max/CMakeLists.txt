##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(GRAPH_SRCS
  graph/error
  graph/graph
  graph/symbol
  graph/type

  graph/ops/casting
  graph/ops/complex
  graph/ops/elementwise
  graph/ops/linalg
  graph/ops/reduction
  graph/ops/slicing

  graph/_attributes
  graph/_c
  graph/_testing)

set(ENGINE_SRCS
  engine/info
  engine/model
  engine/session
  engine/shape_element
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

set(REGISTER_SRCS
  register/register
)

set(EXTENSIBILITY_SRCS
  extensibility/__init__
)

if (MODULAR_BUILD_SERVE_RT)
  set(SERVE_SRCS
    serve/_mutt
    serve/kserve
    serve/service
  )
else ()
  set(SERVE_SRCS "")
endif ()

add_mojo_package(max-mojo PRIVATE
  ${GRAPH_SRCS}
  ${ENGINE_SRCS}
  ${EXTENSIBILITY_SRCS}
  ${REGISTER_SRCS}
  ${SERVE_SRCS}

  DOCS_DIR
  SDK/docs

  DEPENDS
  mojo-buffer
  mojo-tensor
  mojo-extensibility
)

# Dependencies aren't automatically included in releases!
# They need to be manually added to utils/build/engine_packaging/c/CMakeLists.txt
# See the install-max-graph-c target.

add_dependencies(max-mojo mlir-mojo-api)

if (MODULAR_BUILD_SERVE_RT)
  add_dependencies(max-mojo ServeRTCAPI)
endif ()
