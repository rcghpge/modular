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

  graph/checkpoint/save_load
  graph/checkpoint/tensor_dict

  graph/ops/casting
  graph/ops/convolution
  graph/ops/complex
  graph/ops/elementwise
  graph/ops/linalg
  graph/ops/reduction
  graph/ops/slicing

  graph/quantization/__init__
  graph/quantization/quantization_encoding
  graph/quantization/bfloat16_encoding
  graph/quantization/q4_0_encoding

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

set(SERVE_SRCS
  serve/kserve/_kserve_impl
  serve/kserve/_serve_rt
  serve/kserve/callbacks
  serve/kserve/client
  serve/kserve/config
  serve/kserve/debug
  serve/kserve/server
  serve/kserve/stats
  serve/batch
  serve/service
)

add_mojo_package(max-mojo PRIVATE
  ${GRAPH_SRCS}
  ${ENGINE_SRCS}
  ${EXTENSIBILITY_SRCS}
  ${REGISTER_SRCS}
  ${SERVE_SRCS}

  DOCS_DIR
  SDK/docs

  DEPENDS
  mof
  mojo-tensor
  mojo-buffer
  mojo-extensibility

  mlir-mojo-api
)

# FIXME: The documentation for extensibility is currently defined in
# mojo-extensibility, make sure to copy this over after building the normal max
# docs. This is a temporary measure until we properly figure out the
# documentation generation and package layering.
if(UNIX)
  add_dependencies(max-mojo-doc mojo-extensibility-doc)
  get_target_property(MAX_DOCS_DIR max-mojo-doc OUTPUT_DIR)
  get_target_property(EXTENSIBILITY_DOCS_DIR mojo-extensibility-doc OUTPUT_DIR)

  # Handle the differences in sed behavior between mac and linux.
  if (APPLE)
    add_custom_command(TARGET max-mojo-doc
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E remove_directory ${MAX_DOCS_DIR}/extensibility
      COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${EXTENSIBILITY_DOCS_DIR}/tensor ${MAX_DOCS_DIR}/extensibility
      COMMAND sed -i '' 's|title: tensor|title: extensibility|g' ${MAX_DOCS_DIR}/extensibility/index.md
      COMMAND sed -i '' 's|type: module|type: package|g' ${MAX_DOCS_DIR}/extensibility/index.md
    )
  else ()
    add_custom_command(TARGET max-mojo-doc
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E remove_directory ${MAX_DOCS_DIR}/extensibility
      COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${EXTENSIBILITY_DOCS_DIR}/tensor ${MAX_DOCS_DIR}/extensibility
      COMMAND sed -i 's|title: tensor|title: extensibility|g' ${MAX_DOCS_DIR}/extensibility/index.md
      COMMAND sed -i 's|type: module|type: package|g' ${MAX_DOCS_DIR}/extensibility/index.md
    )
  endif()
endif()

# Dependencies aren't automatically included in releases!
# They need to be manually added to utils/build/engine_packaging/c/CMakeLists.txt
# See the install-max-graph-c target.

add_dependencies(max-mojo ServeRTCAPI)
