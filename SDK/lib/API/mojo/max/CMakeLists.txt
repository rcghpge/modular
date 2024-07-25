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
  graph/ops/quantized_ops
  graph/ops/reduction
  graph/ops/slicing
  graph/ops/repeat_interleave

  graph/quantization/__init__
  graph/quantization/encodings
  graph/quantization/quantization_encoding

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
  engine/value

  engine/_compilation
  engine/_context
  engine/_dtypes
  engine/_engine_impl
  engine/_model_impl
  engine/_model_specs
  engine/_status
  engine/_tensor_impl
  engine/_tensor_map_impl
  engine/_tensor_spec_impl)

set(REGISTER_SRCS
  register/register
)

set(EXTENSIBILITY_SRCS
  extensibility/__init__
)

set(SERVE_SRCS
  serve/__init__
  serve/_batch
  serve/batch
  serve/http/__init__
  serve/http/service
  serve/http/_server
  serve/http/server
  serve/kserve/_client
  serve/kserve/client
  serve/kserve/_server
  serve/kserve/server
  serve/kserve/_types
  serve/kserve/types
  serve/kserve/mux
  serve/kserve/service
  serve/util/__init__
  serve/util/callbacks
  serve/util/config
  serve/util/debug
  serve/util/stats
  serve/metrics/__init__
  serve/metrics/_telemetry
)

set(DRIVER_SRCS
  _driver/__init__
  _driver/_driver_library
  _driver/_status
  _driver/anytensor
  _driver/cuda
  _driver/device
  _driver/device_memory
  _driver/graph
  _driver/tensor
  _driver/tensor_slice
  _driver/utils
)

set(UTILS_SRCS
  _utils/__init__
)

set(COMPILER_SRCS _compiler/__init__)

set(TENSOR_SRCS tensor/__init__)

set(TENSOR_UTILS_SRCS tensor_utils/__init__)


add_mojo_package(max-mojo PRIVATE
  ${GRAPH_SRCS}
  ${ENGINE_SRCS}
  ${EXTENSIBILITY_SRCS}
  ${REGISTER_SRCS}
  ${SERVE_SRCS}
  ${DRIVER_SRCS}
  ${UTILS_SRCS}
  ${COMPILER_SRCS}
  ${TENSOR_SRCS}
  ${TENSOR_UTILS_SRCS}

  DOCS_DIR
  SDK/docs

  DEPENDS
  mof
  mojo-algorithm
  mojo-buffer
  mojo-extensibility
  DeviceDriver

  max-compiler
  max-tensor-utils
  max-tensor

  mlir-mojo-api
)

# FIXME: The documentation for modules located outside of max/ must be copied
# into the MAX_DOCS_DIR to make it look like they are submodules. This is a hack
# until proper documentation support is figured out.

# Primer:
# 1. A Mojo target that has documentation will have its name appended with -doc.
# 2. Mojo targets that start with an underscore will not have documentation.
# 3. A Mojo target's documentation consists of nested directories, all the way down
#    to mojo entities (structs and functions).
if(UNIX)
  # TARGET_NAME: Name of the mojo target (without the -doc).
  # TARGET_NAME_IN_DOCS: Name it should have in the max-mojo-doc.
  function(import_external_doc_into_max TARGET_NAME TARGET_NAME_IN_DOCS)
    # Handle the differences in sed behavior between mac and linux.
    if (APPLE)
      set(SED_CMD_PREFIX "\'\'")
    else ()
      set(SED_CMD_PREFIX "")
    endif()

    set(TARGET_DOC ${TARGET_NAME}-doc)
    add_dependencies(max-mojo-doc ${TARGET_DOC})
    get_target_property(MAX_DOCS_DIR max-mojo-doc OUTPUT_DIR)
    get_target_property(TARGET_DOCS_DIR ${TARGET_DOC} OUTPUT_DIR)
    add_custom_command(TARGET max-mojo-doc
     POST_BUILD
     COMMAND ${CMAKE_COMMAND} -E remove_directory ${MAX_DOCS_DIR}/${TARGET_NAME_IN_DOCS}
     COMMAND ${CMAKE_COMMAND} -E copy_directory
     ${TARGET_DOCS_DIR} ${MAX_DOCS_DIR}/${TARGET_NAME_IN_DOCS}
     COMMAND sed -i ${SED_CMD_PREFIX} 's|type: module|type: package|g' ${MAX_DOCS_DIR}/${DISPLAY_NAME}/index.md
    )
  endfunction()

  import_external_doc_into_max(max-tensor tensor)
  import_external_doc_into_max(max-tensor-utils tensor_utils)
  import_external_doc_into_max(mojo-extensibility extensibility)
endif()

# Dependencies aren't automatically included in releases!
# They need to be manually added to utils/build/engine_packaging/c/CMakeLists.txt
# See the install-max-graph-c target.
add_dependencies(max-mojo ServeRTCAPI)
