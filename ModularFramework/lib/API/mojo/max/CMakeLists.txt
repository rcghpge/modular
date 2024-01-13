##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mojo_package(mobi-mojo-api PRIVATE
  graph/attr
  graph/graph
  graph/mobicc
  graph/module
  graph/symbol
  graph/testing
  graph/type

  graph/ops/complex
  graph/ops/elementwise
  graph/ops/linalg
  graph/ops/reduction
  graph/ops/shaping
  graph/ops/slicing
)
# TODO: Remove this dependency over time as only `testing.mojo` in this
# directory requires the AI Engine Mojo package.  The core `mobi-mojo-api`
# package shouldn't have a dependency on the AI Engine.
add_dependencies(mobi-mojo-api engine-mojo-api)

set(MOBI_PACKAGE_PATH "${MODULAR_BINARY_DIR}/ModularFramework/lib/API/mojo/mobi/mobi.mojopkg")

# This install rule is consumed by the install-ai-engine-mobi target defined in
# ModularFramework/lib/API/CMakeLists.txt which describes the installation
# process in more detail.
install(
  FILES ${MOBI_PACKAGE_PATH}
  DESTINATION "modular-ai-engine-tuf/pkg"
  COMPONENT ai-engine-mobi
)
