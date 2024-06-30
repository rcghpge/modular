##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_pytest(max-graph
  TOP_LEVEL_CHECK
  check-sdk

  EXTRA_DIRS_FOR_ASAN
  $<TARGET_FILE_DIR:mof>

  DEPENDS
  max-graph-pybind-ext
  MAXGraphMlirPythonModules
)
