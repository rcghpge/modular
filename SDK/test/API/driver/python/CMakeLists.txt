##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_pytest(check-max-driver
  TOP_LEVEL_CHECK
  check-max-driver-api

  EXTRA_DIRS_FOR_ASAN
  $<TARGET_FILE_DIR:mof>

  DEPENDS
  max-driver-pybind-ext
)
