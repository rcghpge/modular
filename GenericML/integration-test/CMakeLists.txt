##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##
#
# Configure the GenericML integration testsuite.
#
##===----------------------------------------------------------------------===##

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

if (NOT BENCHMARKING)
add_modular_testsuite(check-genericml-integration
  TOOLS
  FileCheck
  modbench
  mt
  tf-opt
  is-cuda-available
  mo-cuda-opt
  hello_cuda_async_events

  DEPENDS
  tf
  )
endif()
