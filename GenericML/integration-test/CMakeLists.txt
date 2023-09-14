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
  mo-opt
  modbench
  mt
  tf-opt

  DEPENDS
  tf
  mof
  )
endif()
