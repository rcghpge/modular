##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_custom_target(server_test_models
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/models ${CMAKE_CURRENT_BINARY_DIR}/models
)

add_modular_testsuite(check-max-serve-mojo
  TOOLS
  mojo

  DEPENDS
  max-mojo
  server_test_models

  LIT_ARGS
  --max-time=30
)
