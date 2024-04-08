##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(
  EngineCLI VisualizeOptions.td
  MANUAL_NAME max-visualize
  MARKDOWN_NAME visualize
)

add_genericml_library(EngineCLIVisualize
  Main.cpp

  DEPENDS
  EngineCLIDriverOptions

  LINK_LIBS PUBLIC
  EngineCLICommon
  MSupport
  modular-framework-common
)

target_include_directories(EngineCLIVisualize PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
