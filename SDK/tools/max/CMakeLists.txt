##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(
  EngineCLI DriverOptions.td
  MANUAL_NAME max
  MARKDOWN_NAME index
)

add_modular_tool(max
  max.cpp

  DEPENDS
  EngineCLIDriverOptions

  LINK_LIBS PRIVATE
  EngineCLIBenchmark
  EngineCLIVisualize
  MCrashReporting
  MLLCLInit
  MSupport
  MSupportDriver
  MSupportGlobals
  LLCLRuntimeGlobals
)

target_include_directories(max PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

# Copy all generated Markdown files to a convenient location.
collect_driver_markdown(EngineCLIDriverOptions)
