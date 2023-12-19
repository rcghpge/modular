##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

if (MODULAR_RELEASE_PACKAGE_BUILD)
  set(EngineCLIVisualize_if_enabled)
else ()
  set(EngineCLIVisualize_if_enabled EngineCLIVisualize)
endif ()

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
  ${EngineCLIVisualize_if_enabled}
  MCrashReporting
  MSupport
  MSupportDriver
)

target_include_directories(max PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
