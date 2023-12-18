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

add_modular_tool(max
  max.cpp

  LINK_LIBS PRIVATE
  EngineCLIBenchmark
  ${EngineCLIVisualize_if_enabled}
  MCrashReporting
  MSupport
)
