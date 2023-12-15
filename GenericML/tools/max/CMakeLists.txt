##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

if (MODULAR_RELEASE_PACKAGE_BUILD)
  set(VisualizeCLI_if_enabled)
else ()
  set(VisualizeCLI_if_enabled VisualizeCLI)
endif ()

add_modular_tool(max
  max.cpp

  LINK_LIBS PRIVATE
  BenchmarkCLI
  MCrashReporting
  MSupport
  ${VisualizeCLI_if_enabled}
)
