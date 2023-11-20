##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_genericml_library(VisualizeCLI
  Main.cpp

  LINK_LIBS PUBLIC
  MSupport
  modular-framework-common
)
