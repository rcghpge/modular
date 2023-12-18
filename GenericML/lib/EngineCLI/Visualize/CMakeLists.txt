##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_genericml_library(EngineCLIVisualize
  Main.cpp

  LINK_LIBS PUBLIC
  MSupport
  modular-framework-common
)
