##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(mojo/max)
if(NOT MODULAR_PRODUCTION)
  add_subdirectory(mojo/driver)
endif()
