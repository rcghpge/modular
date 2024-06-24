##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(mojo/max_utils)
if(NOT MODULAR_PRODUCTION)
  add_subdirectory(mojo/driver)
endif()
add_subdirectory(mojo/max)
