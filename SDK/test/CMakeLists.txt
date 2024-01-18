##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_custom_target(check-sdk)

add_subdirectory(API/engine)
add_subdirectory(API/graph)

add_dependencies(check-all check-sdk)
