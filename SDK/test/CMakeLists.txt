##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_custom_target(check-sdk)

add_subdirectory(API/engine)
add_subdirectory(API/graph)
add_subdirectory(API/driver)
add_subdirectory(API/serve)

add_dependencies(check-all check-sdk)
