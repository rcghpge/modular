##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddPython)

# Currently not supporting Python wheel packaging on Windows.
if(NOT UNIX)
  message(WARNING "Wheel packaging is not supported on Windows yet.")
  return()
endif()

modular_find_python3_development()

include(AddPyBind11)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml.in
  ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml.in
  @ONLY)

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml
  INPUT ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml.in
)


# Set up wheel directory structure by copying the pyproject.toml and setup.py
# files needed for setuptools to build the wheel.
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml ${CMAKE_CURRENT_SOURCE_DIR}/setup.py
  DESTINATION  max-engine-py-binding
  COMPONENT ai-engine-py-binding
)

# Also copy the `max.engine` Python files into the wheel.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../max
  DESTINATION max-engine-py-binding
  COMPONENT ai-engine-py-binding
  FILES_MATCHING
  PATTERN "*.py*"
  PATTERN "py.typed"
)

# Finally, install the built Python extension module shared library into the
# wheel.
install(TARGETS max-core
  DESTINATION  max-engine-py-binding/max/engine
  COMPONENT ai-engine-py-binding
)

add_llvm_install_targets(
  install-ai-engine-py-binding

  DEPENDS
  max-engine

  COMPONENT
  ai-engine-py-binding
)
