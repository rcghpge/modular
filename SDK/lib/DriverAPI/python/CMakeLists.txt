##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Build file for the local development `max.driver` Python package.

include(AddPython)

# Currently not supporting Python wheel packaging on Windows.
if(NOT UNIX)
  message(WARNING "Wheel packaging is not supported on Windows yet.")
  return()
endif()

modular_find_python3_development()

include(AddPyBind11)

add_subdirectory(lib)

# Include the build files for the MAX Driver Python bindings.
add_subdirectory(bindings/max/driver)

# Add a custom target indicating that the bindings have been built.
add_custom_target(max-driver-pybind-ext DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/max_driver_installed)

# Collect all `dlopen()`ed shared libs to correctly declare wheel dependencies.
set(MODULAR_FRAMEWORK_LIBS $<TARGET_FILE:mof>)

set(max_python_package_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/max/driver/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/max/driver/driver.py
)

# Configure and generate a pyproject.toml for local editable installation.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml.in
  ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml.in
  @ONLY)

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml
  INPUT ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml.in
)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/max_driver_installed
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/max/driver
  # Symlink all Python source files.
  # Track these manually because we need to copy/symlink max-driver-core-cpython.so to
  # `CMAKE_CURRENT_BINARY_DIR/max/driver/`, but we don't want to put it in the
  # source tree.
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/max/driver/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/max/driver/__init__.py
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/max/driver/driver.py ${CMAKE_CURRENT_BINARY_DIR}/max/driver/driver.py
  # Copy setup.py, which is needed for editable installations.
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py
  # Symlink the lib directory to the same expected relative location from
  # core-cpython.so as in the release package.
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${MODULAR_BINARY_DIR}/lib ${CMAKE_CURRENT_BINARY_DIR}/max/lib
  # Copy the `core-cpython.so` CPython extension where the editable installation can find it.
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:max-driver-core> ${CMAKE_CURRENT_BINARY_DIR}/max/driver/
  COMMAND ${Python3_EXECUTABLE} -m pip install --quiet --editable ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/max_driver_installed
  DEPENDS $<TARGET_FILE:max-driver-core> ${MODULAR_FRAMEWORK_LIBS} ${max_python_package_sources}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Symlinking `max.driver` Python source directory to $MODULAR_DERIVED_PATH and installing to $MODULAR_PYTHON."
  VERBATIM
)
