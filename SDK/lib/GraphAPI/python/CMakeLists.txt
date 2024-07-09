##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Build file for the local development `max.graph` Python package.

include(AddPython)
include(AddMLIRPythonIdempotent)

# Currently not supporting Python wheel packaging on Windows.
if(NOT UNIX)
  message(WARNING "Wheel packaging is not supported on Windows yet.")
  return()
endif()

modular_find_python3_development()

include(AddPyBind11)

# Include the build files for the MAX Graph Python bindings.
add_subdirectory(bindings/max/graph)

# Add a custom target indicating that the bindings have been built.
add_custom_target(max-graph-pybind-ext DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/max_graph_installed)

# Collect all `dlopen()`ed shared libs to correctly declare wheel dependencies.
set(MODULAR_FRAMEWORK_LIBS $<TARGET_FILE:mof>)

set(max_python_package_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/max/graph/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/max/graph/graph.py
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

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/max_graph_installed
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/max/graph
  # Symlink all Python source files.
  # Track these manually because we need to copy/symlink max-graph-core-cpython.so to
  # `CMAKE_CURRENT_BINARY_DIR/max/graph/`, but we don't want to put it in the
  # source tree.
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/max/graph/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/max/graph/__init__.py
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/max/graph/mlir/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/max/graph/mlir/__init__.py
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/max/graph/graph.py ${CMAKE_CURRENT_BINARY_DIR}/max/graph/graph.py
  # Copy setup.py, which is needed for editable installations.
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py
  # Symlink the lib directory to the same expected relative location from
  # core-cpython.so as in the release package.
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${MODULAR_BINARY_DIR}/lib ${CMAKE_CURRENT_BINARY_DIR}/max/lib
  # Copy the `core-cpython.so` CPython extension where the editable installation can find it.
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:max-graph-core> ${CMAKE_CURRENT_BINARY_DIR}/max/graph/
  COMMAND ${Python3_EXECUTABLE} -m pip install --quiet --editable ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/max_graph_installed
  DEPENDS $<TARGET_FILE:max-graph-core> ${MODULAR_FRAMEWORK_LIBS} ${max_python_package_sources}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Symlinking `max.graph` Python source directory to $MODULAR_DERIVED_PATH and installing to $MODULAR_PYTHON."
  VERBATIM
)

# The following is loosely based on upstream's standalone MLIR Python
# [example](https://github.com/llvm/llvm-project/blob/e9b8cd0c806db00f0981fb36717077c941426302/mlir/examples/standalone/python/CMakeLists.txt).

################################################################################
# Instantiation of all Python modules
################################################################################

add_flag_if_supported("-Wno-address" SDK_GRAPHAPI_PYTHON_NO_ADDRESS)

add_mlir_python_modules_idempotent(MAXGraphMlirPythonModules
  ROOT_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/max/graph/mlir"
  INSTALL_PREFIX ""
  DECLARED_SOURCES
    MLIRPythonSources.Core
  COMMON_CAPI_LINK_LIBS
    mof
)
