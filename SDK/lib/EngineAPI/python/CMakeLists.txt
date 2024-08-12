##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Build file for the local development `max.engine` Python package, and for
# installation of files needed for release wheel packaging.

# This creates an editable installation of the `max.engine` without building
# any wheels.
# The general approach is to create symlinks from all of the needed Python
# source files under `CMAKE_CURRENT_SOURCE_DIR`, to the build directory under
# `CMAKE_CURRENT_BINARY_DIR`.
# This approach achieves a few goals:
# - Updates to the local Python source files are immediately reflected in the
#   installed `max.engine` package.
# - There is no need to build a wheel for local development.
# - We can re-create the expected installed layout of engine libs and Python
#   source, which would be installed under site-packages for the release
#   package.
#   This is key to enable the Python bindings to find
#   libKGENCompilerRTShared.so using the same method in development and
#   release.

include(AddPython)

# Currently not supporting Python wheel packaging on Windows.
if(NOT UNIX)
  message(WARNING "Wheel packaging is not supported on Windows yet.")
  return()
endif()

modular_find_python3_development()

include(AddPyBind11)

add_subdirectory(lib)

# MODULAR_EXCLUDE_PYBIND_EXT_FROM_ALL can be set to avoid including
# max-pybind-ext in the 'all' target.  Since building the Python extension
# currently requires Internet access, this is useful if you are developing
# without Internet access (e.g. on an airplane).  It also speeds up builds by
# several seconds, and can be useful even when Internet access is present, as
# long as the wheel is not needed.
if(MODULAR_EXCLUDE_PYBIND_EXT_FROM_ALL)
  set(ext_all)
else()
  set(ext_all ALL)
endif()
add_custom_target(max-pybind-ext ${ext_all} DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/max_engine_installed)

add_dependencies(max-pybind-ext max-engine)
add_subdirectory(bindings/max/_engine)

# Needs to be added after cmake knows about `core` defined in
# bindings/max/engine.
add_subdirectory(tuf_packaging)

# Collect all `dlopen()`ed shared libs to correctly declare wheel dependencies.
set(MODULAR_FRAMEWORK_LIBS $<TARGET_FILE:monnx>)

if (STOCK_PYTORCH_FOUND)
  list(APPEND MODULAR_FRAMEWORK_LIBS $<TARGET_FILE:mtorch>)
endif()

set(max_python_package_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/max/engine/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/max/engine/api.py
  ${CMAKE_CURRENT_SOURCE_DIR}/bindings/max/_engine/core.pyi
)

# Note that ${CMAKE_CURRENT_BINARY_DIR} == ${WHEEL_DIR} so the
# configured/generated pyproject.toml works for local wheel building for target
# max-pybind-ext as well.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml.in
  ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml.in
  @ONLY)

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml
  INPUT ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml.in
)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/max_engine_installed
  COMMAND ${CMAKE_COMMAND} -E make_directory
    ${CMAKE_CURRENT_BINARY_DIR}/max/engine
    ${CMAKE_CURRENT_BINARY_DIR}/max/_engine
  # Symlink all Python source files.
  # Track these manually because we need to copy/symlink core-cpython.so to
  # `CMAKE_CURRENT_BINARY_DIR/max/engine/`, but we don't want to put it in the
  # source tree.
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/max/engine/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/max/engine/__init__.py
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/max/engine/api.py ${CMAKE_CURRENT_BINARY_DIR}/max/engine/api.py
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/bindings/max/_engine/core.pyi ${CMAKE_CURRENT_BINARY_DIR}/max/_engine/core.pyi
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/bindings/max/_engine/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/max/_engine/__init__.py
  # Copy setup.py, which is needed for editable installations.
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py
  # Symlink the lib directory to the same expected relative location from
  # core-cpython.so as in the release package.
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${MODULAR_BINARY_DIR}/lib ${CMAKE_CURRENT_BINARY_DIR}/max/lib
  # Copy the `core-cpython.so` CPython extension where the editable installation can find it.
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:max-core> ${CMAKE_CURRENT_BINARY_DIR}/max/_engine/
  COMMAND ${Python3_EXECUTABLE} -m pip install --quiet --editable ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/max_engine_installed
  DEPENDS $<TARGET_FILE:max-core> ${MODULAR_FRAMEWORK_LIBS} ${max_python_package_sources}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Symlinking `max.engine` Python source directory to $MODULAR_DERIVED_PATH and installing to $MODULAR_PYTHON."
  VERBATIM
)

add_subdirectory(docs)
add_subdirectory(notebooks)

# Set up wheel directory structure by copying the pyproject.toml and setup.py
# files needed for setuptools to build the wheel.
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml ${CMAKE_CURRENT_SOURCE_DIR}/setup.py
  DESTINATION max-engine-wheel
  COMPONENT ai-engine-wheel
)

# Also copy the `max.engine` Python files into the wheel.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/max
  DESTINATION max-engine-wheel
  COMPONENT ai-engine-wheel
  FILES_MATCHING
  PATTERN "*.py*"
  PATTERN "py.typed"
)

# Finally, install the built Python extension module shared library into the
# wheel.
install(TARGETS max-core
  DESTINATION max-engine-wheel/max/engine
  COMPONENT ai-engine-wheel
)
