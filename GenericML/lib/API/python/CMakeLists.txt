##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Currently not supporting Python wheel packaging on Windows.
if(NOT UNIX)
  message(WARNING "Wheel packaging is not supported on Windows yet.")
  return()
endif()

if(${DISABLE_ALL_EXTERNAL})
  message(WARNING "Cannot build Python C++ bindings since PyBind11 is not available.")
  return()
endif()

include(AddPyEnv)

set(MODULAR_PYTHON_VERSION "3.11.1"
  CACHE
  STRING "Python version to use for building Python bindings and wheel."
)
pyenv_install(${MODULAR_PYTHON_VERSION})

# find_package(Python3) has to be called before including pybind11 so that
# pybind11 picks up the correct Python development paths.
pyenv_setup_findpython3_variables(${MODULAR_PYTHON_VERSION})
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include(AddPyBind11)

# Install tools needed for pyproject.toml parsing and wheel repair.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pip install -r ${CMAKE_CURRENT_SOURCE_DIR}/packaging-requirements.txt
  OUTPUT_QUIET
  RESULT_VARIABLE PIP_INSTALL_DEPS_RESULT
)
if(PIP_INSTALL_DEPS_RESULT)
  message(FATAL_ERROR "Installation of Python dependencies failed unexpectedly in Python bindings configuration.")
endif()

# Use pyproject.toml as a single source of truth for the version of the Modular
# Python framework.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import tomli; f = open('${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml', 'rb'); proj = tomli.load(f)['project']; print(proj['name'].replace('-', '_') + '-' + proj['version'])"
  RESULT_VARIABLE TOMLI_RESULT
  OUTPUT_VARIABLE MODULAR_PYPROJECT_NAME_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
if(TOMLI_RESULT)
  message(FATAL_ERROR "Failed to parse pyproject.toml for Modular python bindings.")
endif()

# Package the built extension into a wheel.
set(WHEEL_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(UNPACKED_WHEEL ${CMAKE_CURRENT_BINARY_DIR}/${MODULAR_PYPROJECT_NAME_VERSION})

# Get the expected name of the built Python wheel file before building it.
# This command is expected to error, which is intentionally ignored.
execute_process(
  COMMAND ${Python3_EXECUTABLE} setup.py --dry-run --quiet bdist_wheel
  OUTPUT_VARIABLE BDIST_WHEEL_OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  ERROR_QUIET
)
# For earlier versions of wheel/setuptools, setup.py emits other informational
# messages to stdout even with the --quiet flag.
# So ignore any additional output from setup.py after the wheel name.
string(REGEX MATCH "^[^\n]+" WHEEL_BASENAME "${BDIST_WHEEL_OUTPUT}")

# Add a custom target to build and install the Python bindings and wheel.
# Individual pybind11 modules are expected to be dependencies of
# modular-engine.
add_custom_target(modular-pybind-ext ALL
  COMMAND ${Python3_EXECUTABLE} -m pip install --quiet --force-reinstall ${WHEEL_DIR}/${WHEEL_BASENAME}.whl
  DEPENDS ${WHEEL_DIR}/${WHEEL_BASENAME}.whl
  COMMENT "Installing Modular Python API wheel to CMake-managed pyenv Python."
)
add_dependencies(modular-pybind-ext modular-engine)
add_subdirectory(bindings/modular/engine)

# The wheel is built in the build directory.
# However, setuptools is unaware of the pybind module since CMake built that.
# Therefore we have to put the pybind module shared library and all other
# needed shared libraries into the wheel.
# This way everything is self-contained in the wheel and will still work after
# the wheel is pip installed.
# TODO(#7439): /dev/null is not portable to Windows.
add_custom_command(OUTPUT ${WHEEL_DIR}/${WHEEL_BASENAME}_packed.whl
  COMMAND ${Python3_EXECUTABLE} -m build --wheel --outdir ${WHEEL_DIR} > /dev/null
  COMMAND ${Python3_EXECUTABLE} -m wheel unpack ${WHEEL_DIR}/${WHEEL_BASENAME}.whl --dest ${WHEEL_DIR} > /dev/null
  COMMAND ${CMAKE_COMMAND} -E make_directory ${UNPACKED_WHEEL}/modular/engine
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:core> ${UNPACKED_WHEEL}/modular/engine/
  COMMAND ${CMAKE_COMMAND} -E make_directory ${UNPACKED_WHEEL}/lib
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tfl> ${UNPACKED_WHEEL}/lib
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:KGENCompilerRT> ${UNPACKED_WHEEL}/lib
  COMMAND ${Python3_EXECUTABLE} -m wheel pack ${UNPACKED_WHEEL} --dest ${WHEEL_DIR} > /dev/null
  COMMAND ${CMAKE_COMMAND} -E rename ${WHEEL_DIR}/${WHEEL_BASENAME}.whl ${WHEEL_DIR}/${WHEEL_BASENAME}_packed.whl
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/build
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/modular_engine.egg-info
  BYPRODUCTS ${UNPACKED_WHEEL}
  DEPENDS $<TARGET_FILE:core>
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packing wheel with bindings and shared library dependencies."
  VERBATIM
)

if(APPLE)
  set(REPAIR_CMD ${Python3_EXECUTABLE} -m delocate.cmd.delocate_wheel)
else()
  set(REPAIR_CMD ${Python3_EXECUTABLE} -m auditwheel.main repair --plat linux_x86_64)
endif()

# Detect non-system shared library dependencies and copy them into the wheel,
# updating the RECORDS metadata file.
add_custom_command(OUTPUT ${WHEEL_DIR}/${WHEEL_BASENAME}.whl
  COMMAND ${REPAIR_CMD} ${WHEEL_DIR}/${WHEEL_BASENAME}_packed.whl
  COMMAND ${CMAKE_COMMAND} -E copy ${WHEEL_DIR}/${WHEEL_BASENAME}_packed.whl ${WHEEL_DIR}/${WHEEL_BASENAME}.whl
  DEPENDS ${WHEEL_DIR}/${WHEEL_BASENAME}_packed.whl
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Repairing Modular Python API wheel."
  VERBATIM
)
