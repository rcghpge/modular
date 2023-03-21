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

# TODO: After #8927 is resolved and we no longer need to statically link to tf, we can remove this
if(NOT STOCK_TENSORFLOW_FOUND)
  message(WARNING "Python bindings require Stock Tensorflow to build.")
  return()
endif()

if(${DISABLE_ALL_EXTERNAL})
  message(WARNING "Cannot build Python C++ bindings since PyBind11 is not available.")
  return()
endif()

include(AddPyEnv)

set(MODULAR_PYTHON_API_PYTHON_VERSION "3.11.1"
  CACHE
  STRING "Python version to use for building Python bindings and wheel."
)
set(MODULAR_PYTHON_API_PYTHON_ROOT ""
  CACHE
  STRING "Path to Python installation to use for building Python bindings and wheel. Overrides pyenv Python."
)
message(STATUS "Python version for Python API build: ${MODULAR_PYTHON_API_PYTHON_VERSION}. "
  "Overridden if MODULAR_PYTHON_API_PYTHON_ROOT set: ${MODULAR_PYTHON_API_PYTHON_ROOT}."
)
pyenv_install("${MODULAR_PYTHON_API_PYTHON_VERSION}" "${MODULAR_PYTHON_API_PYTHON_ROOT}")

# find_package(Python3) has to be called before including pybind11 so that
# pybind11 picks up the correct Python development paths.
pyenv_setup_findpython3_variables("${MODULAR_PYTHON_API_PYTHON_VERSION}" "${MODULAR_PYTHON_API_PYTHON_ROOT}")
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

set(MODULAR_FRAMEWORK_LIBS $<TARGET_FILE:monnx> $<TARGET_FILE:tfl>)

# The wheel is built in the build directory.
# However, setuptools is unaware of the pybind module since CMake built that.
# Therefore we have to put the pybind module shared library and all other
# needed shared libraries into the wheel.
# This way everything is self-contained in the wheel and will still work after
# the wheel is pip installed.
# TODO(#7439): /dev/null is not portable to Windows.
if(STOCK_TENSORFLOW_FOUND)
  set(EXTERNAL_TF_EXES
      "${MODULAR_DERIVED_DIR}/third-party/bin/external-tf-translate;"
      "${MODULAR_DERIVED_DIR}/third-party/bin/external-tf-opt")
  list(APPEND MODULAR_FRAMEWORK_LIBS $<TARGET_FILE:tf>)
endif()

# When skipping wheel repair, output the final wheel and name it accordingly.
if (MODULAR_PYTHON_API_REPAIR_WHEEL)
  set(MODULAR_PACKED_WHEEL_NAME ${WHEEL_DIR}/${WHEEL_BASENAME}_packed.whl)
else()
  set(MODULAR_PACKED_WHEEL_NAME ${WHEEL_DIR}/${WHEEL_BASENAME}.whl)
endif()
add_custom_command(OUTPUT ${MODULAR_PACKED_WHEEL_NAME}
  COMMAND ${Python3_EXECUTABLE} -m build --wheel --outdir ${WHEEL_DIR} > /dev/null
  COMMAND ${Python3_EXECUTABLE} -m wheel unpack ${WHEEL_DIR}/${WHEEL_BASENAME}.whl --dest ${WHEEL_DIR} > /dev/null
  COMMAND ${CMAKE_COMMAND} -E make_directory ${UNPACKED_WHEEL}/modular/engine
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:core> ${UNPACKED_WHEEL}/modular/engine/
  COMMAND ${CMAKE_COMMAND} -E make_directory ${UNPACKED_WHEEL}/bin
  # NOTE: Use a temporary file so that the copy command succeeds when NOT
  # STOCK_TENSORFLOW_FOUND.
  COMMAND ${CMAKE_COMMAND} -E touch ${UNPACKED_WHEEL}/bin/temp
  COMMAND ${CMAKE_COMMAND} -E copy ${UNPACKED_WHEEL}/bin/temp ${EXTERNAL_TF_EXES} ${UNPACKED_WHEEL}/bin
  COMMAND ${CMAKE_COMMAND} -E remove ${UNPACKED_WHEEL}/bin/temp
  COMMAND ${Python3_EXECUTABLE} -m wheel pack ${UNPACKED_WHEEL} --dest ${WHEEL_DIR} > /dev/null
  COMMAND ${CMAKE_COMMAND} -E rename ${WHEEL_DIR}/${WHEEL_BASENAME}.whl ${MODULAR_PACKED_WHEEL_NAME}
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/build
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/modular_engine.egg-info
  BYPRODUCTS ${UNPACKED_WHEEL}
  DEPENDS $<TARGET_FILE:core> ${MODULAR_FRAMEWORK_LIBS} ${EXTERNAL_TF_EXES}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packing wheel with bindings and shared library dependencies."
  VERBATIM
)

# Note: the `BYPRODUCTS` specification above is correct, but the generated
# `clean` target won't be correct (i.e. CMake will issue a hard error at
# configure time). Account for this by explicitly setting the
# `ADDITIONAL_CLEAN_FILES` on the directory which makes the generated `clean`
# target work correctly.
set_target_properties(modular-engine
  PROPERTIES
  ADDITIONAL_CLEAN_FILES ${UNPACKED_WHEEL}
)

if (NOT MODULAR_PYTHON_API_REPAIR_WHEEL)
  return()
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
  set(HOST_ARCH x86_64)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(HOST_ARCH aarch64)
else()
  message(FATAL_ERROR "Unsupported target architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(APPLE)
  set(REPAIR_CMD ${Python3_EXECUTABLE} -m delocate.cmd.delocate_wheel)
  set(REPAIRED_WHEEL ${WHEEL_DIR}/${WHEEL_BASENAME}_packed.whl)
  set(REPAIR_SHARED_LIB_ENV "DYLD_LIBRARY_PATH=${STOCK_TENSORFLOW_INSTALL_LIB_DIR}")
else()
  set(REPAIR_CMD ${Python3_EXECUTABLE} -m auditwheel repair --plat linux_${HOST_ARCH} --wheel-dir ${WHEEL_DIR} --strip)
  set(REPAIRED_WHEEL ${WHEEL_DIR}/${WHEEL_BASENAME}_packed.linux_${HOST_ARCH}.whl)
  set(REPAIR_SHARED_LIB_ENV "LD_LIBRARY_PATH=${STOCK_TENSORFLOW_INSTALL_LIB_DIR}")
endif()

# Detect non-system shared library dependencies and copy them into the wheel,
# updating the RECORDS metadata file.
add_custom_command(OUTPUT ${WHEEL_DIR}/${WHEEL_BASENAME}.whl
  COMMAND ${CMAKE_COMMAND} -E env
    PATH=$ENV{PATH}:${Python3_ROOT_DIR}/bin
    ${REPAIR_SHARED_LIB_ENV}
    ${REPAIR_CMD} ${WHEEL_DIR}/${WHEEL_BASENAME}_packed.whl
  COMMAND ${CMAKE_COMMAND} -E copy ${REPAIRED_WHEEL} ${WHEEL_DIR}/${WHEEL_BASENAME}.whl
  DEPENDS ${WHEEL_DIR}/${WHEEL_BASENAME}_packed.whl
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Repairing Modular Python API wheel."
  VERBATIM
)
