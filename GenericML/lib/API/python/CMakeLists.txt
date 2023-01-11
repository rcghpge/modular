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

# find_package(Python3) has to be called before including pybind11 so that
# pybind11 picks up the correct Python development paths.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include(AddPyBind11)

set(MODULAR_PYBIND_EXT pyframework)

pybind11_add_module(${MODULAR_PYBIND_EXT}
  ${MODULAR_PYBIND_EXT}.cpp
)
target_link_libraries(${MODULAR_PYBIND_EXT} PUBLIC modular-framework)

# The extension itself must be compiled with RTTI and exceptions enabled.
target_compile_options(${MODULAR_PYBIND_EXT} PRIVATE
 $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
   # Enable RTTI and exceptions.
   -frtti -fexceptions
 >
 $<$<CXX_COMPILER_ID:MSVC>:
   # Enable RTTI and exceptions.
   /EHsc /GR>
)

# Ensure that tools needed for pyproject.toml parsing and wheel repair are
# installed.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pip install -r ${CMAKE_CURRENT_SOURCE_DIR}/packaging-requirements.txt
  RESULT_VARIABLE PIP_INSTALL_DEPS_RESULT
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND_ECHO STDOUT
)
if(PIP_INSTALL_DEPS_RESULT)
  message(FATAL_ERROR "Failed to pip install dependencies for ${MODULAR_PYBIND_EXT}")
endif()

# Use pyproject.toml as a single source of truth for the version of the Modular
# Python framework.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import tomli; f = open('${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml', 'rb'); proj = tomli.load(f)['project']; print(proj['name'] + '-' + proj['version'])"
  RESULT_VARIABLE TOMLI_RESULT
  OUTPUT_VARIABLE MODULAR_PYPROJECT_NAME_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND_ECHO STDOUT
)
if(TOMLI_RESULT)
  message(FATAL_ERROR "Failed to parse pyproject.toml for ${MODULAR_PYBIND_EXT}")
endif()

# Package the built extension into a wheel.
set(WHEEL_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(UNPACKED_WHEEL ${CMAKE_CURRENT_BINARY_DIR}/${MODULAR_PYPROJECT_NAME_VERSION})

# Get the expected name of the built Python wheel file before building it.
# This command is expected to error, which is intentionally ignored.
execute_process(
  COMMAND ${Python3_EXECUTABLE} setup.py --dry-run --quiet bdist_wheel
  OUTPUT_VARIABLE WHEEL_BASENAME
  OUTPUT_STRIP_TRAILING_WHITESPACE
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  ERROR_QUIET
)

# The wheel is built in the build directory.
# However, setuptools is unaware of the pybind module since CMake built that.
# Therefore we have to put the pybind module shared library and all other
# needed shared libraries into the wheel.
# This way everything is self-contained in the wheel and will still work after
# the wheel is pip installed.
add_custom_command(TARGET ${MODULAR_PYBIND_EXT} POST_BUILD
  COMMAND ${Python3_EXECUTABLE} -m build --wheel --outdir ${WHEEL_DIR}
  COMMAND ${Python3_EXECUTABLE} -m wheel unpack ${WHEEL_DIR}/${WHEEL_BASENAME}.whl --dest ${WHEEL_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${UNPACKED_WHEEL}/modular
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${MODULAR_PYBIND_EXT}> ${UNPACKED_WHEEL}/modular
  COMMAND ${CMAKE_COMMAND} -E make_directory ${UNPACKED_WHEEL}/lib
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tfl> ${UNPACKED_WHEEL}/lib
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:KGENCompilerRT> ${UNPACKED_WHEEL}/lib
  COMMAND ${Python3_EXECUTABLE} -m wheel pack ${UNPACKED_WHEEL} --dest ${WHEEL_DIR}
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/{build,modular.egg-info}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packing wheel with ${MODULAR_PYBIND_EXT} and shared library dependencies."
  VERBATIM
)

if(NOT MODULAR_ENABLE_WHEEL_REPAIR)
  message(STATUS "Skipping wheel repair step in Python packaging.")
  return()
endif()

if(APPLE)
  set(REPAIR_CMD delocate-wheel -v)
else()
  set(REPAIR_CMD auditwheel repair --plat linux_x86_64)
endif()

# Detect non-system shared library dependencies and copy them into the wheel,
# updating the RECORDS metadata file.
add_custom_command(TARGET ${MODULAR_PYBIND_EXT} POST_BUILD
  COMMAND ${REPAIR_CMD} ${WHEEL_DIR}/${WHEEL_BASENAME}.whl
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
