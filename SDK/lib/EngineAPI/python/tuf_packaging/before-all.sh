#!/bin/bash
##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# This script executes before cibuildwheel kicks off the wheel build for
# `modular-engine`.


# Log commands and exit on error.
set -exo pipefail

cd /project
# Opt out of the autovenv to force using cibuildwheel's provided Python
# installation without a venv.
export MODULAR_AUTOVENV_OPTOUT=1
source ./utils/start-modular.sh

rm -rf .derived

setup_virtualenv
${MODULAR_PYTHON} -m pip install -e Support/python
git-lfs install
git-lfs pull

arch=$(uname -m)

# Install requirements for building MLIR Python bindings.
pip install -r $MODULAR_PATH/third-party/llvm-project/mlir/python/requirements.txt

# Build LLVM.
# Note: LLDB Python breaks because manylinux ships with a statically linked Python interpreter.
DISABLE_LLDB_PYTHON_SCRIPTING=true \
cmake-llvm --sanitizer=none --build-type=release --build-profile=production \
  -D CMAKE_C_COMPILER=/usr/bin/clang-15 \
  -D CMAKE_CXX_COMPILER=/usr/bin/clang++-15 \

build-llvm

build-tcmalloc

pip install \
  -e Support/python \
  -r KGEN/utils/requirements.txt \
  -r SDK/lib/EngineAPI/python/packaging-requirements.txt

pip install pipx
