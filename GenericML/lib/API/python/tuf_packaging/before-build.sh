#!/bin/bash
##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# This script executes for each Python version as kicked off from `cibuildwheel` to
# build the `modular-engine` for each respective Python version.

# Log commands and exit on error.
set -exo pipefail

# Expand aliases from start-modular.sh in this script.
shopt -s expand_aliases

cd /project

# Opt out of the autovenv to force using cibuildwheel's provided Python
# installation without a venv.
export MODULAR_AUTOVENV_OPTOUT=1

MAJOR_VERSION=$1
MINOR_VERSION=$2
PATCH_VERSION=$3

source ./utils/start-modular.sh
python3 -m pip install \
  -e Support/python \
  -r KGEN/utils/requirements.txt \
  -r GenericML/lib/API/python/packaging-requirements.txt

pip install pipx

# Configure and build Modular.
cmake-modular --sanitizer=none --build-type=release \
  -D CMAKE_C_COMPILER=/usr/bin/clang-15 \
  -D CMAKE_CXX_COMPILER=/usr/bin/clang++-15 \
  -D MODULAR_RELEASE_PACKAGE_BUILD=ON \
  -D CMAKE_INSTALL_PREFIX=install \
  -D MODULAR_VERSION_MAJOR=$MAJOR_VERSION \
  -D MODULAR_VERSION_MINOR=$MINOR_VERSION \
  -D MODULAR_VERSION_PATCH=$PATCH_VERSION

# The GeneratedTests target explicitly isn't a dependency of the AI Engine Wheel itself
# but is needed for generating the models we run using the just-built wheel.

# Build and install the necessary libs and executables
# for the AI Engine Wheel.
build install-ai-engine-py-binding

echo "sccache stats after building all:"
sccache --show-stats

tree -a -C .derived/build/install/modular-engine-py-binding
