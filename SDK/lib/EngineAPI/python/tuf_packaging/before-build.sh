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

# Opt out of the autovenv to force using cibuildwheel's provided Python
# installation without a venv.
export MODULAR_AUTOVENV_OPTOUT=1

MAJOR_VERSION=$1
MINOR_VERSION=$2
PATCH_VERSION=$3

CLANG_PATH="/usr/bin/clang"
CLANG_PLUS_PATH="/usr/bin/clang++"
# Sourcing start-modular.sh has some strange properties that breaks the build on
# MacOS. Since we're running directly on the host, this causes the build to
# link all wheels against the homebrew 3.11 Python. As a result, we have to
# run this script without invoking `start-modular.sh` on MacOS, which is why
# we need these manual expansions of our `cmake-modular` and `build` commands.
CMAKE_COMMAND="$MODULAR_PYTHON utils/build.py configure modular"
BUILD_COMMAND="$MODULAR_PYTHON utils/build.py build modular"

if [[ "$OSTYPE" != "darwin"* ]]; then
  cd /project
  source ./utils/start-modular.sh
  CLANG_PATH="/usr/bin/clang-15"
  CLANG_PLUS_PATH="/usr/bin/clang++-15"
  CMAKE_COMMAND="cmake-modular"
  BUILD_COMMAND="build"
fi

python3 -m pip install \
  -e Support/python \
  -r KGEN/utils/requirements.txt \
  -r SDK/lib/EngineAPI/python/packaging-requirements.txt

# A torch install is required for the build & tests.
python3 third-party/stock-pytorch/install.py install

pip install pipx

if [ -z ${MOJO_INSTALL_CHANNEL+x} ]; then echo "MOJO_INSTALL_CHANNEL is unset"; else echo "MOJO_INSTALL_CHANNEL is set to '$MOJO_INSTALL_CHANNEL'"; fi

# Configure and build Modular.
$CMAKE_COMMAND --sanitizer=none --build-profile=production --build-type=release \
  -D CMAKE_C_COMPILER=$CLANG_PATH \
  -D CMAKE_CXX_COMPILER=$CLANG_PLUS_PATH \
  -D CMAKE_INSTALL_PREFIX=install \
  -D MODULAR_VERSION_MAJOR=$MAJOR_VERSION \
  -D MODULAR_VERSION_MINOR=$MINOR_VERSION \
  -D MODULAR_VERSION_PATCH=$PATCH_VERSION \
  -D MODULAR_STATIC_LINKED_PYTHON=ON \
  -D MODULAR_MOJO_INSTALL_CHANNEL=${MOJO_INSTALL_CHANNEL:-max} \
  -D MODULAR_MAX_INSTALL_CHANNEL=${MAX_INSTALL_CHANNEL:-stable}


# The GeneratedTests target explicitly isn't a dependency of the AI Engine Wheel itself
# but is needed for generating the models we run using the just-built wheel.

# Build and install the necessary libs and executables
# for the AI Engine Wheel.
$BUILD_COMMAND install-ai-engine-py-binding

echo "sccache stats after building all:"
sccache --show-stats

tree -a -C .derived/build/install/max-engine-py-binding
