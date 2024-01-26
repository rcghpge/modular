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

# Download specific cached TensorFlow
s3_blob="s3://modular-ci-prod-artifact-bucket/ci/v2.13.1/${arch}/main/manylinux_2_28-tf_cache.tar.gz"
download_cached_tf --blob=${s3_blob} --ref=v2.13.1

# Build external PyTorch
# This is needed for the install.py in third-party/stock-pytorch
python3 -m pip install boto3
build-external-pytorch --pytorch-abi cxx11

# Build external torchvision
build-external-torchvision

# Build LLVM.
# Note: LLDB Python breaks because manylinux ships with a statically linked Python interpreter.
DISABLE_LLDB_PYTHON_SCRIPTING=true \
cmake-llvm --sanitizer=none --build-type=release \
  -D CMAKE_C_COMPILER=/usr/bin/clang-15 \
  -D CMAKE_CXX_COMPILER=/usr/bin/clang++-15 \

build-llvm

pip install \
  -e utils/build/engine_packaging[dev] \
  -e Support/python \
  -r KGEN/utils/requirements.txt \
  -r GenericML/lib/API/python/packaging-requirements.txt

echo "Configuring to build dylib-wheel"

# Configure and build Modular.
cmake-modular --sanitizer=none --build-type=release \
  -D CMAKE_C_COMPILER=/usr/bin/clang-15 \
  -D CMAKE_CXX_COMPILER=/usr/bin/clang++-15 \
  -D MODULAR_RELEASE_PACKAGE_BUILD=ON \
  -D CMAKE_INSTALL_PREFIX=install \

echo "Install engine dylibs for wheel"

build GeneratedTests install-ai-engine-dylibs

echo "Building dylib wheel"
rm -rf .derived/build/install/modular-engine-dylibs/dist/  .derived/build/install/modular-engine-dylibs/build
python -m build --wheel .derived/build/install/modular-engine-dylibs

pip install wheel==0.42
# Rename Python wheel to indicate to pip that it is compatible with all Python3 versions
wheel tags --python-tag=py3 --abi-tag=none .derived/build/install/modular-engine-dylibs/dist/*.whl

cp .derived/build/install/modular-engine-dylibs/dist/*py3-none*.whl .
