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
  -e Support/python \
  -r KGEN/utils/requirements.txt \
  -r GenericML/lib/API/python/packaging-requirements.txt

pip install pipx

