# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not kgen -emit=llvm %s 2>&1 | FileCheck %s
# RUN: not kgen -emit=llvm %s --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_89 --target-features="" 2>&1 | FileCheck %s


@export
def cast_from_float8_e3m4(
    dst: Pointer[Float32, MutAnyOrigin], src: Scalar[.float8_e3m4]
) abi("Mojo"):
    # CHECK: error: conversion from 'f8e3m4' to 'f32' is not implemented
    dst[0] = src.cast[.float32]()


@export
def add_from_float8_e3m4(
    dst: Pointer[Scalar[.float8_e3m4], MutAnyOrigin],
    src: Scalar[.float8_e3m4],
) abi("Mojo"):
    # CHECK: error: operation with a type 'f8e3m4' is not implemented
    dst[0] += src
