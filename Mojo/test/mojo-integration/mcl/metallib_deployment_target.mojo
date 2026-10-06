# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: system-darwin, air-objdump
# RUN: env MACOSX_DEPLOYMENT_TARGET=14.0 not %mojo %s 2>&1 | FileCheck %s --check-prefix=TOO-OLD
# RUN: env MACOSX_DEPLOYMENT_TARGET=abc not %mojo %s 2>&1 | FileCheck %s --check-prefix=INVALID

# COM: A deployment target too old for the kernel's AIR makes air-lld drop the
#      kernel and still exit 0.

# TOO-OLD: Metal Compiler dropped the AIR file from the metallib due to:
# TOO-OLD-NEXT: air-lld: warning: ignoring file {{.*}}AIR version

# INVALID: Metal Compiler failed to compile metallib.
# INVALID: invalid version number in 'MACOSX_DEPLOYMENT_TARGET=

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code


def kernel():
    pass


def main():
    print(
        _compile_code[
            kernel,
            emission_kind="object",
            target=get_gpu_target["metal:1"](),
        ]()
    )
