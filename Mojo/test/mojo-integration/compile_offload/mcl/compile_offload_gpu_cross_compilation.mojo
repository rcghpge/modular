# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# COM: cross compilation should work for all platforms no matter whether
# there is a nvidia GPU or not.
# RUN: %mojo-build --target-accelerator="nvidia:sm_90a" %s -o %t
# RUN: llvm-objdump -s %t | FileCheck %s

from max.gpu.host import DeviceContext


def hello():
    pass


# CHECK: Generated
# CHECK: by LLVM NVPTX
def main() raises:
    with DeviceContext() as ctx:
        var t1 = ctx.compile_function[hello]()
        print(t1._func_impl.asm)
