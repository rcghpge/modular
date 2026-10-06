# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -emit=asm --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_89 --target-features="" %s | FileCheck %s

from std.memory import stack_allocation
from std.ffi import external_call


# CHECK-LABEL: .visible .entry local_stack_allocation()
@export
def local_stack_allocation() abi("Mojo"):
    # CHECK: .local .align 4 .b8 __local_depot0[128]
    var ptr = stack_allocation[32, Float32, address_space=.LOCAL]()
    external_call["keep", NoneType](ptr)
