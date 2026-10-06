# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.gpu.host.compile import _compile_code
from max.gpu.host import get_gpu_target
from std.sys._assembly import inlined_assembly

# RUN: %mojo -debug-level full %s

# COM: MOCO-1881 This file tests having one function to be both kernel function
# and callee in another kernel.


def alloc[p: UInt32](i: UInt32, j: Pointer[UInt32, MutAnyOrigin]):
    j[unsafe_offset=0] = p + i


def test_fn(v: Pointer[UInt32, MutAnyOrigin]):
    alloc[1](1, v)


def main() raises:
    # use alloc[1] as kernel entry
    print(
        _compile_code[
            alloc[1],
            target=get_gpu_target["sm_100a"](),
        ]()
    )

    # alloc[1] is a callee in kernel entry test_fn
    print(
        _compile_code[
            test_fn,
            target=get_gpu_target["sm_100a"](),
        ]()
    )
