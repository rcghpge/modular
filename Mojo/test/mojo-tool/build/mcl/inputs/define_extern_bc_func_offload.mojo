# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from std.compile import compile_info
from max.gpu.host.info import _mi355x_target


@export("my_add_one")
def my_add_one(x: Pointer[Int32, MutAnyOrigin]) abi("C"):
    x[] += 1


def main():
    var llvm_text = compile_info[
        my_add_one,
        emission_kind="llvm",
        target=_mi355x_target,
    ]()
    print(llvm_text)
