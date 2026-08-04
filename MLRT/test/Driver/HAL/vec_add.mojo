# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""``vec_add`` test kernel module — built as its own ``vec_add.so``."""

from std.gpu import global_idx
from std.memory import Pointer
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from _mojo_module.bundle import compile_to_python_bundle


def vec_add_kernel(
    a: Pointer[Float32, MutAnyOrigin],
    b: Pointer[Float32, MutAnyOrigin],
    result: Pointer[Float32, MutAnyOrigin],
    n: Int,
):
    var tid = global_idx.x
    if tid >= n:
        return
    result[unsafe_offset=tid] = a[unsafe_offset=tid] + b[unsafe_offset=tid]


def vec_add_compiled(ctx_obj: PythonObject) raises -> PythonObject:
    return compile_to_python_bundle[type_of(vec_add_kernel), vec_add_kernel](
        ctx_obj
    )


@export
def PyInit_vec_add() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("vec_add")
        b.def_function[vec_add_compiled](
            "vec_add",
            docstring="Compile the built-in vec_add test kernel.",
        )
        return b.finalize()
    except e:
        abort(t"failed to create Python module: {e}")
