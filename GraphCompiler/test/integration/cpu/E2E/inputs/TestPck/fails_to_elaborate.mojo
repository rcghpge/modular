# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import extensibility
from extensibility import OutputTensor
from std.collections.string.string_span import _get_kgen_string
from std.collections.string import StaticString


# Custom version of constrained so we get the note in this file.
@always_inline("nodebug")
def constrained[cond: Bool, msg: StaticString]():
    __mlir_op.`kgen.param.assert`[
        cond=cond.__mlir_bool__(),
        message=_get_kgen_string[msg](),
    ]()  # expected-note {{constraint failed}}


@extensibility.register("fails_to_elaborate")
struct InvalidOwnedArgConvention:
    @staticmethod
    def execute[
        target: StaticString, dtype: DType, rank: Int
    ](output: OutputTensor[dtype=dtype, rank=rank, ...]):
        comptime assert (
            False
        ), "oops"  # expected-error {{call expansion failed}}
