# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import compiler_internal as compiler
from tensor import OutputTensor
from collections.string.string_slice import _get_kgen_string
from collections.string import StaticString


# Custom version of constrained so we get the note in this file.
@always_inline("nodebug")
fn constrained[cond: Bool, msg: StaticString]():
    __mlir_op.`kgen.param.assert`[
        cond = cond.__mlir_i1__(),
        message = _get_kgen_string[msg](),
    ]()  # expected-note {{constraint failed}}


@compiler.register("fails_to_elaborate")
struct InvalidOwnedArgConvention:
    @staticmethod
    fn execute[
        target: StaticString, dtype: DType, rank: Int
    ](output: OutputTensor[dtype=dtype, rank=rank]):
        constrained[False, "oops"]()  # expected-error {{call expansion failed}}
