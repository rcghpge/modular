# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not kgen -elaborate %s -o /dev/null 2>&1 | FileCheck %s

# Enum-valued llvm cl options (e.g. `denormal-fp-math-f32`) are settable
# through `emission_option`; a value the option's own parser rejects must
# surface as an error rather than being silently ignored.

from std.collections.string.string_span import StaticString, _get_kgen_string
from max.gpu.host.info import A100
from std.sys.info import CompilationTarget


struct _Info(TrivialRegisterPassable):
    var kernel: __mlir_type.`!kgen.string`
    var name: __mlir_type.`!kgen.string`
    var num_captures: __mlir_type.index


@fieldwise_init
struct Info(TrivialRegisterPassable):
    var kernel: StaticString
    var name: StaticString
    var num_captures: Int


@always_inline
def _compile_info[
    func_type: TrivialRegisterPassable,
    //,
    func: func_type,
    /,
    emission_kind: Int = 0,
    target: CompilationTarget = A100.target,
    compile_options: StaticString = "denormal-fp-math-f32=not-a-denormal-mode",
]() -> Info:
    var info = __mlir_op.`kgen.compile_offload`[
        target_type=target._mlir_value,
        emission_kind=emission_kind.__mlir_index__(),
        emission_option=_get_kgen_string[compile_options](),
        emission_link_option=_get_kgen_string[""](),
        func=func,
        _type=_Info,
    ]()

    return Info(
        StaticString(info.kernel),
        StaticString(info.name),
        Int(mlir_value=info.num_captures),
    )


def hello():
    pass


def main():
    # CHECK: invalid value "not-a-denormal-mode" for emission option "denormal-fp-math-f32"
    var t1 = _compile_info[hello]()
    print(t1.kernel)
