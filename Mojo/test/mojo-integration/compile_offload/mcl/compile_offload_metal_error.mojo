# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not kgen -elaborate %s 2>&1 | FileCheck %s

from std.collections.string.string_span import StaticString, _get_kgen_string
from max.gpu.host.info import MetalM2
from std.sys.info import CompilationTarget


@fieldwise_init
struct _Info(TrivialRegisterPassable):
    var kernel: __mlir_type.`!kgen.string`
    var name: __mlir_type.`!kgen.string`
    var num_captures: __mlir_type.index
    var capture_sizes: __mlir_type.`!kgen.pointer<ui64>`


@fieldwise_init
struct Info(TrivialRegisterPassable):
    var kernel: StaticString
    var name: StaticString
    var num_captures: Int

    @__allow_legacy_any_origin_fields
    var capture_sizes: Pointer[__mlir_type.ui64, ImmUnsafeAnyOrigin]


@always_inline
def _compile_info[
    func_type: TrivialRegisterPassable,
    //,
    func: func_type,
    /,
    emission_kind: Int = 0,
    target: CompilationTarget = MetalM2.target,
    compile_options: StaticString = target.default_compile_options(),
]() -> Info:
    var info = __mlir_op.`kgen.compile_offload`[
        target_type=target._mlir_value,
        emission_kind=emission_kind.__mlir_index__(),
        emission_option=_get_kgen_string[compile_options](),
        emission_link_option=_get_kgen_string[""](),
        func=func,
        _type=_Info,
    ]()
    var num_captures_ptr_ui64 = Pointer[__mlir_type.ui64, ImmUnsafeAnyOrigin](
        _mlir_value=__mlir_op.`pop.pointer.bitcast`[
            _type=Pointer[__mlir_type.ui64, ImmUnsafeAnyOrigin]._mlir_type,
        ](info.capture_sizes)
    )

    return Info(
        StaticString(info.kernel),
        StaticString(info.name),
        Int(mlir_value=info.num_captures),
        num_captures_ptr_ui64,
    )


def hello(x: __mlir_type.f64):
    pass


def main():
    # CHECK: argument{{.*}} has unsupported type 'double'
    var t1 = _compile_info[hello]()
    print(t1.kernel)
