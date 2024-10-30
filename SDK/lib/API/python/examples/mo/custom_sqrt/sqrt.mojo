# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from register import mogg_elementwise, mogg_register_override


@mogg_register_override("mo.sqrt", 1)
@mogg_elementwise
fn my_sqrt[
    type: DType, simd_width: Int
](value: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return 42
