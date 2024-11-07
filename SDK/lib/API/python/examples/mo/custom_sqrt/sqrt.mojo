# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from register import mogg_elementwise, register_internal_override


@register_internal_override("mo.sqrt", 1)
@mogg_elementwise
fn my_sqrt[
    type: DType, simd_width: Int
](value: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return 42
