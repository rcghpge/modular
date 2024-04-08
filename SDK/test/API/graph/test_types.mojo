# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo %s

from tensor import TensorSpec
import testing

import _mlir

from max.graph import Dim, ElementType, MOList, MOTensor


fn context() -> _mlir.Context:
    var ctx = _mlir.Context()
    ctx.load_modular_dialects()
    ctx.load_all_available_dialects()
    return ctx


fn test_dtype_to_string() raises:
    var ui = ElementType(DType.uint8)
    var s = ui.to_string(context())

    testing.assert_equal(len(s), 3)
    testing.assert_equal(s, "ui8")


fn test_dtype() raises:
    testing.assert_equal(ElementType(DType.uint8).to_string(context()), "ui8")
    testing.assert_equal(ElementType(DType.float32).to_string(context()), "f32")
    testing.assert_equal(ElementType(DType.int64).to_string(context()), "si64")


fn test_tensor_type() raises:
    var dt = ElementType(DType.uint8)
    var t = MOTensor(dt, 1, 2, 3)

    var s = t.to_string(context())
    testing.assert_equal(s, "!mo.tensor<[1, 2, 3], ui8>")


fn test_tensor_unranked() raises:
    var dt = ElementType(DType.float32)
    var t = MOTensor(dt, False)

    var s = t.to_string(context())
    testing.assert_equal(s, "!mo.tensor<?, f32>")


fn test_tensor_dynamic_dims() raises:
    var dt = ElementType(DType.float32)
    var t = MOTensor(dt, 1, Dim.dynamic(), 4)

    var s = t.to_string(context())
    testing.assert_equal(s, "!mo.tensor<[1, ?, 4], f32>")


fn test_tensor_dynamic_dims_roundtrip() raises:
    var t = MOTensor(DType.float32, Dim.dynamic(), 4)
    var t2 = MOTensor.from_mlir(t.to_mlir(context()))
    testing.assert_true(t == t2, "roundtrip dynamic dim")


fn test_tensor_symbolic_dims() raises:
    var dt = ElementType(DType.float32)
    var t = MOTensor(dt, Dim.symbolic("batch"), 4)

    var s = t.to_string(context())
    testing.assert_equal(s, "!mo.tensor<[batch, 4], f32>")


fn test_tensor_symbolic_dims_roundtrip() raises:
    var t = MOTensor(DType.float32, Dim.symbolic("batch"), 4)
    var t2 = MOTensor.from_mlir(t.to_mlir(context()))
    print(t.to_mlir(context()))
    print(t2.to_mlir(context()))
    testing.assert_true(t == t2, "roundtrip symbolic dim")


fn test_tensor_scalar() raises:
    var dt = ElementType(DType.float32)
    var t = MOTensor(dt)

    var s = t.to_string(context())
    testing.assert_equal(s, "!mo.tensor<[], f32>")


fn test_dtype_autocast() raises:
    testing.assert_equal(
        MOTensor(DType.float32, 1, 2, 3).to_string(context()),
        "!mo.tensor<[1, 2, 3], f32>",
    )


fn test_int_autocast() raises:
    # This just verifes that the constructor doesn't go to the wrong overload,
    # like the List one.
    testing.assert_equal(
        MOTensor(DType.float32, 1).to_string(context()),
        "!mo.tensor<[1], f32>",
    )


fn test_tensor_spec_cast() raises:
    testing.assert_equal(
        MOTensor(TensorSpec(DType.float32)).to_string(context()),
        "!mo.tensor<[], f32>",
    )
    testing.assert_equal(
        MOTensor(TensorSpec(DType.float64, 1)).to_string(context()),
        "!mo.tensor<[1], f64>",
    )
    testing.assert_equal(
        MOTensor(TensorSpec(DType.int32, 1, 2)).to_string(context()),
        "!mo.tensor<[1, 2], si32>",
    )


fn test_list_type() raises:
    var l = MOList(MOTensor(DType.uint8, 1, Dim.dynamic(), 3))

    var s = l.to_string(context())
    testing.assert_equal(s, "!mo.list<!mo.tensor<[1, ?, 3], ui8>>")


fn main() raises:
    test_dtype_to_string()
    test_dtype()
    test_tensor_type()
    test_tensor_unranked()
    test_tensor_dynamic_dims()
    test_tensor_dynamic_dims_roundtrip()
    test_tensor_symbolic_dims()
    test_tensor_symbolic_dims_roundtrip()
    test_tensor_scalar()
    test_dtype_autocast()
    test_int_autocast()
    test_tensor_spec_cast()
    test_list_type()
