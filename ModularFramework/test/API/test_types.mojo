# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo -I %mobi_pkg_dir %s

from testing import *

from mobi.motype import *


fn test_dtype_to_string() raises:
    let ui = ElementType(DType.uint8)
    let s = ui.to_string(mobi.Module())

    assert_equal(len(s), 3)
    assert_equal(s, "ui8")


fn test_dtype() raises:
    assert_equal(ElementType(DType.uint8).to_string(mobi.Module()), "ui8")
    assert_equal(ElementType(DType.float32).to_string(mobi.Module()), "f32")
    assert_equal(ElementType(DType.int64).to_string(mobi.Module()), "si64")


fn test_tensor_type() raises:
    let dt = ElementType(DType.uint8)
    let t = MOTensor(dt, 1, 2, 3)

    let s = t.to_string(mobi.Module())
    assert_equal(s, "!mo.tensor<[1, 2, 3], ui8>")


fn test_tensor_unranked() raises:
    let dt = ElementType(DType.float32)
    let t = MOTensor(dt, False)

    let s = t.to_string(mobi.Module())
    assert_equal(s, "!mo.tensor<?, f32>")


fn test_tensor_dynamic_dims() raises:
    let dt = ElementType(DType.float32)
    let t = MOTensor(dt, 1, dyn(), 4)

    let s = t.to_string(mobi.Module())
    assert_equal(s, "!mo.tensor<[1, ?, 4], f32>")


fn test_tensor_scalar() raises:
    let dt = ElementType(DType.float32)
    let t = MOTensor(dt)

    let s = t.to_string(mobi.Module())
    assert_equal(s, "!mo.tensor<[], f32>")


fn test_dtype_cast() raises:
    assert_equal(
        MOTensor(DType.float32, 1, 2, 3).to_string(mobi.Module()),
        "!mo.tensor<[1, 2, 3], f32>",
    )


fn test_int_cast() raises:
    assert_equal(
        MOTensor(DType.float32, 1).to_string(mobi.Module()),
        "!mo.tensor<[1], f32>",
    )


fn test_tensor_spec_cast() raises:
    assert_equal(
        MOTensor(TensorSpec(DType.float32)).to_string(mobi.Module()),
        "!mo.tensor<[], f32>",
    )
    assert_equal(
        MOTensor(TensorSpec(DType.float64, 1)).to_string(mobi.Module()),
        "!mo.tensor<[1], f64>",
    )
    assert_equal(
        MOTensor(TensorSpec(DType.int32, 1, 2)).to_string(mobi.Module()),
        "!mo.tensor<[1, 2], si32>",
    )


fn main() raises:
    test_dtype_to_string()
    test_dtype()
    test_tensor_type()
    test_tensor_unranked()
    test_tensor_dynamic_dims()
    test_tensor_scalar()
    test_dtype_cast()
    test_int_cast()
    test_tensor_spec_cast()
