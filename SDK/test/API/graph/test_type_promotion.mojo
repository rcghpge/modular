# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s

from testing import assert_equal, assert_raises

from max.graph.type_promotion import implicit_cast_type


fn test_type_promotion() raises:
    assert_equal(implicit_cast_type(DType.int8, DType.int32), DType.int32)
    assert_equal(implicit_cast_type(DType.int8, DType.uint8), DType.int16)
    assert_equal(implicit_cast_type(DType.int64, DType.int64), DType.int64)
    assert_equal(implicit_cast_type(DType.int64, DType.uint64), DType.float16)
    with assert_raises(contains="no legal promotion"):
        _ = implicit_cast_type(DType.invalid, DType.int32)


def main():
    test_type_promotion()
