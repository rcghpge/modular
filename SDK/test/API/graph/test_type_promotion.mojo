# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo --debug-level full %s

from testing import assert_equal, assert_raises

from max.graph.type_promotion import promote


fn test_type_promotion() raises:
    assert_equal(promote(DType.int8, DType.int32), DType.int32)
    assert_equal(promote(DType.int8, DType.uint8), DType.int16)
    assert_equal(promote(DType.int64, DType.int64), DType.int64)
    assert_equal(promote(DType.int64, DType.uint64), DType.float16)
    with assert_raises(contains="No legal promotion"):
        _ = promote(DType.invalid, DType.int32)


def main():
    test_type_promotion()
