# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo "%s"

from max.graph import _testing, Graph, TensorType, Type, Dim, Symbol
from max.graph.ops.slicing import concat
from testing import assert_true


fn test_concat() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.int32, 3, 2),
            TensorType(DType.int32, 3, 4),
            TensorType(DType.int32, 3, Dim(String("x"))),
        ),
    )

    #      concat axis
    #              |
    #              v
    # shape_0: [3, 2]
    # shape_1: [3, 4]
    var cc1 = concat(List[Symbol](g[0], g[1]), axis=1)
    # TODO(GRA-552)
    assert_true(str(cc1).endswith("!mo.tensor<[3, 6], si32>"))

    #      concat axis
    #              |
    #              v
    # shape_0: [3, 4]
    # shape_1: [3, x]
    var cc2 = concat(List[Symbol](g[1], g[2]), axis=1)
    # TODO(GRA-552)
    assert_true(str(cc2).endswith("!mo.tensor<[3, add(x, 4)], si32>"))


fn test_concat_error() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.int32, 3, String("x")),
            TensorType(DType.int32, 5, 6),
        ),
    )
    #      concat axis
    #              |
    #              v
    # shape_0: [3, x]
    # shape_1: [5, 6]
    #
    # NOTE error because `3 != 5`, user made a mistake here.
    try:
        var cc3 = concat(List[Symbol](g[0], g[1]), axis=1)
    except e:
        assert_true(
            str(e).endswith(
                "[concat] input shapes must match except at concat axis"
            )
        )

    #      concat axis
    #           |
    #           v
    # shape_0: [3, x]
    # shape_1: [5, 6]
    #
    # NOTE error because we can't prove `x == 6` at compile time; user must use
    # rebind to "cast" the `x` dimension to 6.
    try:
        var cc4 = concat(List[Symbol](g[0], g[1]), axis=0)
    except e:
        assert_true(
            str(e).endswith(
                "[concat] input shapes must match except at concat axis"
            )
        )


def main():
    test_concat()
    test_concat_error()
