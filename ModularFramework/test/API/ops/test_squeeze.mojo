# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo -I %mobi_pkg_dir -I %test_utils_pkg_dir -D TEMP_DIR=%T "%s"

from tensor import Tensor, TensorShape

from mobi import *
from mobi.ops import squeeze


fn test_squeeze() raises:
    alias name = "test_squeeze"

    var m = Module()
    var g = m.graph(
        name,
        Arity(m.f32(dyn(), 1, 3)),
        Arity(m.f32(dyn(), 3)),
    )
    g.output(squeeze(g[0], 1))
    m.verify()

    let x = Tensor[DType.float32](TensorShape(1, 1, 3), 1, 2, 3)
    let expected = Tensor[DType.float32](TensorShape(1, 3), 1, 2, 3)

    let actual = testing.execute_unary(m, name, x)
    testing.assert_tensors_almost_equal(expected, actual)


def main():
    test_squeeze()
