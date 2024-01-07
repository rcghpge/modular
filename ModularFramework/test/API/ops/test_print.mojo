# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo -I %mobi_pkg_dir -I %test_utils_pkg_dir -D TEMP_DIR=%T "%s" | FileCheck %s

from tensor import Tensor, TensorShape

from mobi import *


fn test_print() raises:
    alias name = "test_print"

    var m = Module()
    var g = m.graph(
        name,
        Arity(m.f32(dyn(), dyn(), 2)),
        Arity(),
    )
    g[0].print()
    g.output()
    m.verify()

    let x = Tensor[DType.float32](TensorShape(2, 2, 2), 1, 2, 3, 4, 5, 6, 7, 8)

    # CHECK{LITERAL}: debug_tensor_cpu = tensor([[[1.0000, 2.0000],
    # CHECK{LITERAL}: [3.0000, 4.0000]],
    # CHECK{LITERAL}: [[5.0000, 6.0000],
    # CHECK{LITERAL}: [7.0000, 8.0000]]], dtype=f32, shape=[2,2,2])
    _ = testing.execute_base(m, name, x)


def main():
    test_print()
