# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo -D TEMP_DIR=%T %s | FileCheck %s
# RUN: mt -execute %T/module_save_to_file.bin | FileCheck %s --check-prefix=CHECK-EXEC

from sys.param_env import env_get_string
from tensor import Tensor, TensorShape
from testing import assert_true, assert_raises
from pathlib.path import Path

from max.graph import AttrMap, Graph, Module, MOList, MOTensor, TypeTuple

alias TEMP_DIR = Path(env_get_string["TEMP_DIR"]())


fn test_identity_graph() raises:
    var m = Module()
    var g = m.graph(
        name="identity_graph",
        in_types=MOTensor(DType.int32),
        out_types=MOTensor(DType.int32),
    )
    _ = g.nvop(name="mo.output", inputs=g[0])

    m.verify()
    # CHECK: module {
    # CHECK:   mo.graph @identity_graph([[arg:.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[], si32>
    # CHECK:     mo.output [[arg]] : !mo.tensor<[], si32>
    # CHECK:   }
    # CHECK: }
    print(m)


fn test_identity_graph_no_module() raises:
    var g = Graph(
        name="identity_graph_no_module",
        in_types=MOTensor(DType.int32),
        out_types=MOTensor(DType.int32),
    )
    _ = g.nvop(name="mo.output", inputs=g[0])

    g.module().verify()
    # CHECK: mo.graph @identity_graph_no_module([[arg:.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[], si32>
    # CHECK:   mo.output [[arg]] : !mo.tensor<[], si32>
    # CHECK: }
    print(g)


fn test_basic_add() raises:
    var m = Module()
    var g = m.graph(
        "basic_add",
        TypeTuple(MOTensor(DType.int32), MOTensor(DType.int32)),
        MOTensor(DType.int32),
    )
    let y = g.op("mo.add", (g[0], g[1]), MOTensor(DType.int32))
    _ = g.nvop("mo.output", y)

    m.verify()
    # CHECK-LABEL: @basic_add
    # CHECK: mo.add
    print(m)


fn test_add_constant() raises:
    var m = Module()
    var g = m.graph(
        "add_constant", MOTensor(DType.float32, 1), MOTensor(DType.float32, 1)
    )
    let x = g.op(
        "mo.constant",
        MOTensor(DType.float32, 1),
        AttrMap(m.scalar_attr("value", Float32(1.0), rank=1)),
    )
    let y = g.op("mo.add", (g[0], x), MOTensor(DType.float32, 1))
    _ = g.nvop("mo.output", y)

    m.verify()
    # CHECK-LABEL: @add_constant
    # CHECK: mo.constant {value = #M.dense_array<1.000000e+00> : tensor<1xf32>}
    # CHECK: mo.add
    print(m)


# CHECK-EXEC-LABEL: save_to_file
fn test_save_to_file() raises:
    print("test_save_to_file")
    let m = Module()
    let g = m.graph("save_to_file", TypeTuple(), MOTensor(DType.int32))
    let x = g.constant(Tensor[DType.int32](TensorShape(), 7))
    g.output(x)

    # CHECK-EXEC{LITERAL}: 'save_to_file' returned tensor([[7]], dtype=si32, shape=[])
    print("True")
    m.save_to_file(TEMP_DIR / "module_save_to_file.bin")


fn main() raises:
    test_identity_graph()
    test_identity_graph_no_module()
    test_basic_add()
    test_add_constant()
    test_save_to_file()
