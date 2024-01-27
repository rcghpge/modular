# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo -D TEMP_DIR=%T %s | FileCheck %s
# RUN: mt -execute %T/module_to_bytecode.bin | FileCheck %s --check-prefix=CHECK-MTB
# RUN: mt -execute %T/module_save_to_file.bin | FileCheck %s --check-prefix=CHECK-MTB2

from sys.param_env import env_get_string
from tensor import Tensor, TensorShape

from max.graph import *


alias TEMP_DIR = Path(env_get_string["TEMP_DIR"]())


fn test_identity_graph() raises:
    var m = Module()
    var g = m.graph(
        name="identity_graph",
        in_types=Arity(m.i32()),
        out_types=Arity(m.i32()),
    )
    _ = g.nvop(name="mo.output", inputs=g[0])

    m.verify()
    # CHECK: module {
    # CHECK:   mo.graph @identity_graph([[arg:.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[], si32>
    # CHECK:     mo.output [[arg]] : !mo.tensor<[], si32>
    # CHECK:   }
    # CHECK: }
    print(m.__str__())


fn test_basic_add() raises:
    var m = Module()
    var g = m.graph("basic_add", Arity(m.i32(), m.i32()), Arity(m.i32()))
    let y = g.op("mo.add", (g[0], g[1]), MOTensor(DType.int32))
    _ = g.nvop("mo.output", y)

    m.verify()
    # CHECK-LABEL: @basic_add
    # CHECK: mo.add
    print(m.__str__())


fn test_add_constant() raises:
    var m = Module()
    var g = m.graph("add_constant", Arity(m.f32(1)), Arity(m.f32(1)))
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
    print(m.__str__())


# CHECK-LABEL: test_bytecode_output
fn test_bytecode_output() raises:
    print("test_bytecode_output")
    let m = Module()
    let g = m.graph("bytecode_output", Arity(), Arity(m.i32()))
    let x = g.constant(Tensor[DType.int32](TensorShape(), 7))
    g.output(x)

    # CHECK: True
    # CHECK-MTB{LITERAL}: 'bytecode_output' returned tensor([[7]], dtype=si32, shape=[])
    with open(TEMP_DIR / "module_to_bytecode.bin", "w") as f:
        print("True")
        f.write(m.m.as_op().bytecode())


# CHECK-LABEL: test_save_to_file
fn test_save_to_file() raises:
    print("test_save_to_file")
    let m = Module()
    let g = m.graph("save_to_file", Arity(), Arity(m.i32()))
    let x = g.constant(Tensor[DType.int32](TensorShape(), 7))
    g.output(x)

    # CHECK: True
    # CHECK-MTB2{LITERAL}: 'save_to_file' returned tensor([[7]], dtype=si32, shape=[])
    print("True")
    m.save_to_file(TEMP_DIR / "module_save_to_file.bin")


fn main() raises:
    test_identity_graph()
    test_basic_add()
    test_add_constant()
    test_bytecode_output()
    test_save_to_file()
