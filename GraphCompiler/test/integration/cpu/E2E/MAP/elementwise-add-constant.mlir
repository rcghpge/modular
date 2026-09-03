// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false" -o %t.adv.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.legacy.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.legacy.mlir | FileCheck %s

// `%bias` is folded into the kernel body by `FuseConstantIterOpaqueInputs`, so
// it reaches the kernel as a constant resource rather than an operand. Unlike
// `elementwise-add.mlir`, which adds two constants and leaves the kernel with
// no operands at all, `%x` is a graph argument (all-1s from `mt`) -- so this
// pins the mixed case, where the kernel keeps a real input alongside the
// folded constant, and checks the constant's bytes survive to execution.

// CHECK-LABEL: Running 'add_constant':
// CHECK: 'add_constant' returned tensor<2x2xf32> [1.100000e+01, 2.100000e+01, 3.100000e+01, 4.100000e+01]
mo.graph @add_constant(%x: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>) {
  %bias = mo.constant {value = #M.dense_array<10.0, 20.0, 30.0, 40.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %c = mo.add(%x, %bias) : !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}
