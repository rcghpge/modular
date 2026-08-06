// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false" -o %t.adv.mlir
// RUN: mt --execute --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.legacy.mlir
// RUN: mt --execute --result-output-style=full %t.legacy.mlir | FileCheck %s

// CHECK-LABEL: Running 'add':
// CHECK: 'add' returned tensor<2x2xf32> [1.100000e+01, 2.200000e+01, 3.300000e+01, 4.400000e+01]
mo.graph @add() -> (!mo.tensor<[2, 2], f32>) {
  %a = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %b = mo.constant {value = #M.dense_array<10.0, 20.0, 30.0, 40.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %c = mo.add(%a, %b) : !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}
