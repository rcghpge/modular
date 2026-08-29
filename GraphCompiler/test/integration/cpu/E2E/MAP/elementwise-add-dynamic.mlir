// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false" -o %t.adv.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=2x2 --input-shapes=2x2 --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.legacy.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=2x2 --input-shapes=2x2 --result-output-style=full %t.legacy.mlir | FileCheck %s

// CHECK-LABEL: Running 'add_dynamic':
// CHECK: 'add_dynamic' returned tensor<2x2xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
mo.graph @add_dynamic<M, N>(%a: !mo.tensor<[M, N], f32>, %b: !mo.tensor<[M, N], f32>) -> (!mo.tensor<[M, N], f32>) {
  %c = mo.add(%a, %b) : !mo.tensor<[M, N], f32>
  mo.output %c : !mo.tensor<[M, N], f32>
}
