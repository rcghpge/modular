// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false" -o %t.adv.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=2x2 --input-shapes=2x2 --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.legacy.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=2x2 --input-shapes=2x2 --result-output-style=full %t.legacy.mlir | FileCheck %s

// CHECK-LABEL: Running 'add_dynamic':
// CHECK: 'add_dynamic' returned tensor<2x2xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
mo.graph @add_dynamic<M, N>(%a: !mo.tensor<[M, N], f32>, %b: !mo.tensor<[M, N], f32>) -> (!mo.tensor<[M, N], f32>) {
  %ch0 = mo.chain.create()
  %a_gpu, %ch1 = mo.transfer[%ch0] %a : !mo.tensor<[M, N], f32> to <"gpu", 0>
  %b_gpu, %ch2 = mo.transfer[%ch1] %b : !mo.tensor<[M, N], f32> to <"gpu", 0>
  %c_gpu = mo.add(%a_gpu, %b_gpu) : !mo.tensor<[M, N], f32, gpu:0>
  %c, %ch3 = mo.transfer[%ch2] %c_gpu : !mo.tensor<[M, N], f32, gpu:0> to <"cpu", 0>
  mo.output %c : !mo.tensor<[M, N], f32>
}
