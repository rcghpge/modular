// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false" -o %t.adv.mlir
// RUN: mt --execute --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.legacy.mlir
// RUN: mt --execute --result-output-style=full %t.legacy.mlir | FileCheck %s

// CHECK-LABEL: Running 'add':
// CHECK: 'add' returned tensor<2x2xf32> [1.100000e+01, 2.200000e+01, 3.300000e+01, 4.400000e+01]
mo.graph @add() -> (!mo.tensor<[2, 2], f32>) {
  %a = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %b = mo.constant {value = #M.dense_array<10.0, 20.0, 30.0, 40.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %ch0 = mo.chain.create()
  %a_gpu, %ch1 = mo.transfer[%ch0] %a : !mo.tensor<[2, 2], f32> to <"gpu", 0>
  %b_gpu, %ch2 = mo.transfer[%ch1] %b : !mo.tensor<[2, 2], f32> to <"gpu", 0>
  %c_gpu = mo.add(%a_gpu, %b_gpu) : !mo.tensor<[2, 2], f32, gpu:0>
  %c, %ch3 = mo.transfer[%ch2] %c_gpu : !mo.tensor<[2, 2], f32, gpu:0> to <"cpu", 0>
  mo.output %c : !mo.tensor<[2, 2], f32>
}
