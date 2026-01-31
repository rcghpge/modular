// RUN: mo-opt %s --mo-to-mgp=constant-fold=false -o %t.mlir
// RUN: mt --execute --model-inputs=ones -result-output-style=full %t.mlir | FileCheck %s

mo.graph @cumsum(%ones : !mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>) {
  %x = mo.constant {value = #M.dense_array<-1.0, 1.0, 2.0, -2.0, 5.0> : tensor<5xf32>} : !mo.tensor<[5], f32>
  %inp = mo.mul(%ones, %x) : !mo.tensor<[5], f32>
  %axis = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %res = mo.cumsum(%inp, %axis) {reverse = 0 : index, exclusive = 0 : index} : (!mo.tensor<[5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[5], f32>
  mo.output %res : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'cumsum' returned tensor<5xf32> [-1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 5.000000e+00]

mo.graph @cumsum_axis1(%ones : !mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>) {
  %x = mo.constant {value = #M.dense_array<-1.0, 1.0, 2.0, -2.0, 5.0> : tensor<5xf32>} : !mo.tensor<[5], f32>
  %inp = mo.mul(%ones, %x) : !mo.tensor<[5], f32>
  %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %res = mo.cumsum(%inp, %axis) {reverse = 0 : index, exclusive = 0 : index} : (!mo.tensor<[5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[5], f32>
  mo.output %res : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'cumsum_axis1' returned tensor<5xf32> [-1.000000e+00, 1.000000e+00, 2.000000e+00, -2.000000e+00, 5.000000e+00]

mo.graph @cumsum_reverse(%ones : !mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>) {
  %x = mo.constant {value = #M.dense_array<-1.0, 1.0, 2.0, -2.0, 5.0> : tensor<5xf32>} : !mo.tensor<[5], f32>
  %inp = mo.mul(%ones, %x) : !mo.tensor<[5], f32>
  %axis = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %res = mo.cumsum(%inp, %axis) {reverse = 1 : index, exclusive = 0 : index} : (!mo.tensor<[5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[5], f32>
  mo.output %res : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'cumsum_reverse' returned tensor<5xf32> [5.000000e+00, 6.000000e+00, 5.000000e+00, 3.000000e+00, 5.000000e+00]

mo.graph @cumsum_exclusive(%ones : !mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>) {
  %x = mo.constant {value = #M.dense_array<-1.0, 1.0, 2.0, -2.0, 5.0> : tensor<5xf32>} : !mo.tensor<[5], f32>
  %inp = mo.mul(%ones, %x) : !mo.tensor<[5], f32>
  %axis = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %res = mo.cumsum(%inp, %axis) {reverse = 0 : index, exclusive = 1 : index} : (!mo.tensor<[5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[5], f32>
  mo.output %res : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'cumsum_exclusive' returned tensor<5xf32> [0.000000e+00, -1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00]
