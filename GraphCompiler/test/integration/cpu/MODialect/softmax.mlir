// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// TODO(GEX-1573): need to investigate

mo.graph @softmax() -> (!mo.tensor<[1, 5], f32>) {
  %0 = "mo.constant"() {value = #M.dense_array<-1.0, 2.0, -3.0, -4.0, 5.0> : tensor<1x5xf32>} : () -> !mo.tensor<[1, 5], f32>
  %1 = mo.reduce.softmax(%0) {axis = 1 : index} : (!mo.tensor<[1, 5], f32>) -> !mo.tensor<[1, 5], f32>
  mo.output %1 : !mo.tensor<[1, 5], f32>
}
// CHECK{LITERAL}: 'softmax' returned tensor<1x5xf32> [2.354606e-03, 4.729353e-02, 3.186613e-04, 1.172289e-04, 9.499159e-01]

mo.graph @softmax_3D() -> (!mo.tensor<[1, 2, 3], f32>) {
  %0 = "mo.constant"() {value = #M.dense_array<-1.0, 2.0, -3.0, -4.0, 5.0, 2.0> : tensor<1x2x3xf32>} : () -> !mo.tensor<[1, 2, 3], f32>
  %1 = mo.reduce.softmax(%0) {axis = 2 : index} : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32>
  mo.output %1 : !mo.tensor<[1, 2, 3], f32>
}
// CHECK{LITERAL}: 'softmax_3D' returned tensor<1x2x3xf32> [4.712342e-02, 9.464992e-01, 6.377461e-03, 1.175432e-04, 9.524622e-01, 4.742030e-02]

mo.graph @logsoftmax() -> (!mo.tensor<[1, 5], f32>) {
  %0 = "mo.constant"() {value = #M.dense_array<-1.0, 2.0, -3.0, -4.0, 5.0> : tensor<1x5xf32>} : () -> !mo.tensor<[1, 5], f32>
  %1 = mo.reduce.logsoftmax(%0) {axis = 1 : index} : (!mo.tensor<[1, 5], f32>) -> !mo.tensor<[1, 5], f32>
  mo.output %1 : !mo.tensor<[1, 5], f32>
}
// CHECK{LITERAL}: 'logsoftmax' returned tensor<1x5xf32> [-6.051382e+00, -3.051382e+00, -8.051382e+00, -9.051382e+00, -5.138177e-02]
