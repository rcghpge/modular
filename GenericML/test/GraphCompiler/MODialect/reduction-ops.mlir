// RUN: mo-opt %s | FileCheck %s

// CHECK: mo.graph @mean(%[[INPUT:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32> {
mo.graph @mean(%input: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 1], f32>) {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: mo.mean(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
  %output = mo.mean(%input, %axis) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
  mo.output %output : !mo.tensor<[2, 1], f32>
}

// CHECK: mo.graph @prod(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32> {
mo.graph @prod(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: mo.prod(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  %output = mo.prod(%input, %axis) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}

// CHECK: mo.graph @sum(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32> {
mo.graph @sum(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: mo.sum(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  %output = mo.sum(%input, %axis) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}
