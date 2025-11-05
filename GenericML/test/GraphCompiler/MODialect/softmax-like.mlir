// RUN: mo-opt %s | FileCheck %s

// CHECK: mo.graph @softmax(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[2, 3], f32> {
mo.graph @softmax(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[], si64>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.softmax(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 3], f32>
  %0 = mo.softmax(%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @softmax_dyn_rank(%[[ARG0:.*]]: !mo.tensor<?, f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<?, f32> {
mo.graph @softmax_dyn_rank(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<[], si64>) -> (!mo.tensor<?, f32>) {
  // CHECK-NEXT: mo.softmax(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<?, f32>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  %0 = mo.softmax(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// CHECK: mo.graph @logsoftmax(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[2, 3], f32> {
mo.graph @logsoftmax(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[], si64>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.logsoftmax(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 3], f32>
  %0 = mo.logsoftmax(%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @logsoftmax_dyn_rank(%[[ARG0:.*]]: !mo.tensor<?, f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<?, f32> {
mo.graph @logsoftmax_dyn_rank(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<[], si64>) -> (!mo.tensor<?, f32>) {
  // CHECK-NEXT: mo.logsoftmax(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<?, f32>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  %0 = mo.logsoftmax(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
