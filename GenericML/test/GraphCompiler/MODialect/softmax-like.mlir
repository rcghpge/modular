// RUN: mo-opt %s | FileCheck %s

// CHECK: mo.graph @softmax(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[2, 3], f32> {
mo.graph @softmax(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[], si64>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.softmax(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 3], f32>
  %0 = mo.softmax(%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @softmax_dyn_rank<[[SH0:[^,>]*]]: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[[SH0]], f32> {
mo.graph @softmax_dyn_rank<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<[], si64>) -> (!mo.tensor<Sh0, f32>) {
  // CHECK-NEXT: mo.softmax(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH0]], f32>
  %0 = mo.softmax(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32>
  mo.output %0 : !mo.tensor<Sh0, f32>
}

// CHECK: mo.graph @logsoftmax(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[2, 3], f32> {
mo.graph @logsoftmax(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[], si64>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.logsoftmax(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 3], f32>
  %0 = mo.logsoftmax(%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @logsoftmax_dyn_rank<[[SH0:[^,>]*]]: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[[SH0]], f32> {
mo.graph @logsoftmax_dyn_rank<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<[], si64>) -> (!mo.tensor<Sh0, f32>) {
  // CHECK-NEXT: mo.logsoftmax(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH0]], f32>
  %0 = mo.logsoftmax(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32>
  mo.output %0 : !mo.tensor<Sh0, f32>
}
