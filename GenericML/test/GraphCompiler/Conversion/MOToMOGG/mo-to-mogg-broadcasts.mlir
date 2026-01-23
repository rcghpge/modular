// RUN: mo-opt %s --mo-to-mogg | FileCheck %s

mo.graph @bcast_basic<N, L, M, K>(%arg0: !mo.tensor<[N, M, L], f32>) -> (!mo.tensor<[N, M, K], f32>) {
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[N, M, L], f32> -> !mo.tensor<[N, M, K], f32>
  mo.output %0: !mo.tensor<[N, M, K], f32>
}

// CHECK-LABEL: mo.graph @bcast_basic
// CHECK-NEXT: %[[KERNEL:.*]] = mogg.kernel(%[[IN1:.*]]: !mo.tensor<[N, M, L], f32>) -> !mo.tensor<[N, M, K], f32>
// CHECK: %[[OUT:.*]] = mogg.output_placeholders : !mo.tensor<[N, M, K], f32>
// CHECK: mogg.tensor.shape
// CHECK: mogg.call.view["mo.static.broadcast_to"]
// CHECK: mogg.call.materialize
// CHECK: mo.output %[[KERNEL]] : !mo.tensor<[N, M, K], f32>
