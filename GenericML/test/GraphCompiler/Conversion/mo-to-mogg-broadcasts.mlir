// RUN: mo-opt %s --mo-to-mogg | FileCheck %s

mo.graph @bcast_basic<N, L, M, K>(%arg0: !mo.tensor<[N, M, L], f32>) -> (!mo.tensor<[N, M, K], f32>) {
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[N, M, L], f32> -> !mo.tensor<[N, M, K], f32>
  mo.output %0: !mo.tensor<[N, M, K], f32>
}

// CHECK-LABEL: mo.graph @bcast_basic
// CHECK-NEXT: %[[KERNEL:.*]] = mogg.kernel (%[[NULL:.*]]) : (%[[IN1:.*]]: !mo.tensor<[N, M, L], f32>) -> !mo.tensor<[N, M, K], f32>{
// CHECK-NEXT: %[[OUT:.*]] = mogg.output_placeholders : !mo.tensor<[N, M, K], f32>

// We broadcast the tensor.
// CHECK-NEXT: %[[SHAPE:.*]] = mogg.param_shape_to_value -> !mogg.shape<[N, M, K]>
// CHECK-NEXT: %[[BCAST:.*]] = mogg.call "mo.static.broadcast_to"(%[[IN1]], %[[SHAPE]]) {single_thread_blocking_override = false} : !mo.tensor<[N, M, L], f32>, !mogg.shape<[N, M, K]> -> !mo.tensor<[N, M, K], f32>

// The copy is implemented with the elementwise kernel as before.
// CHECK-NEXT:  mogg.lambda @copy_view = (%[[INDICES:.*]]: !mogg.indices){
// CHECK-NEXT:    %[[LOAD:.*]] = mogg.call "simd_load"(%[[BCAST]], %[[INDICES]]) : !mo.tensor<[N, M, K], f32>, !mogg.indices -> !mogg.simd<f32>
// CHECK-NEXT:    mogg.call "simd_store"(%[[OUT]], %[[INDICES]], %[[LOAD]]) : !mo.tensor<[N, M, K], f32>, !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:    mogg.output
// CHECK-NEXT:  }

// Invoke the elementwise copy.
// CHECK-NEXT:  mogg.call "elementwise" [@copy_view](%[[OUT]]) {single_thread_blocking_override = false} : !mo.tensor<[N, M, K], f32>
// CHECK-NEXT:  mogg.output %[[OUT]] : !mo.tensor<[N, M, K], f32>

// No fusion interfaces but it does have the indcies of the inputs which are broadcast and the isView marking.
// CHECK-NEXT: } {isAsync = true, isViewOp = true, summary = "mo.static.broadcast_to{{.*}}"}
// CHECK-NEXT: mo.output %[[KERNEL]] : !mo.tensor<[N, M, K], f32>
