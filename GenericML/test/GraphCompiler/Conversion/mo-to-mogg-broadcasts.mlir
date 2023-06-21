// RUN: mo-opt %s --mo-to-mogg=use-jit=true | FileCheck %s

mo.graph @bcast_basic<N, L, M, K>(%arg0: !mo.tensor<[N, M, L], f32>) -> (!mo.tensor<[N, M, K], f32>) {
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[N, M, L], f32> -> !mo.tensor<[N, M, K], f32>
  mo.output %0: !mo.tensor<[N, M, K], f32>
}

// CHECK-LABEL: mo.graph @bcast_basic
// CHECK-NEXT: %[[KERNEL:.*]] = mogg.kernel (%[[NULL:.*]]) : (%[[IN1:.*]]: !mo.tensor<[N, M, L], f32>) -> !mo.tensor<[N, M, K], f32>{
// CHECK-NEXT: %[[OUT:.*]] = mogg.output_placeholders : !mo.tensor<[N, M, K], f32>

// We broadcast the tensor.
// CHECK-NEXT: %[[BCAST:.*]] = mogg.call "mo.static.broadcast_to"(%[[IN1]], %[[OUT]]) {single_thread_blocking_override = false} : !mo.tensor<[N, M, L], f32>, !mo.tensor<[N, M, K], f32> -> !mo.tensor<[N, M, K], f32>

// Load the first argument for use in the copy. Loads the broadcasted view.
// CHECK-NEXT:  mogg.lambda @input_0_fn = (%[[INDICES:.*]]: !mogg.indices) -> !mogg.simd<f32>{
// CHECK-NEXT:    %[[LOAD:.*]] = mogg.simd_load(%[[BCAST]], %[[INDICES]]) : (!mo.tensor<[N, M, K], f32>, !mogg.indices) -> !mogg.simd<f32>
// CHECK-NEXT:    mogg.output %[[LOAD]] : !mogg.simd<f32>
// CHECK-NEXT:  }

// Store to the output for the copy.
// CHECK-NEXT:  mogg.lambda @output_0_fn = (%[[INDICES:.*]]: !mogg.indices, %[[VAL:.*]]: !mogg.simd<f32>){
// CHECK-NEXT:      mogg.call "simd_store"(%[[OUT]], %[[INDICES]], %[[VAL]]) : !mo.tensor<[N, M, K], f32>, !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:      mogg.output
// CHECK-NEXT:  }

// The copy is implemented with the elementwise kernel as before.
// CHECK-NEXT:  mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices){
// CHECK-NEXT:    %[[COPY:.*]] = mogg.invoke @input_0_fn(%[[INDICES]]) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
// CHECK-NEXT:    mogg.invoke @output_0_fn(%[[INDICES]], %[[COPY]]) : !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:    mogg.output
// CHECK-NEXT:  }

// Invoke the elementwise copy.
// CHECK-NEXT:  mogg.call "elementwise" [@elemwise_compute_lambda](%[[OUT]]) {single_thread_blocking_override = false} : !mo.tensor<[N, M, K], f32>
// CHECK-NEXT:  mogg.output %[[OUT]] : !mo.tensor<[N, M, K], f32>

// No fusion interfaces but it does have the indcies of the inputs which are broadcast and the isView marking.
// CHECK-NEXT: } {isViewOp = true}
// CHECK-NEXT: mo.output %[[KERNEL]] : !mo.tensor<[N, M, K], f32>
