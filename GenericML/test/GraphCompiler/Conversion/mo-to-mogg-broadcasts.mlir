// RUN: mo-opt %s --mo-to-mogg=useJit=true | FileCheck %s

mo.graph @bcast_basic(%arg0: !mo.tensor<[?, ?, ?], f32>, %arg1: !mo.tensor<[?], f32>) -> (!mo.tensor<[?, ?, ?], f32>) {
  %0 = mogg.broadcast_to_tensor(%arg0, %arg1): (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?], f32>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %0: !mo.tensor<[?, ?, ?], f32>
}

// CHECK: mo.graph @bcast_basic
// CHECK-NEXT: %[[KERNEL:.*]] = mogg.kernel (%[[NULL:.*]], %[[NULL:.*]]) : (%[[IN1:.*]]: !mo.tensor<[?, ?, ?], f32>, %[[IN2:.*]]: !mo.tensor<[?], f32>) -> !mo.tensor<[?, ?, ?], f32>{

// We broadcast the tensor.
// CHECK-NEXT: %[[BCAST:.*]] = mogg.call "mogg.broadcast_to_tensor"(%[[IN1]], %[[IN2]]) : !mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?], f32> -> !mo.tensor<[?, ?, ?], f32>
// CHECK-NEXT: %[[OUT:.*]] = mogg.output_placeholders : !mo.tensor<[?, ?, ?], f32>

// Load the first argument for use in the copy. Loads the broadcasted view.
// CHECK-NEXT:  mogg.lambda @input_0_fn = (%[[INDICES:.*]]: !mogg.indices) -> !mogg.simd<f32>{
// CHECK-NEXT:    %[[LOAD:.*]] = mogg.simd_load(%[[BCAST]], %[[INDICES]]) : (!mo.tensor<[?, ?, ?], f32>, !mogg.indices) -> !mogg.simd<f32>
// CHECK-NEXT:    mogg.output %[[LOAD]] : !mogg.simd<f32>
// CHECK-NEXT:  }

// The second input is unused, but is consistent with the other lowerings.
// CHECK-NEXT:  mogg.lambda @input_1_fn = (%[[INDICES:.*]]: !mogg.indices) -> !mogg.simd<f32>{
// CHECK-NEXT:    %[[LOAD:.*]] = mogg.simd_load(%[[IN2]], %[[INDICES]]) : (!mo.tensor<[?], f32>, !mogg.indices) -> !mogg.simd<f32>
// CHECK-NEXT:    mogg.output %[[LOAD]] : !mogg.simd<f32>
// CHECK-NEXT:  }

// Store to the output for the copy.
// CHECK-NEXT:  mogg.lambda @output_0_fn = (%[[INDICES:.*]]: !mogg.indices, %[[VAL:.*]]: !mogg.simd<f32>){
// CHECK-NEXT:      mogg.call "simd_store"(%[[OUT]], %[[INDICES]], %[[VAL]]) : !mo.tensor<[?, ?, ?], f32>, !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:      mogg.output
// CHECK-NEXT:  }

// The copy is implemented with the elementwise kernel as before.
// CHECK-NEXT:  mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices){
// CHECK-NEXT:    %[[COPY:.*]] = mogg.invoke @input_0_fn(%[[INDICES]]) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
// CHECK-NEXT:    mogg.invoke @output_0_fn(%[[INDICES]], %[[COPY]]) : !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:    mogg.output
// CHECK-NEXT:  }

// Invoke the elementwise copy.
// CHECK-NEXT:  mogg.call "elementwise" [@elemwise_compute_lambda](%[[OUT]]) : !mo.tensor<[?, ?, ?], f32>
// CHECK-NEXT:  mogg.output %[[OUT]] : !mo.tensor<[?, ?, ?], f32>

// No fusion interfaces but it does have the indcies of the inputs which are broadcast and the isView marking.
// CHECK-NEXT: } {bcastInputIndices = [0 : i32, 1 : i32], isViewOp = true, shapeFunc = "broadcast_to_tensor"}
// CHECK-NEXT: mo.output %[[KERNEL]] : !mo.tensor<[?, ?, ?], f32>

