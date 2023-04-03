// RUN: mo-opt --mo-to-mogg=use-jit=true --fuse-elementwise --canonicalize --cse --fuse-bcasts --canonicalize --cse %s | FileCheck %s

mo.graph @basicBroadcastFuse(%arg0: !mo.tensor<[?, ?], f32>, %arg1: !mo.tensor<[?, ?], f32>) -> (!mo.tensor<[?, ?], f32>) {
  %bcast1 = mogg.broadcast_to_tensor(%arg0, %arg1) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], f32>
  %bcast2 = mogg.broadcast_to_tensor(%arg1, %arg0) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], f32>
  %1 = mo.add(%bcast1, %bcast2) : !mo.tensor<[?, ?], f32>
  mo.output %1 : !mo.tensor<[?, ?], f32>
}

// CHECK:  mo.graph @basicBroadcastFuse(%[[ARG0:.*]]: !mo.tensor<[?, ?], f32>, %[[ARG1:.*]]: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], f32> {
// CHECK-NEXT:   %[[KERNEL:.*]] = mogg.kernel (%[[ARG1]], %[[ARG0]]) : (%[[IN1:.*]]: !mo.tensor<[?, ?], f32>, %[[IN2:.*]]: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], f32>{
// CHECK-NEXT:   %[[OUT:.*]] = mogg.output_placeholders : !mo.tensor<[?, ?], f32>
// CHECK-NEXT:   %[[BCAST_1:.*]] = mogg.call "mogg.broadcast_to_tensor"(%[[IN2]], %[[IN1]]) : !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32> -> !mo.tensor<[?, ?], f32>
// CHECK-NEXT:   %[[BCAST_2:.*]] = mogg.call "mogg.broadcast_to_tensor"(%[[IN1]], %[[IN2]]) : !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32> -> !mo.tensor<[?, ?], f32>
// CHECK-NEXT:   mogg.lambda @input_0_fn = (%[[INDICES:.*]]: !mogg.indices) -> !mogg.simd<f32>{
// CHECK-NEXT:      %[[LOAD:.*]] = mogg.simd_load(%[[BCAST_1]], %[[INDICES]]) : (!mo.tensor<[?, ?], f32>, !mogg.indices) -> !mogg.simd<f32>
// CHECK-NEXT:      mogg.output %[[LOAD]] : !mogg.simd<f32>
// CHECK-NEXT:   }
// CHECK-NEXT:   mogg.lambda @input_1_fn = (%[[INDICES:.*]]: !mogg.indices) -> !mogg.simd<f32>{
// CHECK-NEXT:      %[[LOAD:.*]] = mogg.simd_load(%[[BCAST_2]], %[[INDICES]]) : (!mo.tensor<[?, ?], f32>, !mogg.indices) -> !mogg.simd<f32>
// CHECK-NEXT:      mogg.output %[[LOAD]] : !mogg.simd<f32>
// CHECK-NEXT:   }
// CHECK-NEXT:   mogg.lambda @output_0_fn = (%[[INDICES:.*]]: !mogg.indices, %[[VALUE:.*]]: !mogg.simd<f32>){
// CHECK-NEXT:      mogg.call "simd_store"(%[[OUT]], %[[INDICES]], %[[VALUE]]) : !mo.tensor<[?, ?], f32>, !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:      mogg.output
// CHECK-NEXT:   }
// CHECK-NEXT:   mogg.lambda @elemwise_compute_lambda = (%[[INDICES]]: !mogg.indices){
// CHECK-NEXT:      %[[LOAD_IN1:.*]] = mogg.invoke @input_0_fn(%[[INDICES]]) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
// CHECK-NEXT:      %[[LOAD_IN2:.*]] = mogg.invoke @input_1_fn(%[[INDICES]]) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
// CHECK-NEXT:      %[[ADD:.*]] = mogg.call "mo.add"(%[[LOAD_IN1]], %[[LOAD_IN2]]) : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:      mogg.invoke @output_0_fn(%[[INDICES]], %[[ADD]]) : !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:      mogg.output
// CHECK-NEXT:   }
// CHECK-NEXT:   mogg.call "elementwise" [@elemwise_compute_lambda](%[[OUT]]) : !mo.tensor<[?, ?], f32>
// CHECK-NEXT:   mogg.output %[[OUT]] : !mo.tensor<[?, ?], f32>

// Check the attributes have been stripped so this can no longer be fused and that
// the broadcast attributes have been attached correctly.
// CHECK-NEXT:   } {bcastInputIndices = [1 : i32, 0 : i32], shapeFunc = "broadcast_to_tensor"}
// CHECK-NEXT:   mo.output %[[KERNEL]] : !mo.tensor<[?, ?], f32>



mo.graph @basicBroadcastIntoElemwise(%arg0: !mo.tensor<[?, ?, ?, ?], f32>, %arg1: !mo.tensor<[?, ?], f32>) -> (!mo.tensor<[?, ?, ?, ?], f32>) {
  %bcast1 = mogg.broadcast_to_tensor(%arg0, %arg1) : (!mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?, ?, ?], f32>
  %bcast2 = mogg.broadcast_to_tensor(%arg1, %arg0) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?, ?], f32>
  %1 = mo.add(%bcast1, %bcast2) : !mo.tensor<[?, ?, ?, ?], f32>
  %2 = mo.sqrt(%1) : !mo.tensor<[?, ?, ?, ?], f32>
  %3 = mo.tanh(%1) : !mo.tensor<[?, ?, ?, ?], f32>
  %4 = mo.mul(%2, %3) : !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %4 : !mo.tensor<[?, ?, ?, ?], f32>
}

// CHECK:  mo.graph @basicBroadcastIntoElemwise(%[[ARG0:.*]]: !mo.tensor<[?, ?, ?, ?], f32>, %[[ARG1:.*]]: !mo.tensor<[?, ?], f32>)
// CHECK-NEXT:   %[[KERNEL:.*]] = mogg.kernel (%[[ARG1]], %[[ARG0]]) : (%[[IN1:.*]]: !mo.tensor<[?, ?], f32>, %[[IN2:.*]]: !mo.tensor<[?, ?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?, ?], f32>
// CHECK-NEXT:   mogg.output_placeholders
// CHECK-NEXT:   %[[BCAST_1:.*]] = mogg.call "mogg.broadcast_to_tensor"(%[[IN2]], %[[IN1]]) : !mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[?, ?], f32> -> !mo.tensor<[?, ?, ?, ?], f32>
// CHECK-NEXT:   %[[BCAST_2:.*]] = mogg.call "mogg.broadcast_to_tensor"(%[[IN1]], %[[IN2]]) : !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?, ?, ?], f32> -> !mo.tensor<[?, ?, ?, ?], f32>

// Check we load from the bcasts
// CHECK: mogg.lambda @input_0_fn = (%[[INDICES:.*]]: !mogg.indices)
// CHECK-NEXT:  mogg.simd_load(%[[BCAST_1]], %[[INDICES]]) : (!mo.tensor<[?, ?, ?, ?], f32>, !mogg.indices) -> !mogg.simd<f32>

// CHECK:mogg.lambda @input_1_fn = (%[[INDICES:.*]]: !mogg.indices)
// CHECK-NEXT:      mogg.simd_load(%[[BCAST_2]], %[[INDICES]]) : (!mo.tensor<[?, ?, ?, ?], f32>, !mogg.indices) -> !mogg.simd<f32>

// Check the elementwise was still fused.
// CHECK:   mogg.lambda @elemwise_compute_lambda = (%[[INDICES]]: !mogg.indices)
// CHECK-NEXT:      %[[LOAD_IN1:.*]] = mogg.invoke @input_0_fn(%[[INDICES]])
// CHECK-NEXT:      %[[LOAD_IN2:.*]] = mogg.invoke @input_1_fn(%[[INDICES]])
// CHECK-NEXT:      %[[ADD:.*]] = mogg.call "mo.add"(%[[LOAD_IN1]], %[[LOAD_IN2]])
// CHECK-NEXT:      %[[SQRT:.*]] = mogg.call "mo.sqrt"(%[[ADD]])
// CHECK-NEXT:      %[[TANH:.*]] = mogg.call "mo.tanh"(%[[ADD]])
// CHECK-NEXT:      %[[MUL:.*]] = mogg.call "mo.mul"(%[[SQRT]], %[[TANH]])
// CHECK-NEXT:      mogg.invoke @output_0_fn(%[[INDICES]], %[[MUL]]) : !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:      mogg.output


// We currently only support fusing one "layer" of operands
mo.graph @checkOneLayerFusion(%arg0: !mo.tensor<[?], f32>, %arg1: !mo.tensor<[?, ?, ?, ?], f32>, %arg2: !mo.tensor<[?, ?, ?], f32>) -> (!mo.tensor<[?, ?, ?, ?], f32>) {
  %bcast_base = mogg.broadcast_to_tensor(%arg0, %arg2) : (!mo.tensor<[?], f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?], f32>

  %bcast1 = mogg.broadcast_to_tensor(%bcast_base, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?, ?], f32>
  %bcast2 = mogg.broadcast_to_tensor(%arg1, %bcast_base) : (!mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?, ?], f32>
  %1 = mo.div(%bcast1, %bcast2) : !mo.tensor<[?, ?, ?, ?], f32>
  %2 = mo.sqrt(%1) : !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %2 : !mo.tensor<[?, ?, ?, ?], f32>
}


// Check we produce to kernels. One unfused kernel feeding into the fused kernel.
// CHECK: mo.graph @checkOneLayerFusion(%[[ARG0:.*]]: !mo.tensor<[?], f32>, %[[ARG1:.*]]: !mo.tensor<[?, ?, ?, ?], f32>, %[[ARG2:.*]]: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?, ?], f32>

// CHECK-NEXT:%[[KERNEL1:.*]] = mogg.kernel (%[[ARG0]], %[[ARG2]]) : (%[[IN1:.*]]: !mo.tensor<[?], f32>, %[[IN2:.*]]: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?], f32>
// CHECK-NEXT: mogg.output_placeholders

// Find the first unfused kernel.
// CHECK-NEXT:      %[[BCAST:.*]] = mogg.call "mogg.broadcast_to_tensor"(%[[IN1]], %[[IN2]]) : !mo.tensor<[?], f32>, !mo.tensor<[?, ?, ?], f32> -> !mo.tensor<[?, ?, ?], f32>
// CHECK:           mogg.lambda @input_0_fn = (%[[INDICES:.*]]: !mogg.indices)
// CHECK-NEXT:           %[[LOAD:.*]] = mogg.simd_load(%[[BCAST]], %[[INDICES]])
// CHECK-NEXT:           mogg.output %[[LOAD]]
// CHECK:           mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices)
// CHECK-NEXT:           %[[LOAD:.*]] = mogg.invoke @input_0_fn(%[[INDICES]])
// CHECK-NEXT:           mogg.invoke @output_0_fn(%[[INDICES]], %[[LOAD]])

// Check the second fused kernel.


// CHECK: %[[KERNEL2:.*]] = mogg.kernel (%[[ARG1]], %[[KERNEL1]]) : (%[[IN1:.*]]: !mo.tensor<[?, ?, ?, ?], f32>, %[[IN2:.*]]: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?, ?], f32>
// CHECK-NEXT: mogg.output_placeholders
// CHECK-NEXT:      %[[BCAST1:.*]] = mogg.call "mogg.broadcast_to_tensor"(%[[IN2]], %[[IN1]]) : !mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?, ?], f32> -> !mo.tensor<[?, ?, ?, ?], f32>
// CHECK-NEXT:      %[[BCAST2:.*]] = mogg.call "mogg.broadcast_to_tensor"(%[[IN1]], %[[IN2]]) : !mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32> -> !mo.tensor<[?, ?, ?, ?], f32>

// Find the broadcast loads.
// CHECK:           mogg.lambda @input_0_fn_0 = (%[[INDICES:.*]]: !mogg.indices)
// CHECK-NEXT:           %[[LOAD:.*]] = mogg.simd_load(%[[BCAST1]], %[[INDICES]])
// CHECK-NEXT:           mogg.output %[[LOAD]]
// CHECK:           mogg.lambda @input_1_fn = (%[[INDICES:.*]]: !mogg.indices)
// CHECK-NEXT:           %[[LOAD:.*]] = mogg.simd_load(%[[BCAST2]], %[[INDICES]])
// CHECK-NEXT:           mogg.output %[[LOAD]]

// The fused body
// CHECK:           mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices)
// CHECK-NEXT:           %[[LOAD1:.*]] = mogg.invoke @input_0_fn_0(%[[INDICES]])
// CHECK-NEXT:           %[[LOAD2:.*]] = mogg.invoke @input_1_fn(%[[INDICES]])
// CHECK-NEXT:           %[[DIV:.*]] = mogg.call "mo.div"(%[[LOAD1]], %[[LOAD2]])
// CHECK-NEXT:           %[[SQRT:.*]] = mogg.call "mo.sqrt"(%[[DIV]])
// CHECK-NEXT:           mogg.invoke @output_0_fn(%[[INDICES]], %[[SQRT]])


// Weird case from mask-rcnn where the broadcast is used by another broadcast and both are then used by an op.
mo.graph @bcastUsesRHS(%arg0: !mo.tensor<[?], f32>, %arg1: !mo.tensor<[?, ?, ?, ?], f32>, %arg2: !mo.tensor<[?, ?], f32>) -> (!mo.tensor<[?, ?, ?, ?], f32>) {
  %bcast_base = mogg.broadcast_to_tensor(%arg0, %arg1) : (!mo.tensor<[?], f32>, !mo.tensor<[?, ?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?, ?], f32>
  %bcast1 = mogg.broadcast_to_tensor(%arg2, %bcast_base) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?, ?], f32>
  %1 = mo.sub(%bcast1, %bcast_base) : !mo.tensor<[?, ?, ?, ?], f32>
  %2 = mo.rsqrt(%1) : !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %2 : !mo.tensor<[?, ?, ?, ?], f32>
}


// CHECK: mo.graph @bcastUsesRHS(%[[ARG0:.*]]: !mo.tensor<[?], f32>, %[[ARG1:.*]]: !mo.tensor<[?, ?, ?, ?], f32>, %[[ARG2:.*]]: !mo.tensor<[?, ?], f32>)
// CHECK-NEXT: %[[KERNEL_OUT:.*]] = mogg.kernel (%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (%[[IN1:.*]]: !mo.tensor<[?], f32>, %[[IN2:.*]]: !mo.tensor<[?, ?, ?, ?], f32>, %[[IN3:.*]]: !mo.tensor<[?, ?], f32>)
// CHECK-NEXT: mogg.output_placeholders
// CHECK-NEXT: %[[BCAST_1:.*]] = mogg.call "mogg.broadcast_to_tensor"(%[[IN1]], %[[IN2]])
// CHECK-NEXT: %[[BCAST_2:.*]] = mogg.call "mogg.broadcast_to_tensor"(%[[IN3]], %[[BCAST_1]])

// CHECK: mogg.lambda @input_0_fn_0 = (%[[INDICES:.*]]: !mogg.indices)
// CHECK-NEXT:  mogg.simd_load(%[[BCAST_2]], %[[INDICES]]) : (!mo.tensor<[?, ?, ?, ?], f32>, !mogg.indices) -> !mogg.simd<f32>
// CHECK:mogg.lambda @input_1_fn = (%[[INDICES:.*]]: !mogg.indices)
// CHECK-NEXT:      mogg.simd_load(%[[BCAST_1]], %[[INDICES]]) : (!mo.tensor<[?, ?, ?, ?], f32>, !mogg.indices) -> !mogg.simd<f32>

// CHECK:           mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices)
// CHECK-NEXT:           %[[LOAD1:.*]] = mogg.invoke @input_0_fn_0(%[[INDICES]])
// CHECK-NEXT:           %[[LOAD2:.*]] = mogg.invoke @input_1_fn(%[[INDICES]])
// CHECK-NEXT:           %[[SUB:.*]] = mogg.call "mo.sub"(%[[LOAD1]], %[[LOAD2]])
// CHECK-NEXT:           %[[RSQRT:.*]] = mogg.call "mo.rsqrt"(%[[SUB]])
// CHECK-NEXT:           mogg.invoke @output_0_fn(%[[INDICES]], %[[RSQRT]])



// Weird case from mask-rcnn where the broadcast is used by another broadcast and both are then used by an op.
mo.graph @bcastUsesLHS(%arg0: !mo.tensor<[?], f32>, %arg1: !mo.tensor<[?, ?, ?, ?], f32>, %arg2: !mo.tensor<[?, ?], f32>) -> (!mo.tensor<[?, ?, ?, ?], f32>) {
  %bcast_base = mogg.broadcast_to_tensor(%arg0, %arg1) : (!mo.tensor<[?], f32>, !mo.tensor<[?, ?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?, ?], f32>
  %bcast1 = mogg.broadcast_to_tensor(%arg2, %bcast_base) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?, ?], f32>
  %1 = mo.mul(%bcast_base, %bcast1) : !mo.tensor<[?, ?, ?, ?], f32>
  %2 = mo.relu(%1) : !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %2 : !mo.tensor<[?, ?, ?, ?], f32>
}

// CHECK: mo.graph @bcastUsesLHS(%[[ARG0:.*]]: !mo.tensor<[?], f32>, %[[ARG1:.*]]: !mo.tensor<[?, ?, ?, ?], f32>, %[[ARG2:.*]]: !mo.tensor<[?, ?], f32>)
// CHECK-NEXT: %[[KERNEL_OUT:.*]] = mogg.kernel (%[[ARG2]], %[[ARG0]], %[[ARG1]]) : (%[[IN1:.*]]: !mo.tensor<[?, ?], f32>, %[[IN2:.*]]: !mo.tensor<[?], f32>, %[[IN3:.*]]: !mo.tensor<[?, ?, ?, ?], f32>)
// CHECK-NEXT: mogg.output_placeholders
// CHECK-NEXT: %[[BCAST_1:.*]] = mogg.call "mogg.broadcast_to_tensor"(%[[IN2]], %[[IN3]])
// CHECK-NEXT: %[[BCAST_2:.*]] = mogg.call "mogg.broadcast_to_tensor"(%[[IN1]], %[[BCAST_1]])

// CHECK: mogg.lambda @input_0_fn_0 = (%[[INDICES:.*]]: !mogg.indices)
// CHECK-NEXT:  mogg.simd_load(%[[BCAST_1]], %[[INDICES]]) : (!mo.tensor<[?, ?, ?, ?], f32>, !mogg.indices) -> !mogg.simd<f32>
// CHECK:mogg.lambda @input_1_fn = (%[[INDICES:.*]]: !mogg.indices)
// CHECK-NEXT:      mogg.simd_load(%[[BCAST_2]], %[[INDICES]]) : (!mo.tensor<[?, ?, ?, ?], f32>, !mogg.indices) -> !mogg.simd<f32>

// CHECK:           mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices)
// CHECK-NEXT:           %[[LOAD1:.*]] = mogg.invoke @input_0_fn_0(%[[INDICES]])
// CHECK-NEXT:           %[[LOAD2:.*]] = mogg.invoke @input_1_fn(%[[INDICES]])
// CHECK-NEXT:           %[[SUB:.*]] = mogg.call "mo.mul"(%[[LOAD1]], %[[LOAD2]])
// CHECK-NEXT:           %[[RSQRT:.*]] = mogg.call "mo.relu"(%[[SUB]])
// CHECK-NEXT:           mogg.invoke @output_0_fn(%[[INDICES]], %[[RSQRT]])


// Check we produce one kernel. The mo primitive tests check this kernel so we don't
// need to check that thoroughly just check one kernel is produced.
mo.graph @many_bcasts() -> !mo.tensor<[2, 2, 3], f32> {
  %0 = mo.constant {value = #M.dense_array<1.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 10.000000e+00, 11.000000e+00> : tensor<12xf32>} : !mo.tensor<[2, 2, 3], f32>
  %1 = mo.constant {value = #M.dense_array<2.000000e+00, 0.000000e+00> : tensor<2xf32>} : !mo.tensor<[2, 1, 1], f32>
  %2 = mo.constant {value = #M.dense_array<5.000000e-01, 2.500000e-01> : tensor<2xf32>} : !mo.tensor<[2, 1], f32>
  %3 = mo.constant {value = #M.dense_array<1.000000e+00, 7.000000e+00, 8.000000e+00> : tensor<3xf32>} : !mo.tensor<[3], f32>

  %bcast = mogg.broadcast_to_tensor(%1, %0) : (!mo.tensor<[2, 1, 1], f32>, !mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 2, 3], f32>
  %bcast2 = mogg.broadcast_to_tensor(%2, %0) : (!mo.tensor<[2, 1], f32>, !mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 2, 3], f32>
  %bcast3 = mogg.broadcast_to_tensor(%3, %0) : (!mo.tensor<[3], f32>, !mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 2, 3], f32>

  %out = mo.add(%bcast, %0) : !mo.tensor<[2, 2, 3], f32>
  %out2 = mo.mul(%bcast2, %out) : !mo.tensor<[2, 2, 3], f32>
  %out3 = mo.sub(%out2, %bcast3) : !mo.tensor<[2, 2, 3], f32>

  mo.output %out3 : !mo.tensor<[2, 2, 3], f32>
}


// CHECK: mogg.kernel
// CHECK-NEXT: mogg.output_placeholders
// CHECK-NEXT:  %[[BCAST_1:.*]] = mogg.call "mogg.broadcast_to_tensor"
// CHECK-NEXT:  %[[BCAST_2:.*]] = mogg.call "mogg.broadcast_to_tensor"
// CHECK-NEXT:  %[[BCAST_3:.*]] = mogg.call "mogg.broadcast_to_tensor"

// CHECK: mogg.lambda @elemwise_compute_lambda =
// CHECK-NEXT:  %[[LOAD1:.*]] = mogg.invoke @input_0_fn_0
// CHECK-NEXT:  %[[LOAD2:.*]] = mogg.invoke @input_0_fn
// CHECK-NEXT:  %[[LOAD3:.*]] = mogg.invoke @input_1_fn_0
// CHECK-NEXT:  %[[ADD:.*]] = mogg.call "mo.add"(%[[LOAD2]], %[[LOAD3]])
// CHECK-NEXT:  %[[MUL:.*]] = mogg.call "mo.mul"(%[[LOAD1]], %[[ADD]])
// CHECK-NEXT:  %[[LOAD4:.*]] = mogg.invoke @input_1_fn
// CHECK-NEXT:  %[[SUB:.*]] = mogg.call "mo.sub"(%[[MUL]], %[[LOAD4]])


mo.graph @check_no_fuse_if(%cond : !mo.tensor<[], bool>, %arg0: !mo.tensor<[?, ?], f32>, %arg1: !mo.tensor<[?, ?], f32>) -> (!mo.tensor<[?, ?], f32>) {
  %bcast1 = mogg.broadcast_to_tensor(%arg0, %arg1) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], f32>
  %bcast2 = mogg.broadcast_to_tensor(%arg1, %arg0) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], f32>

  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[?, ?], f32> {
    %1 = mo.add(%bcast1, %bcast2) : !mo.tensor<[?, ?], f32>
    mo.yield %1 : !mo.tensor<[?, ?], f32>
  } else {
    %2 = mo.mul(%bcast1, %bcast2) : !mo.tensor<[?, ?], f32>
    mo.yield %2 : !mo.tensor<[?, ?], f32>
  }
  mo.output %0 : !mo.tensor<[?, ?], f32>
}


// CHECK: mo.graph @check_no_fuse_if

// Check the bcasts still become standalone kernels
// CHECK-NEXT: %[[BCAST1:.*]] = mogg.kernel
// CHECK-NEXT: mogg.output_placeholders
// CHECK-NEXT:  mogg.call "mogg.broadcast_to_tensor"
// CHECK:  mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices)
// CHECK-NEXT:  %[[LOAD:.*]] = mogg.invoke @input_0_fn
// CHECK-NEXT:  mogg.invoke @output_0_fn(%arg5, %5)


// CHECK: %[[BCAST2:.*]] = mogg.kernel
// CHECK-NEXT: mogg.output_placeholders
// CHECK-NEXT:  mogg.call "mogg.broadcast_to_tensor"
// CHECK:  mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices)
// CHECK-NEXT:  %[[LOAD:.*]] = mogg.invoke @input_0_fn
// CHECK-NEXT:  mogg.invoke @output_0_fn(%arg5, %5)


// CHECK: mo.if
// CHECK-NEXT: mogg.ext.op "mop.kernels.add"(%[[BCAST1]], %[[BCAST2]])
// CHECK-NEXT: mo.yield
// CHECK-NEXT: } else {
// CHECK-NEXT: mogg.ext.op "mop.kernels.mul"(%[[BCAST1]], %[[BCAST2]])
// CHECK-NEXT: mo.yield
