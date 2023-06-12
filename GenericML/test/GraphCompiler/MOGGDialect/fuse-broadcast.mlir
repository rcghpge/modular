// RUN: mo-opt --mo-to-mogg=use-jit=true --fuse-elementwise --canonicalize --cse --fuse-bcasts --canonicalize --cse %s | FileCheck %s

mo.graph @basicBroadcastFuse<N, M, K>(%arg0: !mo.tensor<[N, M], f32>, %arg1: !mo.tensor<[K, M], f32>) -> (!mo.tensor<[K, M], f32>) {
  %bcast1 = mo.static.broadcast_to(%arg0) : !mo.tensor<[N, M], f32> -> !mo.tensor<[K, M], f32>
  %bcast2 = mo.static.broadcast_to(%arg1) : !mo.tensor<[K, M], f32> -> !mo.tensor<[K, M], f32>
  %1 = mo.add(%bcast1, %bcast2) : !mo.tensor<[K, M], f32>
  mo.output %1 : !mo.tensor<[K, M], f32>
}

// CHECK:  mo.graph @basicBroadcastFuse<N, M, K>(%[[ARG0:.*]]: !mo.tensor<[N, M], f32>, %[[ARG1:.*]]: !mo.tensor<[K, M], f32>) -> !mo.tensor<[K, M], f32> {
// CHECK-NEXT:   %[[KERNEL:.*]] = mogg.kernel (%[[ARG1]], %[[ARG0]]) : (%[[IN1:.*]]: !mo.tensor<[K, M], f32>, %[[IN2:.*]]: !mo.tensor<[N, M], f32>) -> !mo.tensor<[K, M], f32>{
// CHECK-NEXT:   %[[OUT:.*]] = mogg.output_placeholders : !mo.tensor<[K, M], f32>
// CHECK-NEXT:   %[[BCAST_1:.*]] = mogg.call "mo.static.broadcast_to"(%[[IN2]], %[[OUT]]) : !mo.tensor<[N, M], f32>, !mo.tensor<[K, M], f32> -> !mo.tensor<[K, M], f32>
// CHECK-NEXT:   %[[BCAST_2:.*]] = mogg.call "mo.static.broadcast_to"(%[[IN1]], %[[OUT]]) : !mo.tensor<[K, M], f32>, !mo.tensor<[K, M], f32> -> !mo.tensor<[K, M], f32>
// CHECK-NEXT:   mogg.lambda @input_0_fn = (%[[INDICES:.*]]: !mogg.indices) -> !mogg.simd<f32>{
// CHECK-NEXT:      %[[LOAD:.*]] = mogg.simd_load(%[[BCAST_1]], %[[INDICES]]) : (!mo.tensor<[K, M], f32>, !mogg.indices) -> !mogg.simd<f32>
// CHECK-NEXT:      mogg.output %[[LOAD]] : !mogg.simd<f32>
// CHECK-NEXT:   }
// CHECK-NEXT:   mogg.lambda @input_1_fn = (%[[INDICES:.*]]: !mogg.indices) -> !mogg.simd<f32>{
// CHECK-NEXT:      %[[LOAD:.*]] = mogg.simd_load(%[[BCAST_2]], %[[INDICES]]) : (!mo.tensor<[K, M], f32>, !mogg.indices) -> !mogg.simd<f32>
// CHECK-NEXT:      mogg.output %[[LOAD]] : !mogg.simd<f32>
// CHECK-NEXT:   }
// CHECK-NEXT:   mogg.lambda @output_0_fn = (%[[INDICES:.*]]: !mogg.indices, %[[VALUE:.*]]: !mogg.simd<f32>){
// CHECK-NEXT:      mogg.call "simd_store"(%[[OUT]], %[[INDICES]], %[[VALUE]]) : !mo.tensor<[K, M], f32>, !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:      mogg.output
// CHECK-NEXT:   }
// CHECK-NEXT:   mogg.lambda @elemwise_compute_lambda = (%[[INDICES]]: !mogg.indices){
// CHECK-NEXT:      %[[LOAD_IN1:.*]] = mogg.invoke @input_0_fn(%[[INDICES]]) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
// CHECK-NEXT:      %[[LOAD_IN2:.*]] = mogg.invoke @input_1_fn(%[[INDICES]]) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
// CHECK-NEXT:      %[[ADD:.*]] = mogg.call "mo.add"(%[[LOAD_IN1]], %[[LOAD_IN2]]) : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:      mogg.invoke @output_0_fn(%[[INDICES]], %[[ADD]]) : !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:      mogg.output
// CHECK-NEXT:   }
// CHECK-NEXT:   mogg.call "elementwise" [@elemwise_compute_lambda](%[[OUT]]) : !mo.tensor<[K, M], f32>
// CHECK-NEXT:   mogg.output %[[OUT]] : !mo.tensor<[K, M], f32>

// Check the attributes have been stripped so this can no longer be fused.
// CHECK-NEXT:   }
// CHECK-NEXT:   mo.output %[[KERNEL]] : !mo.tensor<[K, M], f32>

mo.graph @basicBroadcastIntoElemwise(%arg0: !mo.tensor<[10, 12, 1, 50], f32>, %arg1: !mo.tensor<[100, 1], f32>) -> (!mo.tensor<[10, 12, 100, 50], f32>) {
  %bcast1 = mo.static.broadcast_to(%arg0) : !mo.tensor<[10, 12, 1, 50], f32> -> !mo.tensor<[10, 12, 100, 50], f32>
  %bcast2 = mo.static.broadcast_to(%arg1) : !mo.tensor<[100, 1], f32> -> !mo.tensor<[10, 12, 100, 50], f32>
  %1 = mo.add(%bcast1, %bcast2) : !mo.tensor<[10, 12, 100, 50], f32>
  %2 = mo.sqrt(%1) : !mo.tensor<[10, 12, 100, 50], f32>
  %3 = mo.tanh(%1) : !mo.tensor<[10, 12, 100, 50], f32>
  %4 = mo.mul(%2, %3) : !mo.tensor<[10, 12, 100, 50], f32>
  mo.output %4 : !mo.tensor<[10, 12, 100, 50], f32>
}

// CHECK:  mo.graph @basicBroadcastIntoElemwise(%[[ARG0:.*]]: !mo.tensor<[10, 12, 1, 50], f32>, %[[ARG1:.*]]: !mo.tensor<[100, 1], f32>)
// CHECK-NEXT:   %[[KERNEL:.*]] = mogg.kernel (%[[ARG1]], %[[ARG0]]) : (%[[IN1:.*]]: !mo.tensor<[100, 1], f32>, %[[IN2:.*]]: !mo.tensor<[10, 12, 1, 50], f32>) -> !mo.tensor<[10, 12, 100, 50], f32>
// CHECK-NEXT:   mogg.output_placeholders
// CHECK-NEXT:   %[[BCAST_1:.*]] = mogg.call "mo.static.broadcast_to"(%[[IN2]], %[[OUT]]) : !mo.tensor<[10, 12, 1, 50], f32>, !mo.tensor<[10, 12, 100, 50], f32> -> !mo.tensor<[10, 12, 100, 50], f32>
// CHECK-NEXT:   %[[BCAST_2:.*]] = mogg.call "mo.static.broadcast_to"(%[[IN1]], %[[OUT]]) : !mo.tensor<[100, 1], f32>, !mo.tensor<[10, 12, 100, 50], f32> -> !mo.tensor<[10, 12, 100, 50], f32>

// Check we load from the bcasts
// CHECK: mogg.lambda @input_0_fn = (%[[INDICES:.*]]: !mogg.indices)
// CHECK-NEXT:  mogg.simd_load(%[[BCAST_1]], %[[INDICES]]) : (!mo.tensor<[10, 12, 100, 50], f32>, !mogg.indices) -> !mogg.simd<f32>

// CHECK:mogg.lambda @input_1_fn = (%[[INDICES:.*]]: !mogg.indices)
// CHECK-NEXT:      mogg.simd_load(%[[BCAST_2]], %[[INDICES]]) : (!mo.tensor<[10, 12, 100, 50], f32>, !mogg.indices) -> !mogg.simd<f32>

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

// Check we produce one kernel. The mo primitive tests check this kernel so we don't
// need to check that thoroughly just check one kernel is produced.
mo.graph @many_bcasts() -> !mo.tensor<[2, 2, 3], f32> {
  %0 = mo.constant {value = #M.dense_array<1.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 10.000000e+00, 11.000000e+00> : tensor<12xf32>} : !mo.tensor<[2, 2, 3], f32>
  %1 = mo.constant {value = #M.dense_array<2.000000e+00, 0.000000e+00> : tensor<2xf32>} : !mo.tensor<[2, 1, 1], f32>
  %2 = mo.constant {value = #M.dense_array<5.000000e-01, 2.500000e-01> : tensor<2xf32>} : !mo.tensor<[2, 1], f32>
  %3 = mo.constant {value = #M.dense_array<1.000000e+00, 7.000000e+00, 8.000000e+00> : tensor<3xf32>} : !mo.tensor<[3], f32>

  %bcast = mo.static.broadcast_to(%1) : !mo.tensor<[2, 1, 1], f32> -> !mo.tensor<[2, 2, 3], f32>
  %bcast2 = mo.static.broadcast_to(%2) : !mo.tensor<[2, 1], f32> -> !mo.tensor<[2, 2, 3], f32>
  %bcast3 = mo.static.broadcast_to(%3) : !mo.tensor<[3], f32> -> !mo.tensor<[2, 2, 3], f32>

  %out = mo.add(%bcast, %0) : !mo.tensor<[2, 2, 3], f32>
  %out2 = mo.mul(%bcast2, %out) : !mo.tensor<[2, 2, 3], f32>
  %out3 = mo.sub(%out2, %bcast3) : !mo.tensor<[2, 2, 3], f32>

  mo.output %out3 : !mo.tensor<[2, 2, 3], f32>
}

// CHECK-LABEL: mo.graph @many_bcasts
// CHECK: mogg.kernel
// CHECK-NEXT: mogg.output_placeholders
// CHECK-NEXT:  %[[BCAST_1:.*]] = mogg.call "mo.static.broadcast_to"
// CHECK-NEXT:  %[[BCAST_2:.*]] = mogg.call "mo.static.broadcast_to"
// CHECK-NEXT:  %[[BCAST_3:.*]] = mogg.call "mo.static.broadcast_to"

// CHECK: mogg.lambda @elemwise_compute_lambda =
// CHECK-NEXT:  %[[LOAD1:.*]] = mogg.invoke @input_0_fn_0
// CHECK-NEXT:  %[[LOAD2:.*]] = mogg.invoke @input_0_fn
// CHECK-NEXT:  %[[LOAD3:.*]] = mogg.invoke @input_1_fn_0
// CHECK-NEXT:  %[[ADD:.*]] = mogg.call "mo.add"(%[[LOAD2]], %[[LOAD3]])
// CHECK-NEXT:  %[[MUL:.*]] = mogg.call "mo.mul"(%[[LOAD1]], %[[ADD]])
// CHECK-NEXT:  %[[LOAD4:.*]] = mogg.invoke @input_1_fn
// CHECK-NEXT:  %[[SUB:.*]] = mogg.call "mo.sub"(%[[MUL]], %[[LOAD4]])


mo.graph @check_no_fuse_if<N, M>(%cond : !mo.tensor<[], bool>, %arg0: !mo.tensor<[], f32>) -> (!mo.tensor<[N, M], f32>) {
  %bcast1 = mo.static.broadcast_to(%arg0) : !mo.tensor<[], f32> -> !mo.tensor<[N, M], f32>

  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[N, M], f32> {
    %1 = mo.add(%bcast1, %bcast1) : !mo.tensor<[N, M], f32>
    mo.yield %1 : !mo.tensor<[N, M], f32>
  } else {
    %2 = mo.mul(%bcast1, %bcast1) : !mo.tensor<[N, M], f32>
    mo.yield %2 : !mo.tensor<[N, M], f32>
  }
  mo.output %0 : !mo.tensor<[N, M], f32>
}


// CHECK-LABEL: mo.graph @check_no_fuse_if

// Check the bcasts still become standalone kernels
// CHECK-NEXT: %[[BCAST1:.*]] = mogg.kernel
// CHECK-NEXT: mogg.output_placeholders
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"

// Check that the elementwise kernel is a straight up copy, i.e an unfused broadcast.
// CHECK:  mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices)
// CHECK-NEXT:  %[[LOAD:.*]] = mogg.invoke @input_0_fn
// CHECK-NEXT:  mogg.invoke @output_0_fn(%[[INDICES]], %[[LOAD]])

// Check the kernels are compiled unfused.
// CHECK: mo.if
// CHECK-NEXT: mogg.kernel
// CHECK: mogg.call "mo.add"
// CHECK: mo.yield
// CHECK-NEXT: } else {
// CHECK-NEXT: mogg.kernel
// CHECK: mogg.call "mo.mul"
// CHECK: mo.yield

// Check everything gets fused together.
mo.graph @transpose_and_bcast_fused(%arg0 : !mo.tensor<[], si32>) -> !mo.tensor<[5, 4], si32>  {
  %0 = mo.constant {value = #M.dense_array<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20> : tensor<20xsi32>} : !mo.tensor<[4, 5], si32>
  %perm = mo.constant {value = #M.dense_array<1, 0> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
  %trans = mo.transpose(%0, %perm) : (!mo.tensor<[4, 5], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, 4], si32>
  %bcast = mo.static.broadcast_to(%arg0) : !mo.tensor<[], si32> -> !mo.tensor<[5, 4], si32>
  %add = mo.add(%bcast, %trans) : !mo.tensor<[5, 4], si32>
  mo.output %add : !mo.tensor<[5, 4], si32>
}

// CHECK-LABEL: mo.graph @transpose_and_bcast_fused
// CHECK: mogg.kernel
// CHECK-NEXT: mogg.output_placeholders
// CHECK-NEXT:  %[[BCAST:.*]] = mogg.call "mo.static.broadcast_to"
// CHECK-NEXT:  %[[TRANSPOSE:.*]] = mogg.call "mo.transpose"

// CHECK:   mogg.lambda @input_0_fn = (%[[INDICES:.*]]: !mogg.indices) -> !mogg.simd<si32>{
// CHECK-NEXT:      mogg.simd_load(%[[BCAST]], %[[INDICES]])

// CHECK:   mogg.lambda @input_1_fn = (%[[INDICES:.*]]: !mogg.indices) -> !mogg.simd<si32>{
// CHECK-NEXT:      mogg.simd_load(%[[TRANSPOSE]], %[[INDICES]]) 

// CHECK: mogg.lambda @elemwise_compute_lambda =
// CHECK-NEXT:  %[[LOAD1:.*]] = mogg.invoke @input_0_fn
// CHECK-NEXT:  %[[LOAD2:.*]] = mogg.invoke @input_1_fn
// CHECK-NEXT:  %[[ADD:.*]] = mogg.call "mo.add"(%[[LOAD1]], %[[LOAD2]])
