// RUN: mo-opt %s --mo-to-mgp=constant-fold=false | FileCheck %s

// Entry point
// CHECK: mgp.model @reshape_static_input_static_output
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[INPUT:.*]]: !mgp.tensor<4xf32>)

// Extract the input tensor buffer
// CHECK: %[[IN_BUF:.*]] = mgp.tensor.extract.buffer %[[CTX]][1], %[[INPUT]] : !mgp.tensor<4xf32>

// Create output shape spec
// CHECK: {{.*}}, %[[OUT_SPEC:.*]] = "mgp.generic.execute"{{.*}} -> (!mgp.chain, !mgp.tensor_spec<2x2xf32>)

// Create output tensor with same buffer and new spec
// CHECK: %[[OUT_TENSOR:.*]] =  mgp.tensor.create %{{.*}}, %[[CTX]][1], %[[IN_BUF]], %[[OUT_SPEC]] : !mgp.tensor<2x2xf32>

// Return output
// CHECK: mef.output {{.*}}, %[[OUT_TENSOR]] : !mef.chain, !mgp.tensor<2x2xf32>
mo.graph @reshape_static_input_static_output(%arg: !mo.tensor<[4], f32>) -> !mo.tensor<[2, 2], f32> {
  %shape = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.reshape(%arg, %shape) : (!mo.tensor<[4], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %res : !mo.tensor<[2, 2], f32>
}

// Entry point
// CHECK: mgp.model @reshape_dyn_input_dyn_output
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[INPUT:.*]]: !mgp.tensor<?xf32>, %[[SHAPE:.*]]: !mgp.tensor<2xsi64>)

// Extract the input tensor buffer and dynamic dimension
// CHECK: %[[IN_SPEC:.*]] = mgp.tensor.extract.tensor_spec %[[CTX]][1], %[[INPUT]] : !mgp.tensor<?xf32>
// CHECK: %[[IN_BUF:.*]] = mgp.tensor.extract.buffer %[[CTX]][1], %[[INPUT]] : !mgp.tensor<?xf32>
// CHECK: %[[IN_DIM:.*]] = mgp.tensor_spec.dims.dynamic %[[IN_SPEC]] : !mgp.tensor_spec<?xf32>

// Extract the target shape tensor buffer
// CHECK: %[[SHAPE_BUF:.*]] = mgp.tensor.extract.buffer %[[CTX]][1], %[[SHAPE]] : !mgp.tensor<2xsi64>

// Invoke fallback shape function
// CHECK: %[[FALLBACK_CHAIN:.*]], %[[OUT_DIMS:.*]]:2 = mgp.kernel.execute %{{.*}}, {{.*}}, summary "reshape_shape{{.*}}" out_dims 2

// Create output tensor with same buffer and new spec
// CHECK: {{.*}}, %[[OUT_SPEC:.*]] = "mgp.generic.execute"{{.*}} %[[OUT_DIMS]]#0, %[[OUT_DIMS]]#1{{.*}} -> (!mgp.chain, !mgp.tensor_spec<?x?xf32>)
// CHECK: %[[OUT_TENSOR:.*]] =  mgp.tensor.create %{{.*}}, %[[CTX]][1], %[[IN_BUF]], %[[OUT_SPEC]] : !mgp.tensor<?x?xf32>

// Return output
// CHECK: mef.output {{.*}}, %[[OUT_TENSOR]] : !mef.chain, !mgp.tensor<?x?xf32>
mo.graph @reshape_dyn_input_dyn_output(%arg: !mo.tensor<[?], f32>, %shape : !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32> {
  %res = mo.reshape(%arg, %shape) : (!mo.tensor<[?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// Entry point
// CHECK: mgp.model @reshape_dyn_input_static_output
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[INPUT:.*]]: !mgp.tensor<?xf32>, %[[SHAPE:.*]]: !mgp.tensor<2xsi64>)

// Extract the input tensor buffer
// CHECK: %[[IN_BUF:.*]] = mgp.tensor.extract.buffer %[[CTX]][1], %[[INPUT]] : !mgp.tensor<?xf32>

// Create output shape spec
// CHECK: {{.*}}, %[[OUT_SPEC:.*]] = "mgp.generic.execute"{{.*}} -> (!mgp.chain, !mgp.tensor_spec<4x2xf32>)

// Create output tensor with same buffer and new spec
// CHECK: %[[SIZED_BUF:.*]] = mgp.buffer.resize %[[IN_BUF]] : !mgp.buffer<32>
// CHECK: %[[CH1:.*]] = mgp.chain.host_to_device %[[CH0]], %[[CTX]][1]
// CHECK: %[[OUT_TENSOR:.*]] =  mgp.tensor.create %[[CH1]], %[[CTX]][1], %[[SIZED_BUF]], %[[OUT_SPEC]] : !mgp.tensor<4x2xf32>

// Return output
// CHECK: mef.output {{.*}}, %[[OUT_TENSOR]] : !mef.chain, !mgp.tensor<4x2xf32>
mo.graph @reshape_dyn_input_static_output(%arg: !mo.tensor<[?], f32>, %shape : !mo.tensor<[2], si64>) -> !mo.tensor<[4, 2], f32> {
  %res = mo.reshape(%arg, %shape) : (!mo.tensor<[?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 2], f32>
  mo.output %res : !mo.tensor<[4, 2], f32>
}

// Entry point
// CHECK: mgp.model @reshape_static_input_dyn_output
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[INPUT:.*]]: !mgp.tensor<8xf32>, %[[SHAPE:.*]]: !mgp.tensor<2xsi64>)

// Extract the target shape tensor buffer
// CHECK: %[[SHAPE_BUF:.*]] = mgp.tensor.extract.buffer %[[CTX]][1], %[[SHAPE]] : !mgp.tensor<2xsi64>

// Invoke fallback shape function
// CHECK: %[[FALLBACK_CHAIN:.*]], %[[OUT_DIMS:.*]]:2 = mgp.kernel.execute %{{.*}}, {{.*}}, summary "reshape_shape{{.*}}"

// Extract the input tensor buffer
// CHECK: %[[IN_BUF:.*]] = mgp.tensor.extract.buffer %[[CTX]][1], %[[INPUT]] : !mgp.tensor<8xf32>

// Create output spec
// CHECK: {{.*}}, %[[OUT_SPEC:.*]] = "mgp.generic.execute"{{.*}} %[[OUT_DIMS]]#0, %[[OUT_DIMS]]#1{{.*}} -> (!mgp.chain, !mgp.tensor_spec<?x?xf32>)

// Create output tensor with same buffer and new spec
// CHECK: %[[UNSIZED_BUF:.*]] = mgp.buffer.unsize %[[IN_BUF]] : !mgp.buffer<32>
// CHECK: %[[OUT_TENSOR:.*]] =  mgp.tensor.create %{{.*}}, %[[CTX]][1], %[[UNSIZED_BUF]], %[[OUT_SPEC]] : !mgp.tensor<?x?xf32>

// Return output
// CHECK: mef.output {{.*}}, %[[OUT_TENSOR]] : !mef.chain, !mgp.tensor<?x?xf32>
mo.graph @reshape_static_input_dyn_output(%arg: !mo.tensor<[8], f32>, %shape : !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32> {
  %res = mo.reshape(%arg, %shape) : (!mo.tensor<[8], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}
