// RUN: mo-opt %s --mo-to-mgp="constant-fold=false enable-compiled-regions-fusion=true skip-jit-compile-middle=true" | FileCheck %s

// Entry point
// CHECK: mgp.model @reshape_static_input_static_output
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[INPUT:[^:]*]]: !mgp.tensor<4xf32>)

// Extract the input tensor buffer
// CHECK: %[[IN_BUF:.*]] = mgp.tensor.extract.buffer %[[CTX]][0], %[[INPUT]] : !mgp.tensor<4xf32>

// Create output shape spec
// CHECK: {{.*}}, %[[OUT_RESULT:.*]]:2 = "mgp.generic.execute"{{.*}} -> (!mgp.chain, !mef.chain, !mgp.tensor_spec<2x2xf32>)

// Create output tensor with same buffer and new spec
// CHECK: %[[OUT_TENSOR:.*]] =  mgp.tensor.create %{{.*}}, %[[CTX]][0], %[[IN_BUF]], %[[OUT_RESULT]]#1 : !mgp.tensor<2x2xf32>

// Return output
// CHECK: mef.output {{.*}}, %[[OUT_TENSOR]] : !mef.chain, !mgp.tensor<2x2xf32>
mo.graph @reshape_static_input_static_output(%arg: !mo.tensor<[4], f32>) -> !mo.tensor<[2, 2], f32> {
  %shape = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.reshape(%arg, %shape) : (!mo.tensor<[4], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %res : !mo.tensor<[2, 2], f32>
}

// Entry point
// CHECK: mgp.model @reshape_dyn_input_dyn_output
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[INPUT:[^:]*]]: !mgp.tensor<?xf32>, %[[SHAPE:.*]]: !mgp.tensor<2xsi64>)

// Extract the input tensor buffer and dynamic dimension
// CHECK: %[[IN_SPEC:.*]] = mgp.tensor.extract.tensor_spec %[[CTX]][0], %[[INPUT]] : !mgp.tensor<?xf32>
// CHECK: %[[IN_BUF:.*]] = mgp.tensor.extract.buffer %[[CTX]][0], %[[INPUT]] : !mgp.tensor<?xf32>
// CHECK: %[[IN_DIM:.*]] = mgp.tensor_spec.dims.dynamic %[[IN_SPEC]] : !mgp.tensor_spec<?xf32>

// Extract the target shape tensor buffer
// CHECK: %[[SHAPE_BUF:.*]] = mgp.tensor.extract.buffer %[[CTX]][0], %[[SHAPE]] : !mgp.tensor<2xsi64>

// Invoke fallback shape function (compiled region fusion enabled)
// CHECK: "mgp.generic.execute"{{.*}}dSummary = {{.*}}reshape_shape{{.*}}

// Create output tensor
// CHECK: %[[OUT_TENSOR:.*]] =  mgp.tensor.create %{{.*}}, %[[CTX]][0], %[[IN_BUF]], %{{.*}} : !mgp.tensor<?x?xf32>

// Return output
// CHECK: mef.output {{.*}}, %[[OUT_TENSOR]] : !mef.chain, !mgp.tensor<?x?xf32>
mo.graph @reshape_dyn_input_dyn_output(%arg: !mo.tensor<[?], f32>, %shape : !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32> {
  %res = mo.reshape(%arg, %shape) : (!mo.tensor<[?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}
