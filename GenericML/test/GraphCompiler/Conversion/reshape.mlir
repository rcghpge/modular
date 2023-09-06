// RUN: mo-opt %s --mo-to-mgp --allow-unregistered-dialect | FileCheck %s

// Entry point
// CHECK: mgp.model @reshape_static_input_static_output
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[INPUT:.*]]: !mgp.cpu.tensor<4xf32>)

// Extract the input tensor buffer
// CHECK-NEXT: %[[IN_BUF:.*]], %[[IN_SPEC:.*]] = mgp.cpu.tensor.extract %[[INPUT]] : !mgp.cpu.tensor<4xf32>

// Create output shape spec
// CHECK-NEXT: %[[OUT_SPEC:.*]] = mgp.cpu.tensor_spec.create : !mgp.cpu.tensor_spec<2x2xf32>

// Create output tensor with same buffer and new spec
// CHECK-NEXT: %[[OUT_TENSOR:.*]] =  mgp.cpu.tensor.create %[[CHAIN]], %[[IN_BUF]], %[[OUT_SPEC]] : !mgp.cpu.tensor<2x2xf32>

// Return output
// CHECK-NEXT: mef.output %[[CHAIN]], %[[OUT_TENSOR]] : !mef.chain, !mgp.cpu.tensor<2x2xf32>
mo.graph @reshape_static_input_static_output(%arg: !mo.tensor<[4], f32>) -> !mo.tensor<[2, 2], f32> {
  %shape = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.reshape(%arg, %shape) : (!mo.tensor<[4], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %res : !mo.tensor<[2, 2], f32>
}

// Entry point
// CHECK: mgp.model @reshape_dyn_input_dyn_output
// CHECK: } execute (%[[JIT_CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[INPUT:.*]]: !mgp.cpu.tensor<?xf32>, %[[SHAPE:.*]]: !mgp.cpu.tensor<2xsi64>)

// Extract the input tensor buffer and dynamic dimension
// CHECK-NEXT: %[[IN_BUF:.*]], %[[IN_SPEC:.*]] = mgp.cpu.tensor.extract %[[INPUT]] : !mgp.cpu.tensor<?xf32>
// CHECK-NEXT: %[[IN_DIM:.*]] = mgp.cpu.tensor_spec.dims.dynamic %[[IN_SPEC]] : !mgp.cpu.tensor_spec<?xf32>

// Extract the target shape tensor buffer
// CHECK-NEXT: %[[SHAPE_BUF:.*]], %[[SHAPE_SPEC:.*]] = mgp.cpu.tensor.extract %[[SHAPE]] : !mgp.cpu.tensor<2xsi64>

// Invoke fallback shape function
// CHECK-NEXT: %[[IGNORED_CH:.*]], %[[OUT_DIMS:.*]]:2 = mgp.cpu.kernel.execute {{.*}}, name "{{.*}}reshape{{.*}}"
// CHECK-NEXT: %[[OUT_SPEC:.*]] = mgp.cpu.tensor_spec.create %[[OUT_DIMS]]#0, %[[OUT_DIMS]]#1 : !mgp.cpu.tensor_spec<?x?xf32>

// Create output tensor with same buffer and new spec
// CHECK-NEXT: %[[OUT_TENSOR:.*]] =  mgp.cpu.tensor.create %[[CHAIN]], %[[IN_BUF]], %[[OUT_SPEC]] : !mgp.cpu.tensor<?x?xf32>

// Return output
// CHECK-NEXT: mef.output %[[CHAIN]], %[[OUT_TENSOR]] : !mef.chain, !mgp.cpu.tensor<?x?xf32>
mo.graph @reshape_dyn_input_dyn_output(%arg: !mo.tensor<[?], f32>, %shape : !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32> {
  %res = mo.reshape(%arg, %shape) : (!mo.tensor<[?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// Entry point
// CHECK: mgp.model @reshape_dyn_input_static_output
// CHECK: } execute (%[[JIT_CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[INPUT:.*]]: !mgp.cpu.tensor<?xf32>, %[[SHAPE:.*]]: !mgp.cpu.tensor<2xsi64>)

// Extract the input tensor buffer
// CHECK-NEXT: %[[IN_BUF:.*]], %[[IN_SPEC:.*]] = mgp.cpu.tensor.extract %[[INPUT]] : !mgp.cpu.tensor<?xf32>

// Create output shape spec
// CHECK-NEXT: %[[OUT_SPEC:.*]] = mgp.cpu.tensor_spec.create : !mgp.cpu.tensor_spec<4x2xf32>

// Create output tensor with same buffer and new spec
// CHECK-NEXT: %[[OUT_TENSOR:.*]] =  mgp.cpu.tensor.create %[[CHAIN]], %[[IN_BUF]], %[[OUT_SPEC]] !mgp.cpu.buffer : !mgp.cpu.tensor<4x2xf32>

// Return output
// CHECK-NEXT: mef.output %[[CHAIN]], %[[OUT_TENSOR]] : !mef.chain, !mgp.cpu.tensor<4x2xf32>
mo.graph @reshape_dyn_input_static_output(%arg: !mo.tensor<[?], f32>, %shape : !mo.tensor<[2], si64>) -> !mo.tensor<[4, 2], f32> {
  %res = mo.reshape(%arg, %shape) : (!mo.tensor<[?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 2], f32>
  mo.output %res : !mo.tensor<[4, 2], f32>
}

// Entry point
// CHECK: mgp.model @reshape_static_input_dyn_output
// CHECK: } execute (%[[JIT_CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[INPUT:.*]]: !mgp.cpu.tensor<8xf32>, %[[SHAPE:.*]]: !mgp.cpu.tensor<2xsi64>)

// Extract the target shape tensor buffer
// CHECK-NEXT: %[[SHAPE_BUF:.*]], %[[SHAPE_SPEC:.*]] = mgp.cpu.tensor.extract %[[SHAPE]] : !mgp.cpu.tensor<2xsi64>

// Invoke fallback shape function
// CHECK-NEXT: %[[IGNORED_CH:.*]], %[[OUT_DIMS:.*]]:2 = mgp.cpu.kernel.execute {{.*}}, name "{{.*}}reshape{{.*}}"

// Extract the input tensor buffer
// CHECK-NEXT: %[[IN_BUF:.*]], %[[IN_SPEC:.*]] = mgp.cpu.tensor.extract %[[INPUT]] : !mgp.cpu.tensor<8xf32>

// Create output spec
// CHECK-NEXT: %[[OUT_SPEC:.*]] = mgp.cpu.tensor_spec.create %[[OUT_DIMS]]#0, %[[OUT_DIMS]]#1 : !mgp.cpu.tensor_spec<?x?xf32>

// Create output tensor with same buffer and new spec
// CHECK-NEXT: %[[OUT_TENSOR:.*]] =  mgp.cpu.tensor.create %[[CHAIN]], %[[IN_BUF]], %[[OUT_SPEC]] !mgp.cpu.buffer<32> : !mgp.cpu.tensor<?x?xf32>

// Return output
// CHECK-NEXT: mef.output %[[CHAIN]], %[[OUT_TENSOR]] : !mef.chain, !mgp.cpu.tensor<?x?xf32>
mo.graph @reshape_static_input_dyn_output(%arg: !mo.tensor<[8], f32>, %shape : !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32> {
  %res = mo.reshape(%arg, %shape) : (!mo.tensor<[8], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}
