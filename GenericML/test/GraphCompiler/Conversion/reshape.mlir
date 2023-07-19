// RUN: mo-opt %s --mo-to-mogg --mo-to-primitives --allow-unregistered-dialect | FileCheck %s

// Entry point
// CHECK: mgp.model @reshape_static_input_static_output : tf {{.*}} init {
// CHECK-NEXT: %[[INIT_CHAIN:.*]] = grt.chain.create ()
// CHECK-NEXT: %[[INIT_CTX:.*]] = mgp.context.create 1
// CHECK-NEXT: %[[JIT_CHAIN:.*]] = "mop.create_jit_context"(%[[INIT_CHAIN]], %[[INIT_CTX]]) {slot = 0 : ui32} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK-NEXT: mef.output %[[JIT_CHAIN]], %[[INIT_CTX]] : !mef.chain, !mgp.context
// CHECK-NEXT: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[INPUT:.*]]: !mgp.cpu.tensor<4xf32>) -> (!mef.chain, !mgp.cpu.tensor<2x2xf32>) {

// Constant for output shape
// CHECK-NEXT: %[[SHAPE_CONST:.*]] = mgp.cpu.buffer.constant
// CHECK-NEXT: %[[SHAPE_SPEC:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<2xsi64>
// CHECK-NEXT: %[[SHAPE_TENSOR:.*]] = mgp.cpu.tensor.create %[[CHAIN]], %[[SHAPE_CONST]], %[[SHAPE_SPEC]] : !mgp.cpu.tensor<2xsi64>

// Extract the input tensor buffer
// CHECK-NEXT: %[[IN_BUF:.*]], %[[IN_SPEC:.*]] = mgp.cpu.tensor.extract %[[INPUT]] : !mgp.cpu.tensor<4xf32>

// NOTE call hacky primitive to compute output spec
// CHECK-NEXT: %[[OUT_SPEC:.*]] = "mop.shape.reshape"(%[[IN_SPEC]], %[[SHAPE_TENSOR]]) : (!mgp.cpu.tensor_spec<4xf32>, !mgp.cpu.tensor<2xsi64>) -> !mgp.cpu.tensor_spec<2x2xf32>

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
// CHECK: mgp.model @reshape_dyn_input_dyn_output : tf {{.*}} init {
// CHECK-NEXT: %[[INIT_CHAIN:.*]] = grt.chain.create ()
// CHECK-NEXT: %[[INIT_CTX:.*]] = mgp.context.create 1
// CHECK-NEXT: %[[JIT_CHAIN:.*]] = "mop.create_jit_context"(%[[INIT_CHAIN]], %[[INIT_CTX]]) {slot = 0 : ui32} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK-NEXT: mef.output %[[JIT_CHAIN]], %[[INIT_CTX]] : !mef.chain, !mgp.context
// CHECK-NEXT: } execute (%[[JIT_CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[INPUT:.*]]: !mgp.cpu.tensor<?xf32>, %[[SHAPE:.*]]: !mgp.cpu.tensor<2xsi64>) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>) {

// Extract the input tensor buffer
// CHECK-NEXT: %[[IN_BUF:.*]], %[[IN_SPEC:.*]] = mgp.cpu.tensor.extract %[[INPUT]] : !mgp.cpu.tensor<?xf32>

// NOTE call hacky primitive to compute output spec
// CHECK-NEXT: %[[OUT_SPEC:.*]] = "mop.shape.reshape"(%[[IN_SPEC]], %[[SHAPE]]) : (!mgp.cpu.tensor_spec<?xf32>, !mgp.cpu.tensor<2xsi64>) -> !mgp.cpu.tensor_spec<?x?xf32>

// Create output tensor with same buffer and new spec
// CHECK-NEXT: %[[OUT_TENSOR:.*]] =  mgp.cpu.tensor.create %[[CHAIN]], %[[IN_BUF]], %[[OUT_SPEC]] : !mgp.cpu.tensor<?x?xf32>

// Return output
// CHECK-NEXT: mef.output %[[CHAIN]], %[[OUT_TENSOR]] : !mef.chain, !mgp.cpu.tensor<?x?xf32>
mo.graph @reshape_dyn_input_dyn_output(%arg: !mo.tensor<[?], f32>, %shape : !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32> {
  %res = mo.reshape(%arg, %shape) : (!mo.tensor<[?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// Entry point
// CHECK: mgp.model @reshape_dyn_input_static_output : tf {{.*}} init {
// CHECK-NEXT: %[[INIT_CHAIN:.*]] = grt.chain.create ()
// CHECK-NEXT: %[[INIT_CTX:.*]] = mgp.context.create 1
// CHECK-NEXT: %[[JIT_CHAIN:.*]] = "mop.create_jit_context"(%[[INIT_CHAIN]], %[[INIT_CTX]]) {slot = 0 : ui32} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK-NEXT: mef.output %[[JIT_CHAIN]], %[[INIT_CTX]] : !mef.chain, !mgp.context
// CHECK-NEXT: } execute (%[[JIT_CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[INPUT:.*]]: !mgp.cpu.tensor<?xf32>, %[[SHAPE:.*]]: !mgp.cpu.tensor<2xsi64>) -> (!mef.chain, !mgp.cpu.tensor<4x2xf32>) {

// Extract the input tensor buffer
// CHECK-NEXT: %[[IN_BUF:.*]], %[[IN_SPEC:.*]] = mgp.cpu.tensor.extract %[[INPUT]] : !mgp.cpu.tensor<?xf32>

// NOTE call hacky primitive to compute output spec
// CHECK-NEXT: %[[OUT_SPEC:.*]] = "mop.shape.reshape"(%[[IN_SPEC]], %[[SHAPE]]) : (!mgp.cpu.tensor_spec<?xf32>, !mgp.cpu.tensor<2xsi64>) -> !mgp.cpu.tensor_spec<4x2xf32>

// Create output tensor with same buffer and new spec
// CHECK-NEXT: %[[OUT_TENSOR:.*]] =  mgp.cpu.tensor.create %[[CHAIN]], %[[IN_BUF]], %[[OUT_SPEC]] !mgp.cpu.buffer : !mgp.cpu.tensor<4x2xf32>

// Return output
// CHECK-NEXT: mef.output %[[CHAIN]], %[[OUT_TENSOR]] : !mef.chain, !mgp.cpu.tensor<4x2xf32>
mo.graph @reshape_dyn_input_static_output(%arg: !mo.tensor<[?], f32>, %shape : !mo.tensor<[2], si64>) -> !mo.tensor<[4, 2], f32> {
  %res = mo.reshape(%arg, %shape) : (!mo.tensor<[?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 2], f32>
  mo.output %res : !mo.tensor<[4, 2], f32>
}

// Entry point
// CHECK: mgp.model @reshape_static_input_dyn_output : tf {{.*}} init {
// CHECK-NEXT: %[[INIT_CHAIN:.*]] = grt.chain.create ()
// CHECK-NEXT: %[[INIT_CTX:.*]] = mgp.context.create 1
// CHECK-NEXT: %[[JIT_CHAIN:.*]] = "mop.create_jit_context"(%[[INIT_CHAIN]], %[[INIT_CTX]]) {slot = 0 : ui32} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK-NEXT: mef.output %[[JIT_CHAIN]], %[[INIT_CTX]] : !mef.chain, !mgp.context
// CHECK-NEXT: } execute (%[[JIT_CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[INPUT:.*]]: !mgp.cpu.tensor<8xf32>, %[[SHAPE:.*]]: !mgp.cpu.tensor<2xsi64>) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>) {

// Extract the input tensor buffer
// CHECK-NEXT: %[[IN_BUF:.*]], %[[IN_SPEC:.*]] = mgp.cpu.tensor.extract %[[INPUT]] : !mgp.cpu.tensor<8xf32>

// NOTE call hacky primitive to compute output spec
// CHECK-NEXT: %[[OUT_SPEC:.*]] = "mop.shape.reshape"(%[[IN_SPEC]], %[[SHAPE]]) : (!mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.tensor<2xsi64>) -> !mgp.cpu.tensor_spec<?x?xf32>

// Create output tensor with same buffer and new spec
// CHECK-NEXT: %[[OUT_TENSOR:.*]] =  mgp.cpu.tensor.create %[[CHAIN]], %[[IN_BUF]], %[[OUT_SPEC]] !mgp.cpu.buffer<32> : !mgp.cpu.tensor<?x?xf32>

// Return output
// CHECK-NEXT: mef.output %[[CHAIN]], %[[OUT_TENSOR]] : !mef.chain, !mgp.cpu.tensor<?x?xf32>
mo.graph @reshape_static_input_dyn_output(%arg: !mo.tensor<[8], f32>, %shape : !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32> {
  %res = mo.reshape(%arg, %shape) : (!mo.tensor<[8], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}
