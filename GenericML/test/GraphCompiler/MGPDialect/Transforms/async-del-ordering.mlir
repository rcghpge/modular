// RUN: mo-opt %s -mo-to-mogg -mo-to-primitives -canonicalize -outline-kernels -add-try-except-blocks -emit-mojo -dump-mojo 2>&1 | FileCheck %s

// Test that mogg_async_del is emitted BEFORE mogg_async_ready/mogg_async_error.
// This ensures that when await() returns for downstream consumers, all cleanup
// from this kernel (including destructor calls for opaque types) has completed.
// See AddTryExceptBlocks.cpp for the implementation.
//
// Background: The AsyncRT await() function only waits for values to be READY,
// but does NOT guarantee all cleanup work has completed. By placing async_del
// before async_ready, we ensure cleanup happens as part of the kernel execution
// rather than being deferred to the work queue.

// ============================================================================
// Non-throwing kernel (mo.cumsum)
// For non-throwing kernels, there is no try/except block. AsyncDelOp should
// appear before AsyncReadyOp.
// ============================================================================
mo.graph @non_throwing_cumsum(%input : !mo.tensor<[5], f32>, %axis : !mo.tensor<[], si64>) -> !mo.tensor<[5], f32> {
  %res = mo.cumsum(%input, %axis) {reverse = 0 : index, exclusive = 0 : index} : (!mo.tensor<[5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[5], f32>
  mo.output %res : !mo.tensor<[5], f32>
}

// ============================================================================
// Throwing kernel (mo.add)
// For throwing kernels, AsyncDelOp should be inside BOTH try and except blocks,
// before AsyncReadyOp/AsyncErrorOp respectively.
// ============================================================================
mo.graph @throwing_add(%0 : !mo.tensor<[4], si32>, %1: !mo.tensor<[4], si32>) -> !mo.tensor<[4], si32> {
  %2 = mo.add(%0, %1) : !mo.tensor<[4], si32>
  mo.output %2 : !mo.tensor<[4], si32>
}

// FileCheck patterns must be in output order (which is reverse of file order).
// throwing_add appears first in output, non_throwing_cumsum appears second.

// CHECK-LABEL: fn region_1
// CHECK:       try:
// CHECK:         mogg_async_del
// CHECK:         mogg_async_ready
// CHECK:       except
// CHECK:         mogg_async_del
// CHECK:         mogg_async_error

// CHECK-LABEL: fn region_0
// CHECK:       mogg_async_del
// CHECK:       mogg_async_ready
