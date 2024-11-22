// RUN: mo-opt %s --split-input-file --allow-unregistered-dialect --canonicalize | FileCheck %s

// CHECK-LABEL: mgp.model @chain_empty
mgp.model @chain_empty : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
// CHECK: execute (%[[ARG0:.*]]: !mef.chain,
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch1 = grt.chain.create ()
  %ch2 = grt.chain.create (%arg0)
  %ch3 = grt.chain.create (%ch1, %ch2)
  %ch4 = grt.chain.create (%ch3)
  // CHECK-NEXT: mef.output %[[ARG0]] : !mef.chain
  mef.output %ch3 : !mef.chain
}

// -----

// CHECK-LABEL: mgp.model @fold_true_assert
mgp.model @fold_true_assert : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.opaque<"cpu0_context">) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %2 = mgp.runtime.create %0, %1[1]
  mef.output %2, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
// CHECK: execute (%[[ARG0:.*]]: !mef.chain,
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.opaque<"cpu0_context">) -> (!mef.chain) {
  %true = index.bool.constant true
  %assert = mgp.assert %true, "Tensor type mismatch for argument 0: Expected shape: [2,4]"
  %chain = grt.chain.create (%arg0, %assert)
  // CHECK-NEXT: mef.output %[[ARG0]] : !mef.chain
  mef.output %chain : !mef.chain
}

// -----

// CHECK-LABEL: mgp.model @fold_buffer_get_size
mgp.model @fold_buffer_get_size : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.opaque<"cpu0_context">) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %2 = mgp.runtime.create %0, %1[1]
  mef.output %2, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
// CHECK: execute (%[[ARG0:.*]]: !mef.chain,
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.opaque<"cpu0_context">) -> (!mef.chain, index) {
  %buf0 = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 128>
  // CHECK: %[[SIZE:.*]] = index.constant 4
  %size = mgp.buffer.get_size %arg0, %buf0 : !mgp.buffer<4>
  // CHECK: mef.output %[[ARG0]], %[[SIZE]] : !mef.chain, index
  mef.output %arg0, %size : !mef.chain, index
}
