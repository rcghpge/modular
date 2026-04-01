// RUN: mo-opt %s --split-input-file --allow-unregistered-dialect --canonicalize | FileCheck %s

// CHECK-LABEL: mgp.model @chain_empty
mgp.model @chain_empty : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
// CHECK: execute (%[[ARG0:.*]]: !mef.chain,
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch1 = grt.chain.create ()
  %ch2 = grt.chain.create (%arg0)
  %ch3 = grt.chain.create (%ch1, %ch2)
  %ch4 = grt.chain.create (%ch3)
  // CHECK-NEXT: mef.output %[[ARG0]] : !mef.chain
  mef.output %ch3 : !mef.chain
}

// -----

// CHECK-LABEL: mgp.model @fold_true_assert
mgp.model @fold_true_assert : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %2 = mgp.runtime.create %0, %1[1]
  mef.output %2, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
// CHECK: execute (%[[ARG0:.*]]: !mef.chain,
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain) {
  %true = index.bool.constant true
  %assert = mgp.assert %true, "Tensor type mismatch for argument 0: Expected shape: [2,4]"
  %chain = grt.chain.create (%arg0, %assert)
  // CHECK-NEXT: mef.output %[[ARG0]] : !mef.chain
  mef.output %chain : !mef.chain
}

// -----

// CHECK-LABEL: mgp.model @fold_buffer_get_size_constant
mgp.model @fold_buffer_get_size_constant : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %2 = mgp.runtime.create %0, %1[1]
  mef.output %2, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
// CHECK: execute (%[[ARG0:.*]]: !mef.chain,
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, index) {
  %buf0 = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 128>
  // CHECK: %[[SIZE:.*]] = index.constant 4
  %size = mgp.buffer.get_size %arg0, %buf0 : !mgp.buffer
  // CHECK: mef.output %[[ARG0]], %[[SIZE]] : !mef.chain, index
  mef.output %arg0, %size : !mef.chain, index
}

// -----

// CHECK-LABEL: mgp.model @fold_buffer_get_size_alloc
mgp.model @fold_buffer_get_size_alloc : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %2 = mgp.runtime.create %0, %1[1]
  mef.output %2, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
// CHECK: execute (%[[ARG0:.*]]: !mef.chain,
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, index) {
  %alloc_size = index.constant 4
  %buf0 = mgp.buffer.alloc %arg0, %alloc_size, %arg3 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  // CHECK: %[[SIZE:.*]] = index.constant 4
  %size = mgp.buffer.get_size %arg0, %buf0 : !mgp.buffer
  // CHECK: mef.output %[[ARG0]], %[[SIZE]] : !mef.chain, index
  mef.output %arg0, %size : !mef.chain, index
}

// -----

// CHECK-LABEL: mgp.model @buffer_plan_extract_constants
mgp.model @buffer_plan_extract_constants : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 0
  mef.output %ch0, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %dynamic_size: index) -> (!mef.chain, index, index, index, index) {
  // Test: constant runtime sizes should be extracted to staticSizes attribute
  // Input: 3 allocations with sizes [constant 100, dynamic, constant 200]
  // Expected: staticSizes = [100, 200], runtimeSizes = [dynamic]
  // Attributes are permuted so static allocations come first

  %c100 = index.constant 100
  %c200 = index.constant 200

  // CHECK: execute ({{.*}}%[[ARG:.*]]: index)
  // CHECK: %[[HWM:.*]], %[[OFF:.*]]:3 = mgp.buffer.plan(%[[ARG]])
  // CHECK-SAME: alignments = #M.dense_array<16, 16, 16>
  // CHECK-SAME: minPre = #M.dense_array<0, 2, 1>
  // CHECK-SAME: staticSizes = #M.dense_array<100, 200> : !M.array<2xsi64>
  // CHECK: mef.output {{.*}}, %[[HWM]], %[[OFF]]#0, %[[OFF]]#2, %[[OFF]]#1
  %hwm, %offsets:3 = mgp.buffer.plan(%c100, %dynamic_size, %c200) {
    alignments = #M.dense_array<16, 16, 16> : !M.array<3xui64>,
    minPre = #M.dense_array<0, 1, 2> : !M.array<3xui64>,
    minPost = #M.dense_array<0, 1, 2> : !M.array<3xui64>,
    maxPre = #M.dense_array<10, 11, 12> : !M.array<3xui64>,
    maxPost = #M.dense_array<10, 11, 12> : !M.array<3xui64>
  } : (index, index, index) -> (index, index, index, index)

  mef.output %ch0, %hwm, %offsets#0, %offsets#1, %offsets#2 : !mef.chain, index, index, index, index
}

// -----

// CHECK-LABEL: mgp.model @buffer_plan_all_constants
mgp.model @buffer_plan_all_constants : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 0
  mef.output %ch0, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, index, index, index, index) {
  // Test: all runtime sizes are constants - should all be extracted

  %c100 = index.constant 100
  %c200 = index.constant 200
  %c300 = index.constant 300

  // CHECK: %[[HWM:.*]], %[[OFF:.*]]:3 = mgp.buffer.plan() {
  // CHECK-SAME: staticSizes = #M.dense_array<100, 200, 300> : !M.array<3xsi64>
  // CHECK: mef.output {{.*}}, %[[HWM]], %[[OFF]]#0, %[[OFF]]#1, %[[OFF]]#2
  %hwm, %offsets:3 = mgp.buffer.plan(%c100, %c200, %c300) {
    alignments = #M.dense_array<16, 32, 64> : !M.array<3xui64>,
    minPre = #M.dense_array<0, 1, 2> : !M.array<3xui64>,
    minPost = #M.dense_array<5, 6, 7> : !M.array<3xui64>,
    maxPre = #M.dense_array<10, 11, 12> : !M.array<3xui64>,
    maxPost = #M.dense_array<15, 16, 17> : !M.array<3xui64>
  } : (index, index, index) -> (index, index, index, index)

  mef.output %ch0, %hwm, %offsets#0, %offsets#1, %offsets#2 : !mef.chain, index, index, index, index
}

// -----

// CHECK-LABEL: mgp.model @buffer_plan_no_constants
mgp.model @buffer_plan_no_constants : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 0
  mef.output %ch0, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %size0: index, %size1: index) -> (!mef.chain, index, index, index) {
  // Test: no constants - should not canonicalize

  // CHECK: execute ({{.*}}%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
  // CHECK: mgp.buffer.plan(%[[ARG0]], %[[ARG1]])
  // CHECK-NOT: staticSizes
  %hwm, %offsets:2 = mgp.buffer.plan(%size0, %size1) {
    alignments = #M.dense_array<16, 16> : !M.array<2xui64>,
    minPre = #M.dense_array<0, 1> : !M.array<2xui64>,
    minPost = #M.dense_array<5, 6> : !M.array<2xui64>,
    maxPre = #M.dense_array<10, 11> : !M.array<2xui64>,
    maxPost = #M.dense_array<15, 16> : !M.array<2xui64>
  } : (index, index) -> (index, index, index)

  mef.output %ch0, %hwm, %offsets#0, %offsets#1 : !mef.chain, index, index, index
}

