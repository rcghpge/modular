// RUN: mo-opt %s --dynamic-memory-planning --split-input-file --allow-unregistered-dialect | FileCheck %s

// Test basic memory planning with non-overlapping lifetimes
// CHECK-LABEL: mgp.model @basic_memory_planning
mgp.model @basic_memory_planning : mof devices [#M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: index) -> !mef.chain {
  // CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[WEIGHTS:.*]]: !mgp.opaque<"weights_registry">, %[[GPU_DEV:.*]]: !mgp.device_context<<"gpu", 0>>, %[[SIZE:.*]]: index) -> !mef.chain {

  // These allocations should be merged into a single pool
  // CHECK: %[[HIGH_WATER:.*]], %[[OFFSETS:.*]]:2 = "mgp.buffer.plan"
  // CHECK: %[[POOL:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[HIGH_WATER]], %[[GPU_DEV]], align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  // CHECK: mgp.buffer.slice %[[POOL]], %[[OFFSETS]]#0, %[[SIZE]]
  // CHECK: mgp.buffer.slice %[[POOL]], %[[OFFSETS]]#1, %[[SIZE]]

  %alloc0 = mgp.buffer.alloc %arg0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %alloc1 = mgp.buffer.alloc %arg0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>

  // Sequential use pattern
  %ch0 = "use"(%arg0, %alloc0) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  %ch1 = "use"(%ch0, %alloc1) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain

  mef.output %ch1 : !mef.chain
}

// -----

// Test that CPU allocations are ignored
// CHECK-LABEL: mgp.model @cpu_allocations_ignored
mgp.model @cpu_allocations_ignored : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0, %arg1, %arg2 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>, %arg4: !mgp.device_context<<"gpu", 0>>, %arg5: index) -> !mef.chain {
  // CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[WEIGHTS:.*]]: !mgp.opaque<"weights_registry">, %[[CPU_DEV:.*]]: !mgp.device_context<<"cpu", 0>>, %[[GPU_DEV:.*]]: !mgp.device_context<<"gpu", 0>>, %[[SIZE:.*]]: index) -> !mef.chain {
  // GPU allocations should be planned
  // CHECK: %[[GPU_HIGH_WATER:.*]], %[[GPU_OFFSETS:.*]]:2 = "mgp.buffer.plan"
  // CHECK: %[[GPU_POOL:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[GPU_HIGH_WATER]], %[[GPU_DEV]], align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  // CHECK: mgp.buffer.slice %[[GPU_POOL]], %[[GPU_OFFSETS]]#0, %[[SIZE]]
  // CHECK: mgp.buffer.slice %[[GPU_POOL]], %[[GPU_OFFSETS]]#1, %[[SIZE]]
  %gpu_alloc0 = mgp.buffer.alloc %arg0, %arg5, %arg4, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %gpu_alloc1 = mgp.buffer.alloc %arg0, %arg5, %arg4, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>

  // CPU allocations should remain untouched
  // CHECK: %[[CPU_ALLOC:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[SIZE]], %[[CPU_DEV]], align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %cpu_alloc = mgp.buffer.alloc %arg0, %arg5, %arg3, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>

  %ch0 = "use"(%arg0, %cpu_alloc) : (!mef.chain, !mgp.buffer) -> !mef.chain
  %ch1 = "use"(%arg0, %gpu_alloc0) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  %ch2 = "use"(%ch1, %gpu_alloc1) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  %ch3 = grt.chain.create (%ch0, %ch2)

  mef.output %ch3 : !mef.chain
}

// -----

// Test single allocation is not planned
// CHECK-LABEL: mgp.model @single_allocation_not_planned
mgp.model @single_allocation_not_planned : mof devices [#M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: index) -> !mef.chain {
  // CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[WEIGHTS:.*]]: !mgp.opaque<"weights_registry">, %[[GPU_DEV:.*]]: !mgp.device_context<<"gpu", 0>>, %[[SIZE:.*]]: index) -> !mef.chain {
  // Single allocation should remain as-is
  // CHECK-NOT: mgp.buffer.plan
  // CHECK: %[[SINGLE_ALLOC:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[SIZE]], %[[GPU_DEV]], align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %alloc = mgp.buffer.alloc %arg0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>

  %ch = "use"(%arg0, %alloc) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain

  mef.output %ch : !mef.chain
}

// -----

// Test different alignments
// CHECK-LABEL: mgp.model @different_alignments
mgp.model @different_alignments : mof devices [#M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: index) -> !mef.chain {
  // CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[WEIGHTS:.*]]: !mgp.opaque<"weights_registry">, %[[GPU_DEV:.*]]: !mgp.device_context<<"gpu", 0>>, %[[SIZE:.*]]: index) -> !mef.chain {
  // Pool should use LCM alignment (64)
  // CHECK: %[[ALIGN_HIGH_WATER:.*]], %[[ALIGN_OFFSETS:.*]]:3 = "mgp.buffer.plan"
  // CHECK-SAME: aAlignments = #M.dense_array<16, 32, 64>
  // CHECK: %[[ALIGN_POOL:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[ALIGN_HIGH_WATER]], %[[GPU_DEV]], align 64 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  // CHECK: mgp.buffer.slice %[[ALIGN_POOL]], %[[ALIGN_OFFSETS]]#0, %[[SIZE]]
  // CHECK: mgp.buffer.slice %[[ALIGN_POOL]], %[[ALIGN_OFFSETS]]#1, %[[SIZE]]
  // CHECK: mgp.buffer.slice %[[ALIGN_POOL]], %[[ALIGN_OFFSETS]]#2, %[[SIZE]]
  %alloc1 = mgp.buffer.alloc %arg0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %alloc2 = mgp.buffer.alloc %arg0, %arg4, %arg3, align 32 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %alloc3 = mgp.buffer.alloc %arg0, %arg4, %arg3, align 64 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>

  %ch0 = "use"(%arg0, %alloc1) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  %ch1 = "use"(%ch0, %alloc2) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  %ch2 = "use"(%ch1, %alloc3) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain

  mef.output %ch2 : !mef.chain
}

// -----

// Test multiple devices are handled separately
// CHECK-LABEL: mgp.model @multiple_devices
mgp.model @multiple_devices : mof devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>, %arg2: !mgp.device_context<<"gpu", 1>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0, %arg1, %arg2 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: !mgp.device_context<<"gpu", 1>>, %arg5: index) -> !mef.chain {
  // CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[WEIGHTS:.*]]: !mgp.opaque<"weights_registry">, %[[GPU0_DEV:.*]]: !mgp.device_context<<"gpu", 0>>, %[[GPU1_DEV:.*]]: !mgp.device_context<<"gpu", 1>>, %[[SIZE:.*]]: index) -> !mef.chain {
  // Each device should get its own memory pool
  // CHECK-DAG: %[[GPU0_HIGH_WATER:.*]], %[[GPU0_OFFSETS:.*]]:2 = "mgp.buffer.plan"
  // CHECK-DAG: %[[GPU0_POOL:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[GPU0_HIGH_WATER]], %[[GPU0_DEV]], align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  // CHECK-DAG: %[[GPU1_HIGH_WATER:.*]], %[[GPU1_OFFSETS:.*]]:2 = "mgp.buffer.plan"
  // CHECK-DAG: %[[GPU1_POOL:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[GPU1_HIGH_WATER]], %[[GPU1_DEV]], align 16 : !mgp.buffer<<"gpu", 1>>, !mgp.device_context<<"gpu", 1>>

  %alloc0_gpu0 = mgp.buffer.alloc %arg0, %arg5, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %alloc1_gpu0 = mgp.buffer.alloc %arg0, %arg5, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>

  %alloc0_gpu1 = mgp.buffer.alloc %arg0, %arg5, %arg4, align 16 : !mgp.buffer<<"gpu", 1>>, !mgp.device_context<<"gpu", 1>>
  %alloc1_gpu1 = mgp.buffer.alloc %arg0, %arg5, %arg4, align 16 : !mgp.buffer<<"gpu", 1>>, !mgp.device_context<<"gpu", 1>>

  // CHECK-DAG: mgp.buffer.slice %[[GPU0_POOL]], %[[GPU0_OFFSETS]]#0, %[[SIZE]] : <<"gpu", 0>>
  // CHECK-DAG: mgp.buffer.slice %[[GPU0_POOL]], %[[GPU0_OFFSETS]]#1, %[[SIZE]] : <<"gpu", 0>>
  // CHECK-DAG: mgp.buffer.slice %[[GPU1_POOL]], %[[GPU1_OFFSETS]]#0, %[[SIZE]] : <<"gpu", 1>>
  // CHECK-DAG: mgp.buffer.slice %[[GPU1_POOL]], %[[GPU1_OFFSETS]]#1, %[[SIZE]] : <<"gpu", 1>>

  %ch0 = "use"(%arg0, %alloc0_gpu0) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  %ch1 = "use"(%ch0, %alloc1_gpu0) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  %ch2 = "use"(%arg0, %alloc0_gpu1) : (!mef.chain, !mgp.buffer<<"gpu", 1>>) -> !mef.chain
  %ch3 = "use"(%ch2, %alloc1_gpu1) : (!mef.chain, !mgp.buffer<<"gpu", 1>>) -> !mef.chain
  %ch4 = grt.chain.create (%ch1, %ch3)

  mef.output %ch4 : !mef.chain
}

// -----

// Test that buffers with writers (kOut operands) are not incorrectly reused
// This test reproduces the expert_token_counts bug where a buffer written by one
// operation and read by another was incorrectly aliased with a different allocation.
// CHECK-LABEL: mgp.model @buffer_with_writer_not_reused
mgp.model @buffer_with_writer_not_reused : mof devices [#M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: index) -> !mef.chain {
  // CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[WEIGHTS:.*]]: !mgp.opaque<"weights_registry">, %[[GPU_DEV:.*]]: !mgp.device_context<<"gpu", 0>>, %[[SIZE:.*]]: index) -> !mef.chain {

  // Allocate a buffer that will be written to
  %output_buffer = mgp.buffer.alloc %arg0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>

  // Write to the output buffer (kOut semantics on last operand)
  // The buffer contents are not valid until this operation completes
  %ch0 = "compute"(%arg0, %output_buffer) {tempInOutSig = #M.inout_sig<"io">} : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain

  // Allocate another buffer
  %temp_buffer = mgp.buffer.alloc %ch0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>

  // Use the temp buffer
  %ch1 = "use_temp"(%ch0, %temp_buffer) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain

  // Read from the output buffer - its contents must still be valid!
  // This is where the bug manifests: if output_buffer's memory was incorrectly reused for temp_buffer,
  // we'll read garbage/zeros instead of the computed values
  %ch2 = "read_output"(%ch1, %output_buffer) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain

  // With the targeted fix, both buffers CAN be pooled, but the buffer with the writer
  // should have minPost=maxPost to ensure its memory is not reused until all operations complete.
  // The planning algorithm should recognize that the lifetimes don't allow aliasing.
  // CHECK: %[[HIGH_WATER:.*]], %[[OFFSETS:.*]]:2 = "mgp.buffer.plan"
  // CHECK: %[[POOL:.*]] = mgp.buffer.alloc
  // CHECK: %[[SLICE1:.*]] = mgp.buffer.slice %[[POOL]], %[[OFFSETS]]#0
  // CHECK: %[[SLICE2:.*]] = mgp.buffer.slice %[[POOL]], %[[OFFSETS]]#1

  mef.output %ch2 : !mef.chain
}
