// RUN: mo-opt %s --allow-unregistered-dialect --exec-invariant-code-motion --canonicalize --cse --split-input-file | FileCheck %s

//
// @liftable_op
//

// CHECK-LABEL: mgp.model @liftable_op : mof
// CHECK-DAG:   %[[CH0:.*]] = grt.chain.create ()
// CHECK-DAG:   %[[CTX:.*]] = mgp.context.create 3
// CHECK-DAG:   %[[CH1:.*]] = mgp.runtime.create %[[CH0]], %1[1]
// CHECK-DAG:   %[[BUF:.*]] = mgp.buffer.constant.external %[[CH1]], %[[CTX]][0]
// CHECK-DAG:   %[[CACHE_CH:.*]] = mgp.buffer.cache %[[CH0]], %[[CTX]][1, 2], %[[BUF]]
// CHECK-DAG:   %[[CH_OUT:.*]] = grt.chain.create (%[[CH1]], %[[CACHE_CH]])
// CHECK:       mef.output %[[CH_OUT]], %[[CTX]]
// CHECK-NEXT:  } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context

// CHECK:   %[[BUF:.*]] = mgp.buffer.get_cached %[[CH0]], %[[CTX]][1, 2] : !mgp.buffer<37632>
// CHECK:   mef.output %[[CH0]], %[[BUF]] : !mef.chain, !mgp.buffer<37632>
// CHECK: }

mgp.model @liftable_op : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, !mgp.buffer<37632>) {
  %out_buf = mgp.buffer.constant.external %ch0, %ctx[0], %weights["foo", 37632, align 16] : !mgp.opaque<"weights_registry"> -> !mgp.buffer<37632>
  mef.output %ch0, %out_buf : !mef.chain, !mgp.buffer<37632>
}

// -----

//
// @non_pure_op
//

// CHECK-LABEL: mgp.model @non_pure_op : mof
// CHECK-NEXT:  %{{.*}} = grt.chain.create ()
// Nothing was lifted.
// CHECK:       %[[CTX:.*]] = mgp.context.create 2

mgp.model @non_pure_op : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  %const_buf = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000803F0000803F0000803F0000803F", align 16>
  %const_spec = mgp.tensor_spec.create  : !mgp.tensor_spec<2x2xf32>
  %out_spec = "foo.side_effect_op"(%const_spec) : (!mgp.tensor_spec<2x2xf32>) -> !mgp.tensor_spec<?x?xf32>
  %out_dim0, %out_dim1 = mgp.tensor_spec.dims.dynamic %out_spec : !mgp.tensor_spec<?x?xf32>
  %c8 = index.constant 8
  %tmp = index.mul %out_dim0, %out_dim1
  %bytes = index.mul %c8, %tmp
  %out_buf = mgp.buffer.alloc %ch0, %ctx[1], %bytes, %cpu, align 16 : !mgp.buffer<>, !mgp.opaque<"cpu0_context">
  %ch2 = "bar.side_effect_op"(%ch0, %const_buf, %const_spec, %out_buf, %out_spec) : (!mef.chain, !mgp.buffer<16>, !mgp.tensor_spec<2x2xf32>, !mgp.buffer<>, !mgp.tensor_spec<?x?xf32>) -> !mef.chain
  %out_tensor = mgp.tensor.create %ch2, %ctx[1], %out_buf, %out_spec  : !mgp.tensor<?x?xf32>
  mef.output %ch0, %out_tensor : !mef.chain, !mgp.tensor<?x?xf32>
}

// -----

//
// @dont_lift_if_no_computation_saved
//

// CHECK-LABEL: mgp.model @dont_lift_if_no_computation_saved : mof
// CHECK:  %{{.*}} = grt.chain.create ()
// Nothing changed
// CHECK:       %[[CTX:.*]] = mgp.context.create 3
// CHECK:       mef.output %{{.*}}, %[[CTX]]
//
// CHECK-NEXT:  } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// CHECK:       mef.output %[[CH0]], %{{.*}}
// CHECK-NEXT:  }

mgp.model @dont_lift_if_no_computation_saved : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, !mgp.buffer<0>) {
  %size = index.constant 0
  %out_buf = mgp.buffer.alloc %ch0, %ctx[1], %size, %cpu, align 16 : !mgp.buffer<0>, !mgp.opaque<"cpu0_context">
  mef.output %ch0, %out_buf : !mef.chain, !mgp.buffer<0>
}

// -----

// CHECK-LABEL: mgp.model @generic_execute : mof
// CHECK-DAG:     %[[CH0:.*]] = grt.chain.create ()
// CHECK-DAG:     %[[CTX:.*]] = mgp.context.create 3
// CHECK-DAG:     %[[CH1:.*]] = mgp.runtime.create %[[CH0]], %1[1]
// CHECK-DAG:     %[[BUF:.*]] = mgp.buffer.alloc %[[CH1]]
// CHECK-DAG:     %[[CH3:.*]] = mgp.buffer.cache %[[CH0]], %[[CTX]][1, 2], %[[BUF]]
// CHECK-DAG:     %[[MERGE:.*]] = grt.chain.create (%[[CH1]], %[[CH3]])
// CHECK:         mef.output %[[MERGE]], %[[CTX]]
// CHECK-NEXT:  } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// CHECK:         %[[BUF:.*]] = mgp.buffer.get_cached %[[CH0]]
// CHECK:         %[[OUT:.*]] = "mgp.generic.execute"({{.*}}, %[[BUF]])

mgp.model @generic_execute : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, !mgp.buffer<37632>) {
  %weight_buf = mgp.buffer.constant %ch0, %ctx[1], dense_resource<onnx__Conv_497> : !M.aligned_bytes<37632, align 16>

  %size = index.constant 37632
  %out_buf = mgp.buffer.alloc %ch0, %ctx[1], %size, %cpu, align 16 : !mgp.buffer<37632>, !mgp.opaque<"cpu0_context">
  %out_spec = mgp.tensor_spec.create  : !mgp.tensor_spec<7x7x3x64xf32>

  %i42 = index.constant 42

  %ch2 = "mgp.generic.execute"(%ctx, %ch0, %cpu, %weight_buf, %out_buf) {
    aRuntimeSlot = 0 : ui64,
    bModuleIndex = 0 : ui64,
    cFunctionIndex = 0 : ui64,
    dSummary = "mogg.generic.compiled.region:14",
    eIsAsync = false,
    fHasCuda = false,
    gCudaSlot = 0 : ui64
  } : (!mgp.context, !mef.chain, !mgp.opaque<"cpu0_context">, !mgp.buffer<37632>, !mgp.buffer<37632>) -> (!mef.chain)

  mef.output %ch2, %out_buf : !mef.chain, !mgp.buffer<37632>
}

// -----

//
// @while
//

// CHECK-LABEL: mgp.model @while : mof
// Nothing was lifted
// CHECK-NEXT:  %{{.*}} = grt.chain.create ()
// CHECK:       %{{.*}} = mgp.context.create 2

mgp.model @while : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">

} execute (%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain) {

  %i, %j = grt.while (%ch: !mef.chain, %ctx: !mgp.context) {
    // Sync is a liftable op
    %ch2 = mgp.sync(%ch, %ctx) {aRuntimeSlot = 2 : ui64, bDevice = #M.device_ref<"cuda", 0>} : (!mef.chain, !mgp.context) -> !mef.chain

    %cond = "some.op"() : () -> i1
    mef.output %ch2, %ctx, %cond : !mef.chain, !mgp.context, i1
  } do {
    mef.output %ch, %ctx: !mef.chain, !mgp.context
  }

  mef.output %i : !mef.chain
}

