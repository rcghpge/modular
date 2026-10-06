// RUN: mo-opt %s --mgp-partition-device-graph-collective --allow-unregistered-dialect --split-input-file | FileCheck %s

// A multi-device `isDeviceGraph` model after `mgp-stage-device-graphs`. The
// `mgp.device_graph.staging` op wraps the per-device producers, the
// `mgp.device_graph.stable.create` ops, and the per-device consumers. This is
// the input shape expected by `mgp-partition-device-graph-collective`.
//
// Each `stable.create` is lowered per region: the producing region publishes
// its tensor with `stable.place` and takes the spec from that tensor, every
// other region reads the slot with `stable.get` and builds the spec from the
// static type, and `stable.tensor` rebuilds the `!mgp.tensor<..., stable>` the
// consumers were lowered against. The consumer regions themselves are cloned
// unchanged.
//
// CHECK-LABEL: mgp.model @multi_device_stable
// CHECK-NOT:   mgp.device_graph.staging
// CHECK:       mgp.device_graph.collective builder(%[[B:[a-z0-9_]+]]) stable(
// CHECK-SAME:      !mgp.stable_addr<<2x3xf32, <"gpu", 0>>>,
// CHECK-SAME:      !mgp.stable_addr<<2x3xf32, <"gpu", 1>>>)
// CHECK:       <"gpu", 0> {
// CHECK:         %[[PROD0:[a-z0-9_]+]] = mogg.generic.compiled.region
// CHECK:         %[[P0:[a-z0-9_]+]] = mgp.device_graph.stable.place %[[B]][0], %[[PROD0]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK:         %[[S0:[a-z0-9_]+]] = mgp.tensor.extract.tensor_spec %[[PROD0]] : <2x3xf32, <"gpu", 0>>
// CHECK:         %[[T0:[a-z0-9_]+]] = mgp.device_graph.stable.tensor %[[P0]], %[[S0]] : !mgp.stable_addr<<2x3xf32, <"gpu", 0>>>
// CHECK:         %[[G1:[a-z0-9_]+]] = mgp.device_graph.stable.get %[[B]][1] : !mgp.stable_addr<<2x3xf32, <"gpu", 1>>>
// CHECK:         %[[S1:[a-z0-9_]+]] = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
// CHECK:         %[[T1:[a-z0-9_]+]] = mgp.device_graph.stable.tensor %[[G1]], %[[S1]] : !mgp.stable_addr<<2x3xf32, <"gpu", 1>>>
// CHECK:         mogg.generic.compiled.region (%[[T0]], %[[T1]], %{{[a-z0-9_]+}})
// CHECK-SAME:      (%[[A0:[a-z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>, stable>,
// CHECK-SAME:       %[[A1:[a-z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 1>, stable>,
// CHECK:           mogg.async.unpack(%[[A0]]) : !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
// CHECK:           mogg.async.unpack(%[[A1]]) : !mgp.tensor<2x3xf32, <"gpu", 1>, stable>
// CHECK:         mgp.device_graph.collective.yield
// CHECK:       }, <"gpu", 1> {
// CHECK:         %[[PROD1:[a-z0-9_]+]] = mogg.generic.compiled.region
// CHECK:         %[[P1:[a-z0-9_]+]] = mgp.device_graph.stable.place %[[B]][1], %[[PROD1]]
// CHECK:         %[[S1B:[a-z0-9_]+]] = mgp.tensor.extract.tensor_spec %[[PROD1]]
// CHECK:         %[[T1B:[a-z0-9_]+]] = mgp.device_graph.stable.tensor %[[P1]], %[[S1B]]
// CHECK:         %[[G0:[a-z0-9_]+]] = mgp.device_graph.stable.get %[[B]][0]
// CHECK:         %[[S0B:[a-z0-9_]+]] = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
// CHECK:         %[[T0B:[a-z0-9_]+]] = mgp.device_graph.stable.tensor %[[G0]], %[[S0B]]
// CHECK:         mogg.generic.compiled.region (%[[T0B]], %[[T1B]], %{{[a-z0-9_]+}})
// CHECK:         mgp.device_graph.collective.yield
// CHECK:       }

mgp.model @multi_device_stable : mof devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  mef.output %ch0, %ctx, %weights, %gpu0, %gpu1
      : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">,
        !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">,
           %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>,
           %in0: !mgp.tensor<2x3xf32, <"gpu", 0>>, %in1: !mgp.tensor<2x3xf32, <"gpu", 1>>)
    -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>) {

  %c0, %c1 = mgp.device_graph.staging
      inputs(%in0, %gpu0, %in1, %gpu1
          : !mgp.tensor<2x3xf32, <"gpu", 0>>,
            !mgp.device_context<<"gpu", 0>>,
            !mgp.tensor<2x3xf32, <"gpu", 1>>,
            !mgp.device_context<<"gpu", 1>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %in0_t: !mgp.tensor<2x3xf32, <"gpu", 0>>,
         %gpu0_t: !mgp.device_context<<"gpu", 0>>,
         %in1_t: !mgp.tensor<2x3xf32, <"gpu", 1>>,
         %gpu1_t: !mgp.device_context<<"gpu", 1>>):

    // Producer on gpu:0.
    %a = mogg.generic.compiled.region (%in0_t, %gpu0_t)
        : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>,
           %devctx: !mgp.device_context<<"gpu", 0>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 0>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 0>>) -> ()
      %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = "test.producer0"
      {device = #M.device_ref<"gpu", 0>, inOutSignature = #M.inout_sig<"">, isAsync = false}

    // Producer on gpu:1.
    %b = mogg.generic.compiled.region (%in1_t, %gpu1_t)
        : (%arg: !mgp.tensor<2x3xf32, <"gpu", 1>>,
           %devctx: !mgp.device_context<<"gpu", 1>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 1>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 1>>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 1>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 1>>) -> ()
      %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 1>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 1>>>
    } fusionInfo = "test.producer1"
      {device = #M.device_ref<"gpu", 1>, inOutSignature = #M.inout_sig<"">, isAsync = false}

    // Promote each producer result to a stable address inside the staging body.
    %s0 = mgp.device_graph.stable.create %a : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %s1 = mgp.device_graph.stable.create %b : !mgp.tensor<2x3xf32, <"gpu", 1>>

    // Consumer on gpu:0, reading both stable values.
    // The stable flag is an internal device-graph detail: the value returned
    // from the staging region escapes as a normal tensor.
    %c0 = mogg.generic.compiled.region (%s0, %s1, %gpu0_t)
        : (%arg0: !mgp.tensor<2x3xf32, <"gpu", 0>, stable>,
           %arg1: !mgp.tensor<2x3xf32, <"gpu", 1>, stable>,
            %devctx: !mgp.device_context<<"gpu", 0>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
      %u0 = mogg.async.unpack(%arg0) : !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
      %u1 = mogg.async.unpack(%arg1) : !mgp.tensor<2x3xf32, <"gpu", 1>, stable>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 0>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 0>>) -> ()
      %v = builtin.unrealized_conversion_cast %u0
          : !mgp.tensor<2x3xf32, <"gpu", 0>, stable> to !mgp.tensor<2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%v) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = "test.consumer0"
      {device = #M.device_ref<"gpu", 0>, inOutSignature = #M.inout_sig<"">, isAsync = false}

    // Consumer on gpu:1, reading both stable values.
    %c1 = mogg.generic.compiled.region (%s0, %s1, %gpu1_t)
        : (%arg0: !mgp.tensor<2x3xf32, <"gpu", 0>, stable>,
           %arg1: !mgp.tensor<2x3xf32, <"gpu", 1>, stable>,
            %devctx: !mgp.device_context<<"gpu", 1>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 1>> {
      %u0 = mogg.async.unpack(%arg0) : !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
      %u1 = mogg.async.unpack(%arg1) : !mgp.tensor<2x3xf32, <"gpu", 1>, stable>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 1>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 1>>) -> ()
      %v = builtin.unrealized_conversion_cast %u1
          : !mgp.tensor<2x3xf32, <"gpu", 1>, stable> to !mgp.tensor<2x3xf32, <"gpu", 1>>
      %p = mogg.async.pack(%v) : !mgp.tensor<2x3xf32, <"gpu", 1>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 1>>>
    } fusionInfo = "test.consumer1"
      {device = #M.device_ref<"gpu", 1>, inOutSignature = #M.inout_sig<"">, isAsync = false}

    mgp.device_graph.staging.yield %c0, %c1
        : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>

  mef.output %ch0, %c0, %c1
      : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>
} {isDeviceGraph}

// -----

// A consumer lives on a device that has no producer of its own. The partition
// must still create a region for that device so the consumer can read the
// remote stable value and yield its result.
//
// CHECK-LABEL: mgp.model @consumer_only_device
// CHECK-NOT:   mgp.device_graph.staging
// CHECK:       mgp.device_graph.collective builder(%[[B:[a-z0-9_]+]]) stable(
// CHECK-SAME:      !mgp.stable_addr<<2x3xf32, <"gpu", 0>>>)
// CHECK:       <"gpu", 0> {
// CHECK:         mogg.generic.compiled.region
// CHECK:         mgp.device_graph.stable.place
// CHECK:         mgp.device_graph.collective.yield
// CHECK:       }, <"gpu", 1> {
// CHECK:         %[[G0:[a-z0-9_]+]] = mgp.device_graph.stable.get %[[B]][0]
// CHECK:         %[[S0:[a-z0-9_]+]] = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
// CHECK:         %[[T0:[a-z0-9_]+]] = mgp.device_graph.stable.tensor %[[G0]], %[[S0]]
// CHECK:         mogg.generic.compiled.region (%[[T0]], %{{[a-z0-9_]+}})
// CHECK-SAME:      (%[[A0:[a-z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>, stable>,
// CHECK:           mogg.async.unpack(%[[A0]]) : !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
// CHECK:         mgp.device_graph.collective.yield %{{.*}} : !mgp.tensor<2x3xf32, <"gpu", 1>>
// CHECK:       } : !mgp.tensor<2x3xf32, <"gpu", 1>>

mgp.model @consumer_only_device : mof devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  mef.output %ch0, %ctx, %weights, %gpu0, %gpu1
      : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">,
        !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">,
           %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>,
           %in0: !mgp.tensor<2x3xf32, <"gpu", 0>>)
    -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 1>>) {

  %c0 = mgp.device_graph.staging
      inputs(%in0, %gpu0, %gpu1
          : !mgp.tensor<2x3xf32, <"gpu", 0>>,
            !mgp.device_context<<"gpu", 0>>,
            !mgp.device_context<<"gpu", 1>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %in0_t: !mgp.tensor<2x3xf32, <"gpu", 0>>,
         %gpu0_t: !mgp.device_context<<"gpu", 0>>,
         %gpu1_t: !mgp.device_context<<"gpu", 1>>):

    // Producer on gpu:0.
    %a = mogg.generic.compiled.region (%in0_t, %gpu0_t)
        : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>,
           %devctx: !mgp.device_context<<"gpu", 0>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 0>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 0>>) -> ()
      %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = "test.producer0"
      {device = #M.device_ref<"gpu", 0>, inOutSignature = #M.inout_sig<"">, isAsync = false}

    // Promote the producer result to a stable address.
    %s0 = mgp.device_graph.stable.create %a : !mgp.tensor<2x3xf32, <"gpu", 0>>

    // Consumer on gpu:1, reading the stable value from gpu:0.
    %c0 = mogg.generic.compiled.region (%s0, %gpu1_t)
        : (%arg0: !mgp.tensor<2x3xf32, <"gpu", 0>, stable>,
           %devctx: !mgp.device_context<<"gpu", 1>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 1>> {
      %u0 = mogg.async.unpack(%arg0) : !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 1>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 1>>) -> ()
      %v = builtin.unrealized_conversion_cast %u0
          : !mgp.tensor<2x3xf32, <"gpu", 0>, stable> to !mgp.tensor<2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%v) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = "test.consumer1"
      {device = #M.device_ref<"gpu", 1>, inOutSignature = #M.inout_sig<"">, isAsync = false}

    mgp.device_graph.staging.yield %c0
        : !mgp.tensor<2x3xf32, <"gpu", 1>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 1>>

  mef.output %ch0, %c0
      : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 1>>
} {isDeviceGraph}
