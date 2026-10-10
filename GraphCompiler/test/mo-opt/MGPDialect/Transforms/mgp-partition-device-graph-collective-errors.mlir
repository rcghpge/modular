// RUN: mo-opt %s --mgp-partition-device-graph-collective --allow-unregistered-dialect --split-input-file --verify-diagnostics

// A stable slot carries only a data pointer. The producing region rebuilds the
// stable tensor's spec from the tensor itself, but a consumer on another
// device has no tensor to ask, so a dynamic shape cannot cross devices yet.

mgp.model @dynamic_stable_crosses_devices device_graph : mof devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  mef.output %ch0, %ctx, %weights, %gpu0, %gpu1
      : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">,
        !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">,
           %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>,
           %in0: !mgp.tensor<?x3xf32, <"gpu", 0>>)
    -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 1>>) {

  %c0 = mgp.device_graph.staging
      inputs(%in0, %gpu0, %gpu1
          : !mgp.tensor<?x3xf32, <"gpu", 0>>,
            !mgp.device_context<<"gpu", 0>>,
            !mgp.device_context<<"gpu", 1>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %in0_t: !mgp.tensor<?x3xf32, <"gpu", 0>>,
         %gpu0_t: !mgp.device_context<<"gpu", 0>>,
         %gpu1_t: !mgp.device_context<<"gpu", 1>>):

    %a = mogg.routine.async (%in0_t, %gpu0_t)
        : (%arg: !mgp.tensor<?x3xf32, <"gpu", 0>>,
           %devctx: !mgp.device_context<<"gpu", 0>>)
        -> !mgp.tensor<?x3xf32, <"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<?x3xf32, <"gpu", 0>>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 0>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 0>>) -> ()
      %p = mogg.async.pack(%u) : !mgp.tensor<?x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<?x3xf32, <"gpu", 0>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.producer0">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}

    // expected-error@+1 {{'mgp.device_graph.stable.create' op stable tensor with dynamic dimensions is consumed on another device; only its data pointer crosses devices, so its shape cannot be recovered there}}
    %s0 = mgp.device_graph.stable.create %a : !mgp.tensor<?x3xf32, <"gpu", 0>>

    %c0 = mogg.routine.async (%s0, %gpu1_t)
        : (%arg0: !mgp.tensor<?x3xf32, <"gpu", 0>, stable>,
           %devctx: !mgp.device_context<<"gpu", 1>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 1>> {
      %u0 = mogg.async.unpack(%arg0) : !mgp.tensor<?x3xf32, <"gpu", 0>, stable>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 1>>
      %r = "test.kernel"(%u0, %c)
          : (!mgp.tensor<?x3xf32, <"gpu", 0>, stable>,
             !mgp.device_context<<"gpu", 1>>) -> !mgp.tensor<2x3xf32, <"gpu", 1>>
      %p = mogg.async.pack(%r) : !mgp.tensor<2x3xf32, <"gpu", 1>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 1>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.consumer1">> inOutSignature = <""> {device = #M.device_ref<"gpu", 1>}

    mgp.device_graph.staging.yield %c0 : !mgp.tensor<2x3xf32, <"gpu", 1>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 1>>

  mef.output %ch0, %c0 : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 1>>
}
