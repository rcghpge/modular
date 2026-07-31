// RUN: mo-opt %s --mgp-finalize-device-graphs --allow-unregistered-dialect --split-input-file --verify-diagnostics

// Finalization owns the invariant that no `builtin.unrealized_conversion_cast`
// survives device-graph synthesis, so it diagnoses the cases it cannot resolve
// instead of letting them fail later in EmitMojo.

// The builder's backing device context dispatches eagerly, so it may only be
// handed to ops audited to use it for host-side buffer views. `mgp.sync` is
// `Pure` yet records a device event, so it must not silently get the backing
// context -- effect-freedom is not the criterion.

mgp.model @dg_unsafe_context_user : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %out = mgp.device_graph.staging inputs(%ch0, %ctx, %in : !mef.chain, !mgp.context, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder,
       %ch_in: !mef.chain,
       %ctx_in: !mgp.context,
       %in_t: !mgp.tensor<2x3xf32, <"gpu", 0>>):
    // expected-error @+1 {{'mgp.sync' is not known to be safe against the graph builder's backing device context}}
    %dev = builtin.unrealized_conversion_cast %builder : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
    %bch = "test.make_chain"() : () -> !mef.chain
    %synced = mgp.sync (%bch, %ctx_in, %dev) : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
    mgp.device_graph.staging.yield %in_t : !mgp.tensor<2x3xf32, <"gpu", 0>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
} {isDeviceGraph}

// -----

// The graph's device comes from its own inputs/outputs, which is what selects the
// matching execute device context. A staged graph whose I/O carries no device at
// all leaves nothing to select with.

mgp.model @dg_no_device : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %size = "test.produce_size"() : () -> index
  // expected-error @+1 {{none of the staged graph's inputs or outputs carries a device}}
  %out = mgp.device_graph.staging inputs(%size : index) {
  ^bb0(%builder: !mgp.device_graph.builder,
       %size_in: index):
    mgp.device_graph.staging.yield %size_in : index
  } -> index
  mef.output %ch0, %in : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
}  {isDeviceGraph}
