// RUN: mo-opt %s --mgp-generate-build-graph-stubs --allow-unregistered-dialect --split-input-file --verify-diagnostics

// The producer's launch context must be an argument of the kernel stub (in any
// position) -- a context materialized inside the body leaves no argument slot
// for the builder to take over.

#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>

mogg.kernel.decl @k registration = "test.k" device_info = <"", "", "", ""> execute = #kf build_graph = #kf struct_params = [] is_user_defined = false

// expected-error @below {{device-graph recording stub generation: expected the producer launch context to be an argument of the kernel stub}}
mogg.kernel.stub @stub_1 (%in: !mo.tensor<[2, 3], f32, gpu:0, {io = 1 : i32, mut = false}>, %out: !mo.tensor<[2, 3], f32, gpu:0, {io = 0 : i32, mut = true}>) fusionInfo = "test.k":1 attributes {abi = 0 : i32, constantResources = [], device = "gpu", isAsync = true, sym_visibility = "private"} {
  %ctx = "test.make_ctx"() : () -> !mogg.context
  mogg.with_trace {
    mogg.call.execute["k"] inputs(%in, %ctx : !mo.tensor<[2, 3], f32, gpu:0, {io = 1 : i32, mut = false}>, !mogg.context) outputs(%out : !mo.tensor<[2, 3], f32, gpu:0, {io = 0 : i32, mut = true}>) {isThrows, kernel_param = {transpose_b = true}, moOpName = "test.k"}
  }
  mogg.output
}

mgp.model @dg : mof devices [#M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      %t_in = "test.tensor_in"() : () -> !mo.tensor<[2, 3], f32, gpu:0, {io = 1 : i32, mut = false}>
      %t_out = "test.tensor_out"() : () -> !mo.tensor<[2, 3], f32, gpu:0, {io = 0 : i32, mut = true}>
      mogg.call.stub @stub_1 %t_in, %t_out : !mo.tensor<[2, 3], f32, gpu:0, {io = 1 : i32, mut = false}>, !mo.tensor<[2, 3], f32, gpu:0, {io = 0 : i32, mut = true}>
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
} {isDeviceGraph}
