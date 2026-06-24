// RUN: mo-opt %s --mgp-generate-build-graph-stubs --allow-unregistered-dialect --split-input-file | FileCheck %s

// Stage 4 of device-graph synthesis. Input IR mirrors the shape produced by
// `mgp-compute-device-graph-dependencies`: the execute block is a
// `mgp.device_graph.staging` op holding `mgp.device_graph.dependencies` ops, and
// the kernel-dispatch node invokes a module-scope `mogg.kernel.stub @stub_1` via
// `mogg.call.stub`. The kernel decl, stub body, and model are minimal synthetic
// stand-ins (the real builtin kernels lack a `build_graph` method, which
// `mogg.call.build_graph` verification requires).
//
// For each staged `mogg.call.stub @stub_N` the pass generates a sibling recording
// stub `@stub_N_build_graph` (producer dispatch `mogg.call.execute` -> recording
// `mogg.call.build_graph` against a `!mgp.device_graph.builder`, yielding a
// `!mgp.device_graph.node`) and retargets the call to it, threading the staging
// op's builder as the launch operand.

#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>

mogg.kernel.decl @k registration = "test.k" device_info = <"", "", "", ""> execute = #kf build_graph = #kf struct_params = [] is_user_defined = false

// The original stub is retained (still used by the direct-execution / init path),
// keeping its `!mogg.context` launch arg and `mogg.call.execute` dispatch.
// CHECK: mogg.kernel.stub @stub_1 (%{{.*}}: !mo.tensor<[2, 3], f32, gpu:0, {{.*}}>, %{{.*}}: !mo.tensor<[2, 3], f32, gpu:0, {{.*}}>, %{{.*}}: !mogg.context)
// CHECK: mogg.with_trace
// CHECK: mogg.call.execute["k"]
//
// The recording sibling is generated immediately after the original stub (so it
// precedes the model): builder launch arg, node result, producer recorded via
// `mogg.call.build_graph` (no `mogg.with_trace`), node forwarded out, and the
// producer's `kernel_param` carried over.
// CHECK: mogg.kernel.stub @stub_1_build_graph (%{{.*}}: !mo.tensor<[2, 3], f32, gpu:0, {{.*}}>, %{{.*}}: !mo.tensor<[2, 3], f32, gpu:0, {{.*}}>, %{{.*}}: !mgp.device_graph.builder) -> !mgp.device_graph.node
// CHECK: %[[NODE:[a-zA-Z0-9_]+]] = mogg.call.build_graph["k"] inputs(%{{.*}}) outputs(%{{.*}}) builder(%{{.*}}) -> !mgp.device_graph.node {{.*}}kernel_param = {transpose_b = true}
// CHECK: mogg.output %[[NODE]] : !mgp.device_graph.node
mogg.kernel.stub @stub_1 (%in: !mo.tensor<[2, 3], f32, gpu:0, {io = 1 : i32, mut = false}>, %out: !mo.tensor<[2, 3], f32, gpu:0, {io = 0 : i32, mut = true}>, %ctx: !mogg.context) fusionInfo = "test.k":1 attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"gpu", 0>, isAsync = true, sym_visibility = "private"} {
  mogg.with_trace {
    mogg.call.execute["k"] inputs(%in, %ctx : !mo.tensor<[2, 3], f32, gpu:0, {io = 1 : i32, mut = false}>, !mogg.context) outputs(%out : !mo.tensor<[2, 3], f32, gpu:0, {io = 0 : i32, mut = true}>) {isThrows, kernel_param = {transpose_b = true}, moOpName = "test.k"}
  }
  mogg.output
}

// CHECK-LABEL: mgp.model @dg
// The staged call is retargeted to the sibling, with the staging builder block
// arg threaded in as the launch operand and a node result surfaced.
// CHECK: ^bb0(%[[BUILDER:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder,
// CHECK: mogg.call.stub @stub_1_build_graph %{{.*}}, %{{.*}}, %[[BUILDER]] : {{.*}}, !mgp.device_graph.builder -> !mgp.device_graph.node
mgp.model @dg : mof devices [#M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.staging inputs(%gpu : !mgp.device_context<<"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder, %gpu_a: !mogg.async<!mgp.device_context<<"gpu", 0>>>):
    %gpu_u = mogg.async.unpack(%gpu_a) : (!mogg.async<!mgp.device_context<<"gpu", 0>>>) -> !mgp.device_context<<"gpu", 0>>
    %node = mgp.device_graph.dependencies dependencies() {
      %t_in = "test.tensor_in"() : () -> !mo.tensor<[2, 3], f32, gpu:0, {io = 1 : i32, mut = false}>
      %t_out = "test.tensor_out"() : () -> !mo.tensor<[2, 3], f32, gpu:0, {io = 0 : i32, mut = true}>
      mogg.call.stub @stub_1 %t_in, %t_out, %gpu_u : !mo.tensor<[2, 3], f32, gpu:0, {io = 1 : i32, mut = false}>, !mo.tensor<[2, 3], f32, gpu:0, {io = 0 : i32, mut = true}>, !mgp.device_context<<"gpu", 0>>
      mgp.device_graph.dependencies.yield
    } : () -> !mgp.device_graph.node
    mgp.device_graph.staging.yield
  }
  mef.output %ch0 : !mef.chain
} {isDeviceGraph}

// -----

// A model WITHOUT `isDeviceGraph` is left untouched: no sibling stub is generated
// and the staged call still targets `@stub_1`.

#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>

// CHECK-NOT: @stub_1_build_graph
mogg.kernel.decl @k registration = "test.k" device_info = <"", "", "", ""> execute = #kf build_graph = #kf struct_params = [] is_user_defined = false

mogg.kernel.stub @stub_1 (%in: !mo.tensor<[2, 3], f32, gpu:0, {io = 1 : i32, mut = false}>, %out: !mo.tensor<[2, 3], f32, gpu:0, {io = 0 : i32, mut = true}>, %ctx: !mogg.context) fusionInfo = "test.k":1 attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"gpu", 0>, isAsync = true, sym_visibility = "private"} {
  mogg.with_trace {
    mogg.call.execute["k"] inputs(%in, %ctx : !mo.tensor<[2, 3], f32, gpu:0, {io = 1 : i32, mut = false}>, !mogg.context) outputs(%out : !mo.tensor<[2, 3], f32, gpu:0, {io = 0 : i32, mut = true}>) {isThrows, kernel_param = {transpose_b = true}, moOpName = "test.k"}
  }
  mogg.output
}

// CHECK-LABEL: mgp.model @not_dg
// CHECK: mogg.call.stub @stub_1 %{{.*}}, %{{.*}}, %{{.*}} : {{.*}}, !mgp.device_context<<"gpu", 0>>
mgp.model @not_dg : mof devices [#M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.staging inputs(%gpu : !mgp.device_context<<"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder, %gpu_a: !mogg.async<!mgp.device_context<<"gpu", 0>>>):
    %gpu_u = mogg.async.unpack(%gpu_a) : (!mogg.async<!mgp.device_context<<"gpu", 0>>>) -> !mgp.device_context<<"gpu", 0>>
    %node = mgp.device_graph.dependencies dependencies() {
      %t_in = "test.tensor_in"() : () -> !mo.tensor<[2, 3], f32, gpu:0, {io = 1 : i32, mut = false}>
      %t_out = "test.tensor_out"() : () -> !mo.tensor<[2, 3], f32, gpu:0, {io = 0 : i32, mut = true}>
      mogg.call.stub @stub_1 %t_in, %t_out, %gpu_u : !mo.tensor<[2, 3], f32, gpu:0, {io = 1 : i32, mut = false}>, !mo.tensor<[2, 3], f32, gpu:0, {io = 0 : i32, mut = true}>, !mgp.device_context<<"gpu", 0>>
      mgp.device_graph.dependencies.yield
    } : () -> !mgp.device_graph.node
    mgp.device_graph.staging.yield
  }
  mef.output %ch0 : !mef.chain
}
