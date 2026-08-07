// RUN: mo-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect

//
// mgp.device_graph.create: block arg count must equal
//   1 (builder) + 1 (context) + |inputs|.
//

func.func @bad_block_arg_count(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.buffer<<"gpu", 0>>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.create' op build body has 1 block argument(s), expected 3 (the builder, the context, and one per input)}}
  %graph = mgp.device_graph.create
      %ctx, %dctx
      inputs(%in : !mgp.buffer<<"gpu", 0>>) {
    ^bb0(%builder: !mgp.device_graph.builder):
      mgp.device_graph.yield
  } : !mgp.device_context<<"gpu", 0>>
      -> !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>
  return %ch : !mef.chain
}

// -----

//
// mgp.device_graph.create: block arg type mismatch.
//

func.func @bad_block_arg_type(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.buffer<<"gpu", 0>>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.create' op build body argument #2 has type '!mgp.buffer<<"gpu", 1>>', expected '!mgp.buffer<<"gpu", 0>>'}}
  %graph = mgp.device_graph.create
      %ctx, %dctx
      inputs(%in : !mgp.buffer<<"gpu", 0>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %ctx_b: !mgp.context,
         %in_b: !mgp.buffer<<"gpu", 1>>):
      mgp.device_graph.yield
  } : !mgp.device_context<<"gpu", 0>>
      -> !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>
  return %ch : !mef.chain
}

// -----

//
// mgp.device_graph.create: an `!mogg.async<T>` input block arg is rejected --
// the build-body boundary carries bare `T`, no sync/async transition.
//

func.func @bad_async_block_arg(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.buffer<<"gpu", 0>>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.create' op build body argument #2 has type '!mogg.async<!mgp.buffer<<"gpu", 0>>>', expected '!mgp.buffer<<"gpu", 0>>'}}
  %graph = mgp.device_graph.create
      %ctx, %dctx
      inputs(%in : !mgp.buffer<<"gpu", 0>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %ctx_b: !mgp.context,
         %in_b: !mogg.async<!mgp.buffer<<"gpu", 0>>>):
      mgp.device_graph.yield
  } : !mgp.device_context<<"gpu", 0>>
      -> !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>
  return %ch : !mef.chain
}

// -----

//
// mgp.device_graph.create: build body argument #1 must be the `!mgp.context`.
//

func.func @bad_context_block_arg(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.buffer<<"gpu", 0>>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.create' op build body argument #1 must be the}}
  %graph = mgp.device_graph.create
      %ctx, %dctx
      inputs(%in : !mgp.buffer<<"gpu", 0>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %ctx_b: !mgp.buffer<<"gpu", 0>>,
         %in_b: !mgp.buffer<<"gpu", 0>>):
      mgp.device_graph.yield %in_b : !mgp.buffer<<"gpu", 0>>
  } : !mgp.device_context<<"gpu", 0>>
      -> !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>
  return %ch : !mef.chain
}

// -----

//
// mgp.device_graph.create: outGraph signature inputs must match op-level
//   inputs.
//

func.func @bad_signature_inputs(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.buffer<<"gpu", 0>>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.create' op result `!mgp.device_graph` signature inputs do not match the op's `inputs` operand types}}
  %graph = mgp.device_graph.create
      %ctx, %dctx
      inputs(%in : !mgp.buffer<<"gpu", 0>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %ctx_b: !mgp.context,
         %in_b: !mgp.buffer<<"gpu", 0>>):
      mgp.device_graph.yield
  } : !mgp.device_context<<"gpu", 0>>
      -> !mgp.device_graph<() -> (!mgp.buffer<<"gpu", 0>>)>
  return %ch : !mef.chain
}

// -----

//
// mgp.device_graph.create: outGraph signature results must match the build
//   body's yielded value types.
//

func.func @bad_signature_results(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.buffer<<"gpu", 0>>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.create' op result `!mgp.device_graph` signature results do not match the build body's yielded value types}}
  %graph = mgp.device_graph.create
      %ctx, %dctx
      inputs(%in : !mgp.buffer<<"gpu", 0>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %ctx_b: !mgp.context,
         %in_b: !mgp.buffer<<"gpu", 0>>):
      mgp.device_graph.yield %in_b : !mgp.buffer<<"gpu", 0>>
  } : !mgp.device_context<<"gpu", 0>>
      -> !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"cpu", 0>>)>
  return %ch : !mef.chain
}

// -----

//
// mgp.device_graph.execute: inputs operand types must match graph signature
//   inputs.
//

func.func @execute_bad_inputs(
    %ch: !mef.chain,
    %graph: !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %bad_in: !mgp.buffer<<"cpu", 0>>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.execute' op `inputs` operand types do not match the graph's signature inputs}}
  %chain, %out = mgp.device_graph.execute
      %graph, %dctx
      inputs(%bad_in : !mgp.buffer<<"cpu", 0>>)
      : !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>,
        !mgp.device_context<<"gpu", 0>>
        -> !mef.chain, !mgp.buffer<<"gpu", 0>>
  return %ch : !mef.chain
}

// -----

//
// mgp.device_graph.execute: output result types must match graph signature
//   results.
//

func.func @execute_bad_outputs(
    %ch: !mef.chain,
    %graph: !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.buffer<<"gpu", 0>>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.execute' op `outputs` result types do not match the graph's signature results}}
  %chain, %bad_out = mgp.device_graph.execute
      %graph, %dctx
      inputs(%in : !mgp.buffer<<"gpu", 0>>)
      : !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>,
        !mgp.device_context<<"gpu", 0>>
        -> !mef.chain, !mgp.buffer<<"cpu", 0>>
  return %ch : !mef.chain
}

// -----

//
// mgp.device_graph.create: a host-placed input is rejected. `add_input`
// allocates each input's stable location from the graph's device pool and the
// replay copy treats its source as device memory, so a CPU operand cannot be
// recorded (TODO(GEX-4051)).
//

func.func @create_host_input(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.buffer<<"cpu", 0>>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.create' op input #0 is on cpu:0, but device graph inputs must be device-resident}}
  %graph = mgp.device_graph.create
      %ctx, %dctx
      inputs(%in : !mgp.buffer<<"cpu", 0>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %ctx_b: !mgp.context,
         %in_b: !mgp.buffer<<"cpu", 0>>):
      mgp.device_graph.yield
  } : !mgp.device_context<<"gpu", 0>>
      -> !mgp.device_graph<(!mgp.buffer<<"cpu", 0>>) -> ()>
  return %ch : !mef.chain
}

// -----

//
// mgp.device_graph.create: an omitted `#M.device_ref` defaults to the host, so
// it is rejected for the same reason as an explicit `cpu`.
//

func.func @create_default_device_input(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.tensor<4xf32>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.create' op input #0 is on cpu:0, but device graph inputs must be device-resident}}
  %graph = mgp.device_graph.create
      %ctx, %dctx
      inputs(%in : !mgp.tensor<4xf32>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %ctx_b: !mgp.context,
         %in_b: !mgp.tensor<4xf32>):
      mgp.device_graph.yield
  } : !mgp.device_context<<"gpu", 0>>
      -> !mgp.device_graph<(!mgp.tensor<4xf32>) -> ()>
  return %ch : !mef.chain
}

// -----

//
// mgp.device_graph.execute: the same restriction on the replay side, where the
// input copy would pick the wrong direction for host memory.
//

func.func @execute_host_input(
    %ch: !mef.chain,
    %graph: !mgp.device_graph<(!mgp.buffer<<"cpu", 0>>) -> ()>,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.buffer<<"cpu", 0>>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.execute' op input #0 is on cpu:0, but device graph inputs must be device-resident}}
  %chain = mgp.device_graph.execute
      %graph, %dctx
      inputs(%in : !mgp.buffer<<"cpu", 0>>)
      : !mgp.device_graph<(!mgp.buffer<<"cpu", 0>>) -> ()>,
        !mgp.device_context<<"gpu", 0>>
        -> !mef.chain
  return %ch : !mef.chain
}
