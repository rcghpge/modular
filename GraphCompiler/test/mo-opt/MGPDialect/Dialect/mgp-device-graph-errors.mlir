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
// mgp.device_graph.create: a host-placed raw buffer is rejected. Host inputs
// are supported only as immutable tensors (contents baked into the cache
// key); a `!mgp.buffer` should have lowered to a mut-marked tensor earlier.
//

func.func @create_host_input(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.buffer<<"cpu", 0>>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.create' op input #0 is a buffer on cpu:0, but host-placed device graph inputs must be immutable tensors}}
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
// mgp.device_graph.create: an omitted `#M.device_ref` defaults to the host.
// An immutable host tensor is a legal input: it gets a host-pool stable
// location and its contents join the graph cache key.
//

func.func @create_default_device_input(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.tensor<4xf32>) -> !mef.chain {
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
// mgp.device_graph.create: a host-placed *mutable* tensor is rejected --
// in-place writes contradict the value-keyed replay of a host input.
//

func.func @create_host_mut_input(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.tensor<4xf32, <"cpu", 0>, mut>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.create' op input #0 is a mutable tensor on cpu:0; in-place writes contradict the value-keyed replay of a host-placed input}}
  %graph = mgp.device_graph.create
      %ctx, %dctx
      inputs(%in : !mgp.tensor<4xf32, <"cpu", 0>, mut>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %ctx_b: !mgp.context,
         %in_b: !mgp.tensor<4xf32, <"cpu", 0>, mut>):
      mgp.device_graph.yield
  } : !mgp.device_context<<"gpu", 0>>
      -> !mgp.device_graph<(!mgp.tensor<4xf32, <"cpu", 0>, mut>) -> ()>
  return %ch : !mef.chain
}

// -----

//
// mgp.device_graph.create: a host tensor above the warning threshold (but
// under the cap) compiles with a diagnostic about cache growth.
//

func.func @create_host_input_large_warns(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.tensor<1024xf32, <"cpu", 0>>) -> !mef.chain {
  // expected-warning@+1 {{input #0 is a host tensor of 4096 bytes; its contents are baked into the graph cache key, so every distinct value records a new graph}}
  %graph = mgp.device_graph.create
      %ctx, %dctx
      inputs(%in : !mgp.tensor<1024xf32, <"cpu", 0>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %ctx_b: !mgp.context,
         %in_b: !mgp.tensor<1024xf32, <"cpu", 0>>):
      mgp.device_graph.yield
  } : !mgp.device_context<<"gpu", 0>>
      -> !mgp.device_graph<(!mgp.tensor<1024xf32, <"cpu", 0>>) -> ()>
  return %ch : !mef.chain
}

// -----

//
// mgp.device_graph.execute: the same raw-buffer restriction on the replay
// side.
//

func.func @execute_host_input(
    %ch: !mef.chain,
    %graph: !mgp.device_graph<(!mgp.buffer<<"cpu", 0>>) -> ()>,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.buffer<<"cpu", 0>>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.execute' op input #0 is a buffer on cpu:0, but host-placed device graph inputs must be immutable tensors}}
  %chain = mgp.device_graph.execute
      %graph, %dctx
      inputs(%in : !mgp.buffer<<"cpu", 0>>)
      : !mgp.device_graph<(!mgp.buffer<<"cpu", 0>>) -> ()>,
        !mgp.device_context<<"gpu", 0>>
        -> !mef.chain
  return %ch : !mef.chain
}

// -----

//
// mgp.device_graph.collective: block argument #0 must be a device-graph
//   builder.
//

func.func @collective_bad_builder_type() {
  // expected-error@+1 {{'mgp.device_graph.collective' op region #0 block argument #0 has type '!mgp.stable_addr<<4xf32, <"gpu", 0>>>'; expected `!mgp.device_graph.builder`}}
  %0 = mgp.device_graph.collective builder(
      %b: !mgp.stable_addr<<4xf32, <"gpu", 0>>>
  ) stable() <"gpu", 0> {
    mgp.device_graph.collective.yield
  } : !mgp.tensor<4xf32>

  return
}

// -----

//
// mgp.device_graph.collective: trailing block arguments must be stable
//   addresses.
//

func.func @collective_bad_stable_type() {
  // expected-error@+1 {{'mgp.device_graph.collective' op region #0 block argument #1 has type '!mgp.tensor<4xf32>'; expected `!mgp.stable_addr<...>`}}
  %0 = mgp.device_graph.collective builder(
      %b
  ) stable(
      %s0: !mgp.tensor<4xf32>
  ) <"gpu", 0> {
    mgp.device_graph.collective.yield %s0 : !mgp.tensor<4xf32>
  } : !mgp.tensor<4xf32>

  return
}

// -----

//
// mgp.device_graph.collective: each yield operand type must match the
//   corresponding op result type.
//

func.func @collective_bad_yield_type() {
  // expected-error@+1 {{'mgp.device_graph.collective' op region #0 yield operand type '!mgp.tensor<4xf32, <"gpu", 0>, stable>' does not match output #0 type '!mgp.tensor<4xf32>'}}
  %0 = mgp.device_graph.collective builder(
      %b
  ) stable(
      %s0: !mgp.stable_addr<<4xf32, <"gpu", 0>>>
  ) <"gpu", 0> {
    %t0 = mgp.device_graph.stable.get %s0 : !mgp.stable_addr<<4xf32, <"gpu", 0>>>
    mgp.device_graph.collective.yield %t0 : !mgp.tensor<4xf32, <"gpu", 0>, stable>
  } : !mgp.tensor<4xf32>

  return
}

// -----

//
// mgp.device_graph.collective: a region must not yield more operands than
//   the op declares outputs.
//

func.func @collective_too_many_yields() {
  %val = "test.op"() : () -> !mgp.tensor<4xf32>
  // expected-error@+1 {{'mgp.device_graph.collective' op region #0 yields more operands than declared outputs (1)}}
  %0 = mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    mgp.device_graph.collective.yield %val, %val : !mgp.tensor<4xf32>, !mgp.tensor<4xf32>
  } : !mgp.tensor<4xf32>

  return
}

// -----

//
// mgp.device_graph.stable.create: input tensor must not already be `stable`.
//

func.func @stable_create_already_stable(%t: !mgp.tensor<4xf32, stable>) -> !mgp.tensor<4xf32, stable> {
  // expected-error@+1 {{'mgp.device_graph.stable.create' op input tensor is already marked `stable`}}
  %s = mgp.device_graph.stable.create %t : !mgp.tensor<4xf32, stable>
  return %s : !mgp.tensor<4xf32, stable>
}
