// RUN: mo-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect

//
// mgp.device_graph.create: block arg count must equal
//   1 (builder) + |inputs| + |outputs|.
//

func.func @bad_block_arg_count(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.buffer<<"gpu", 0>>,
    %out: !mgp.buffer<<"gpu", 0>>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.create' op build body has 1 block argument(s), expected 3}}
  %ch1, %graph = mgp.device_graph.create
      %ch, %ctx[3], %dctx
      inputs(%in : !mgp.buffer<<"gpu", 0>>)
      outputs(%out : !mgp.buffer<<"gpu", 0>>) {
    ^bb0(%builder: !mgp.device_graph.builder):
      mgp.device_graph.yield
  } : !mgp.device_context<<"gpu", 0>>
      -> !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>
  return %ch1 : !mef.chain
}

// -----

//
// mgp.device_graph.create: block arg type mismatch.
//

func.func @bad_block_arg_type(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.buffer<<"gpu", 0>>,
    %out: !mgp.buffer<<"gpu", 0>>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.create' op build body argument #1 has type '!mgp.buffer<<"gpu", 1>>', expected '!mgp.buffer<<"gpu", 0>>' or a `!mogg.async` of it}}
  %ch1, %graph = mgp.device_graph.create
      %ch, %ctx[3], %dctx
      inputs(%in : !mgp.buffer<<"gpu", 0>>)
      outputs(%out : !mgp.buffer<<"gpu", 0>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %in_b: !mgp.buffer<<"gpu", 1>>,
         %out_b: !mgp.buffer<<"gpu", 0>>):
      mgp.device_graph.yield
  } : !mgp.device_context<<"gpu", 0>>
      -> !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>
  return %ch1 : !mef.chain
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
    %in: !mgp.buffer<<"gpu", 0>>,
    %out: !mgp.buffer<<"gpu", 0>>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.create' op result `!mgp.device_graph` signature inputs do not match the op's `inputs` operand types}}
  %ch1, %graph = mgp.device_graph.create
      %ch, %ctx[3], %dctx
      inputs(%in : !mgp.buffer<<"gpu", 0>>)
      outputs(%out : !mgp.buffer<<"gpu", 0>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %in_b: !mgp.buffer<<"gpu", 0>>,
         %out_b: !mgp.buffer<<"gpu", 0>>):
      mgp.device_graph.yield
  } : !mgp.device_context<<"gpu", 0>>
      -> !mgp.device_graph<() -> (!mgp.buffer<<"gpu", 0>>)>
  return %ch1 : !mef.chain
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
    %bad_in: !mgp.buffer<<"cpu", 0>>,
    %out: !mgp.buffer<<"gpu", 0>>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.execute' op `inputs` operand types do not match the graph's signature inputs}}
  %ch1 = mgp.device_graph.execute
      %ch, %graph, %dctx
      inputs(%bad_in : !mgp.buffer<<"cpu", 0>>)
      outputs(%out : !mgp.buffer<<"gpu", 0>>)
      : !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>,
        !mgp.device_context<<"gpu", 0>>
  return %ch1 : !mef.chain
}

// -----

//
// mgp.device_graph.execute: outputs operand types must match graph signature
//   results.
//

func.func @execute_bad_outputs(
    %ch: !mef.chain,
    %graph: !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.buffer<<"gpu", 0>>,
    %bad_out: !mgp.buffer<<"cpu", 0>>) -> !mef.chain {
  // expected-error@+1 {{'mgp.device_graph.execute' op `outputs` operand types do not match the graph's signature results}}
  %ch1 = mgp.device_graph.execute
      %ch, %graph, %dctx
      inputs(%in : !mgp.buffer<<"gpu", 0>>)
      outputs(%bad_out : !mgp.buffer<<"cpu", 0>>)
      : !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>,
        !mgp.device_context<<"gpu", 0>>
  return %ch1 : !mef.chain
}
