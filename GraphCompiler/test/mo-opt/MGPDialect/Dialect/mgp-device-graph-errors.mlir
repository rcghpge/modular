// RUN: mo-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect

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
