// RUN: mo-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect

//
// mgp.device_graph.dependencies: the yield operand types must match the op's
// body result types.
//

func.func @bad_dependencies_yield(%b: !mgp.device_graph.builder, %in: !mgp.buffer<<"gpu", 0>>)
    -> !mgp.device_graph.node {
  // expected-error@+1 {{operand types must match the op's body result types}}
  %n, %r = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    mgp.device_graph.dependencies.yield %in : !mgp.buffer<<"gpu", 0>>
  } : () -> (!mgp.device_graph.node, !mgp.tensor<2x3xf32, <"gpu", 0>>)
  return %n : !mgp.device_graph.node
}

// -----

//
// mgp.device_graph.dependencies: the sole body block argument must be the
// !mgp.device_graph.builder.
//

func.func @bad_dependencies_builder_type(%b: !mgp.device_graph.builder) -> !mgp.device_graph.node {
  // expected-error@+1 {{body argument #0 must be a `!mgp.device_graph.builder`}}
  %n = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%not_a_builder: !mgp.buffer<<"gpu", 0>>):
    mgp.device_graph.dependencies.yield
  }
  return %n : !mgp.device_graph.node
}

// -----

//
// mgp.device_graph.dependencies: the body must have exactly one block argument
// (the builder).
//

func.func @bad_dependencies_arg_count(%b: !mgp.device_graph.builder) -> !mgp.device_graph.node {
  // expected-error@+1 {{body has 0 block argument(s), expected exactly 1}}
  %n = mgp.device_graph.dependencies builder(%b) dependencies() {
    mgp.device_graph.dependencies.yield
  }
  return %n : !mgp.device_graph.node
}

// -----

//
// mgp.device_graph.staging: the first body block argument must be the
// !mgp.device_graph.builder.
//

func.func @bad_staging_builder_type() {
  // expected-error@+1 {{staging body argument #0 must be a `!mgp.device_graph.builder`}}
  mgp.device_graph.staging inputs() {
  ^bb0(%not_a_builder: !mgp.buffer<<"gpu", 0>>):
    mgp.device_graph.staging.yield
  }
  return
}

// -----

//
// mgp.device_graph.staging: block arg count must be 1 (builder) + |inputs|.
//

func.func @bad_staging_arg_count(%in: !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  // expected-error@+1 {{staging body has 1 block argument(s), expected 2}}
  mgp.device_graph.staging inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder):
    mgp.device_graph.staging.yield
  }
  return
}

// -----

//
// mgp.device_graph.staging: each input block arg must be `!mogg.async<T>` (or
// the plain operand type `T`).
//

func.func @bad_staging_input_type(%in: !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  // expected-error@+1 {{staging body argument #1 has type '!mgp.buffer<<"gpu", 0>>', expected '!mgp.tensor<2x3xf32, <"gpu", 0>>' or a `!mogg.async` of it}}
  mgp.device_graph.staging inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder, %bad: !mgp.buffer<<"gpu", 0>>):
    mgp.device_graph.staging.yield
  }
  return
}

// -----

//
// mgp.device_graph.staging: each yield operand must be `!mogg.async<T>` (or the
// plain result type `T`) where `T` is the corresponding result type.
//

func.func @bad_staging_yield(%in: !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  // expected-error@+1 {{`mgp.device_graph.staging.yield` operand #0 has type '!mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>', expected '!mgp.buffer<<"gpu", 0>>' or a `!mogg.async` of it}}
  %out = mgp.device_graph.staging inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder,
       %in_a: !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>):
    mgp.device_graph.staging.yield %in_a : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
  } -> !mgp.buffer<<"gpu", 0>>
  return
}
