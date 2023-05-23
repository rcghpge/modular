// RUN: mo-opt %s --allow-unregistered-dialect --split-input-file --verify-diagnostics

mo.graph @tensor_incorrect_syntax(%arg0: !mo.tensor<[], f32>) {
  // expected-error@+2 {{expected string}}
  // expected-error@+1 {{couldn't parse shape parameter: expected `?`, shape reference, or literal shape reference (i.e. `[...]`)}}
  %0 = mo.rebind(%arg0) : !mo.tensor<[], f32> -> !mo.tensor<*, f32>
  mo.output
}

// -----

mo.graph @constant_illegal_attr() -> !mo.tensor<[4], f32> {
  // expected-error@+1 {{'mo.constant' op attribute 'value' failed to satisfy constraint}}
  %0 = mo.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// -----

mo.graph @graph_early_output(%arg0: !mo.tensor<[], f32>) -> !mo.tensor<[], f32> {
  // expected-error @below {{'mo.output' op must be the last operation in the parent block}}
  mo.output %arg0 : !mo.tensor<[], f32>
  %0 = mo.constant {value = #M.dense_array<1, 2> : tensor<2xui32>} : !mo.tensor<[2], ui32>
}

// -----

mo.graph @if_op_missing_yield(%arg0: !mo.tensor<[2, ?, 1], f32>, %arg1: !mo.tensor<[2, ?, 1], f32>, %cond : !mo.tensor<[], bool>) -> (!mo.tensor<[2, ?, 1], f32>) {
  // expected-error@+1 {{'mo.yield' op expected 1 operands, but given 0}}
  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2, ?, 1], f32> {
    mo.yield %arg0 : !mo.tensor<[2, ?, 1], f32>
  } else {
  }
  mo.output %0 : !mo.tensor<[2, ?, 1], f32>
}

// -----

mo.graph @if_op_cond_mismatch(%arg0: !mo.tensor<[2, ?, 1], f32>, %arg1: !mo.tensor<[2, ?, 1], f32>, %cond : !mo.tensor<[5], bool>) -> (!mo.tensor<[2, ?, 1], f32>) {
  // expected-error@+1 {{'mo.if' op operand #0 must be rank 0 and boolean element type, but got '!mo.tensor<[5], bool>'}}
  %0 = mo.if(%cond : !mo.tensor<[5], bool>) -> !mo.tensor<[2, ?, 1], f32> {
    mo.yield %arg0 : !mo.tensor<[2, ?, 1], f32>
  } else {
    mo.yield %arg1 : !mo.tensor<[2, ?, 1], f32>
  }
  mo.output %0 : !mo.tensor<[2, ?, 1], f32>
}

// -----

mo.graph @if_op_type_mismatch(%arg0: !mo.tensor<[2, ?, 1], f32>, %arg1: !mo.tensor<[3, ?, 1], f32>, %cond : !mo.tensor<[], bool>) -> (!mo.tensor<[2, ?, 1], f32>) {
  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2, ?, 1], f32> {
    mo.yield %arg0 : !mo.tensor<[2, ?, 1], f32>
  } else {
    // expected-error@+1 {{'mo.yield' op operand #0 has type '!mo.tensor<[3, ?, 1], f32>' but expected '!mo.tensor<[2, ?, 1], f32>'}}
    mo.yield %arg1 : !mo.tensor<[3, ?, 1], f32>
  }
  mo.output %0 : !mo.tensor<[2, ?, 1], f32>
}

// -----

mo.graph @while_op_type_mismatch_cond(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[3], f32>, %cond : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
  %0:2 = mo.while (%arg0 : !mo.tensor<[2], f32>, %cond: !mo.tensor<[], bool>) {
    // expected-error@+1 {{'mo.yield' op operand #0 has type '!mo.tensor<[2], f32>' but expected '!mo.tensor<[], bool>'}}
    mo.yield %arg0 : !mo.tensor<[2], f32>
  } do {
    mo.yield %arg0, %cond : !mo.tensor<[2], f32>, !mo.tensor<[], bool>
  }
  mo.output %0#0 : !mo.tensor<[2], f32>
}

// -----

mo.graph @while_op_type_mismatch_body(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[3], f32>, %cond : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
  %0:3 = mo.while (%arg0 : !mo.tensor<[2], f32>, %arg1: !mo.tensor<[3], f32>, %cond: !mo.tensor<[], bool>) {
    mo.yield %cond : !mo.tensor<[], bool>
  } do {
    // expected-error@+1 {{'mo.yield' op operand #0 has type '!mo.tensor<[3], f32>' but expected '!mo.tensor<[2], f32>'}}
    mo.yield %arg1, %arg0, %cond : !mo.tensor<[3], f32>, !mo.tensor<[2], f32>, !mo.tensor<[], bool>
  }
  mo.output %0#0 : !mo.tensor<[2], f32>
}

// -----

// expected-error @below {{'mo.graph' op expects regions to end with 'mo.output', found 'graph.return'}}
// expected-note @below {{in custom textual format, the absence of terminator implies 'mo.output'}}
mo.graph @graph_kgen_return(%arg0: !mo.tensor<[], f32>) -> !mo.tensor<[], f32> {
  "graph.return"(%arg0) : (!mo.tensor<[], f32>) -> ()
}

// -----

mo.graph @graph_multiple_output(%arg0: !mo.tensor<[], f32>) -> !mo.tensor<[], f32> {
  // expected-error @below {{'mo.output' op must be the last operation in the parent block}}
  mo.output %arg0 : !mo.tensor<[], f32>
  mo.output %arg0 : !mo.tensor<[], f32>
}

// -----

mo.graph @rebind_different_static_ranks<D0, D1, D2>(
  %arg1: !mo.tensor<[D0], f32>,
  %arg2: !mo.tensor<[D1, D2], f32>
) -> !mo.tensor<[D0], f32> {
  // expected-error @below {{input and output static ranks must match}}
  %0 = mo.rebind(%arg1) : !mo.tensor<[D0], f32> -> !mo.tensor<[D1, D2], f32>
  mo.output %arg1: !mo.tensor<[D0], f32>
}

// -----

mo.graph @rebind_different_static_dimensions<D0, D1>(
  %arg1: !mo.tensor<[D0, 42, D1], f32>
) -> !mo.tensor<[D0, 42, D1], f32> {
  %0 = mo.rebind(%arg1) : !mo.tensor<[D0, 42, D1], f32> -> !mo.tensor<[D1, 42, D0], f32>
  // expected-error @below {{statically known dimensions must match}}
  %1 = mo.rebind(%arg1) : !mo.tensor<[D0, 42, D1], f32> -> !mo.tensor<[D1, 24, D0], f32>
  mo.output %arg1: !mo.tensor<[D0, 42, D1], f32>
}

// -----

mo.graph @rebind_same_in_param_bind_to_different_ints<D0>(
  %arg1: !mo.tensor<[D0, D0], f32>
) -> !mo.tensor<[D0, D0], f32> {
  %0 = mo.rebind(%arg1) : !mo.tensor<[D0, D0], f32> -> !mo.tensor<[42, 42], f32>
  // expected-error @below {{one dimension parameter cannot bind to different static dimensions}}
  %1 = mo.rebind(%arg1) : !mo.tensor<[D0, D0], f32> -> !mo.tensor<[42, 24], f32>
  mo.output %arg1: !mo.tensor<[D0, D0], f32>
}

// -----

mo.graph @rebind_same_out_param_bind_to_different_ints<D0>(
  %arg1: !mo.tensor<[42, 42], f32>,
  %arg2: !mo.tensor<[42, 24], f32>,
  %arg3: !mo.tensor<[D0], f32>
) -> !mo.tensor<[D0], f32> {
  %0 = mo.rebind(%arg1) : !mo.tensor<[42, 42], f32> -> !mo.tensor<[D0, D0], f32>
  // expected-error @below {{one dimension parameter cannot bind to different static dimensions}}
  %1 = mo.rebind(%arg2) : !mo.tensor<[42, 24], f32> -> !mo.tensor<[D0, D0], f32>
  mo.output %arg3: !mo.tensor<[D0], f32>
}
