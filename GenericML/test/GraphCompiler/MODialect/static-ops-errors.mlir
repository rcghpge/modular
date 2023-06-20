// RUN: mo-opt %s --split-input-file --verify-diagnostics

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// StaticBroadcastToOp
//===----------------------------------------------------------------------===//

mo.graph @static_broadcast_input_unknown(%from: !mo.tensor<[?, 3], f32>) -> !mo.tensor<[1, 3], f32> {
  // expected-error@+1 {{'mo.static.broadcast_to' op operand #0 must be parametrically known shape}}
  %res = mo.static.broadcast_to(%from) : !mo.tensor<[?, 3], f32> -> !mo.tensor<[1, 3], f32>
  mo.output %res : !mo.tensor<[1, 3], f32>
}

// -----

mo.graph @static_broadcast_res_unknown(%from: !mo.tensor<[1, 3], f32>) -> !mo.tensor<?, f32> {
  // expected-error@+1 {{'mo.static.broadcast_to' op result #0 must be parametrically known shape}}
  %res = mo.static.broadcast_to(%from) : !mo.tensor<[1, 3], f32> -> !mo.tensor<?, f32>
  mo.output %res : !mo.tensor<?, f32>
}

// -----

mo.graph @static_broadcast_input_shape_rank_res_rank(%from: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  // expected-error@+1 {{result rank must be at least the input rank}}
  %res = mo.static.broadcast_to(%from) : !mo.tensor<[1, 2, 3], f32> -> !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

//===----------------------------------------------------------------------===//
// Static_ReshapeOp
//===----------------------------------------------------------------------===//
// -----
mo.graph @static_reshape_no_inferred_dims<A, B>(%arg0: !mo.tensor<[A, B], si64>) -> !mo.tensor<[-1], si64> {
  // expected-error @below {{'mo.static.reshape' op static Reshape can not have inferred dimensions}}
  %0 = mo.static.reshape(%arg0) : !mo.tensor<[A, B], si64> -> !mo.tensor<[-1], si64>
  mo.output %0 : !mo.tensor<[-1], si64>
}

// -----
mo.graph @static_reshape_non_matching_dims(%arg0: !mo.tensor<[5, 10], si64>) -> !mo.tensor<[45], si64> {
  // expected-error @below {{'mo.static.reshape' op input and output elements do not match: 50 vs 45}}
  %0 = mo.static.reshape(%arg0) : !mo.tensor<[5, 10], si64> -> !mo.tensor<[45], si64>
  mo.output %0 : !mo.tensor<[45], si64>
}
