// RUN: mo-opt %s --split-input-file --verify-diagnostics

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// BroadcastShapeOp
//===----------------------------------------------------------------------===//

mo.graph @broadcast_shape_f32_output(%arg0: !mo.tensor<[3], si64>, %arg1: !mo.tensor<[2], si64>) -> !mo.tensor<[2], f32> {
  // expected-error@+1 {{'mo.broadcast_shape' op result #0 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<[3], f32>'}}
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], f32>
  mo.output %res : !mo.tensor<[3], f32>
}

// -----

mo.graph @broadcast_shape_dyn_rank(%arg0: !mo.tensor<[3], si64>, %arg1: !mo.tensor<[2], si64>) -> !mo.tensor<?, si64> {
  // expected-error@+1 {{'mo.broadcast_shape' op result #0 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<?, si64>'}}
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<?, si64>
  mo.output %res : !mo.tensor<?, si64>
}

// -----

mo.graph @broadcast_shape_vector_output(%arg0: !mo.tensor<[3], si64>, %arg1: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 1], si64> {
  // expected-error@+1 {{'mo.broadcast_shape' op result #0 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<[2, 1], si64>'}}
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 1], si64>
  mo.output %res : !mo.tensor<[2, 1], si64>
}

// -----

mo.graph @broadcast_shape_f32_input(%arg0: !mo.tensor<[3], f32>, %arg1: !mo.tensor<[2], f32>) -> !mo.tensor<[3], si64> {
  // expected-error@+1 {{'mo.broadcast_shape' op operand #0 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<[3], f32>'}}
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], f32>, !mo.tensor<[2], f32>) -> !mo.tensor<[3], si64>
  mo.output %res : !mo.tensor<[3], si64>
}

// -----

mo.graph @broadcast_shape_f32_input2(%arg0: !mo.tensor<[3], si32>, %arg1: !mo.tensor<[2], f32>) -> !mo.tensor<[3], si64> {
  // expected-error@+1 {{'mo.broadcast_shape' op operand #1 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<[2], f32>'}}
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], si32>, !mo.tensor<[2], f32>) -> !mo.tensor<[3], si64>
  mo.output %res : !mo.tensor<[3], si64>
}

// -----

mo.graph @broadcast_shape_invalid_output_rank(%arg0: !mo.tensor<[3], si32>, %arg1: !mo.tensor<[2], si32>) -> !mo.tensor<[3], si32> {
  // expected-error @below {{output rank must be the max between 2 input ranks, if all are statically ranked}}
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[4], si64>
  mo.output %arg0 : !mo.tensor<[3], si32>
}

// -----

mo.graph @broadcast_shape_input_1_partially_ranked<D0>(%arg0: !mo.tensor<[D0], si32>, %arg1: !mo.tensor<[3], si32>) -> !mo.tensor<[2], si32> {
  // expected-error @below {{result rank must be no less than second input rank if both are statically ranked}}
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[D0], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[2], si32>
  mo.output %res : !mo.tensor<[2], si32>
}

// -----

mo.graph @broadcast_shape_input_2_partially_ranked<D0>(%arg0: !mo.tensor<[3], si32>, %arg1: !mo.tensor<[D0], si32>) -> !mo.tensor<[2], si32> {
  // expected-error @below {{result rank must be no less than first input rank if both are statically ranked}}
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], si32>, !mo.tensor<[D0], si32>) -> !mo.tensor<[2], si32>
  mo.output %res : !mo.tensor<[2], si32>
}

// -----

//===----------------------------------------------------------------------===//
// BroadcastToOp
//===----------------------------------------------------------------------===//

mo.graph @broadcast_to_bad_shape(%from: !mo.tensor<?, f32>, %to: !mo.tensor<[2, 3], si64>) -> !mo.tensor<?, f32> {
  // expected-error@+1 {{'mo.broadcast_to' op operand #1 must be rank 1 and signed 64 or signed 32 integer elements}}
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<?, f32>, !mo.tensor<[2, 3], si64>) -> !mo.tensor<?, f32>
  mo.output %res : !mo.tensor<?, f32>
}

// -----

mo.graph @broadcast_to_bad_type(%from: !mo.tensor<?, f32>, %to: !mo.tensor<[?], f32>) -> !mo.tensor<?, f32> {
  // expected-error@+1 {{'mo.broadcast_to' op operand #1 must be rank 1 and signed 64 or signed 32 integer elements}}
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<?, f32>, !mo.tensor<[?], f32>) -> !mo.tensor<?, f32>
  mo.output %res : !mo.tensor<?, f32>
}

// -----

mo.graph @broadcast_new_shape_size_res_unranked(
  %from: !mo.tensor<?, f32>, %to: !mo.tensor<[2], si32>
) -> !mo.tensor<?, f32> {
  // expected-error@+1 {{result must be statically ranked if new shape has static size}}
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<?, f32>, !mo.tensor<[2], si32>) -> !mo.tensor<?, f32>
  mo.output %res : !mo.tensor<?, f32>
}

// -----

mo.graph @broadcast_new_shape_size_res_ranked(
  %from: !mo.tensor<?, f32>, %to: !mo.tensor<[?], si32>
) -> !mo.tensor<[2, 3], f32> {
  // expected-error@+1 {{result must be dynamically ranked if new shape has unknown size}}
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<?, f32>, !mo.tensor<[?], si32>) -> !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

// -----

mo.graph @broadcast_new_shape_size_res_rank(
  %from: !mo.tensor<?, f32>, %to: !mo.tensor<[3], si32>
) -> !mo.tensor<[2, 3], f32> {
  // expected-error@+1 {{new shape size and output rank must match}}
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<?, f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

// -----

mo.graph @broadcast_input_shape_rank_res_rank(
  %from: !mo.tensor<[1, 2, 3], f32>, %to: !mo.tensor<[2], si32>
) -> !mo.tensor<[2, ?], f32> {
  // expected-error@+1 {{result rank must be at least the input rank}}
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, ?], f32>
  mo.output %res : !mo.tensor<[2, ?], f32>
}

// -----

mo.graph @broadcast_input_res_not_broadcastable(
  %from: !mo.tensor<[1, 2, 3], f32>, %to: !mo.tensor<[3], si32>
) -> !mo.tensor<[4, 5, ?], f32> {
  // expected-error@+1 {{input shape not broadcastable to result shape}}
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[4, 5, ?], f32>
  mo.output %res : !mo.tensor<[4, 5, ?], f32>
}

// -----

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

mo.graph @concat_no_args(%arg0: !mo.tensor<[7], f32>) -> !mo.tensor<[7], f32> {
  %ax = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // expected-error@+1 {{concat must have at least 2 input tensors}}
  %0 = mo.concat[%ax: !mo.tensor<[], si64>](%arg0) : (!mo.tensor<[7], f32>) -> !mo.tensor<[7], f32>
  mo.output %0 : !mo.tensor<[7], f32>
}

// -----

mo.graph @concat_non_int_axis(%arg0: !mo.tensor<[7], f32>, %arg1: !mo.tensor<[2], f32>) -> !mo.tensor<[9], f32> {
  %ax = mo.constant {value = #M.dense_array<0.0, 2.0> : tensor<2xf32>} : !mo.tensor<[2], f32>
  // expected-error @below {{'mo.concat' op operand #2 must be ranked scalar index tensor, but got '!mo.tensor<[2], f32>'}}
  %0 = mo.concat[%ax: !mo.tensor<[2], f32>](%arg0, %arg1) : (!mo.tensor<[7], f32>, !mo.tensor<[2], f32>) -> !mo.tensor<[9], f32>
  mo.output %0 : !mo.tensor<[9], f32>
}

// -----

mo.graph @concat_non_scalar_axis(%arg0: !mo.tensor<[7], f32>, %arg1: !mo.tensor<[2], f32>) -> !mo.tensor<[9], f32> {
  %ax = mo.constant {value = #M.dense_array<0, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // expected-error @below {{'mo.concat' op operand #2 must be ranked scalar index tensor, but got '!mo.tensor<[2], si64>'}}
  %0 = mo.concat[%ax: !mo.tensor<[2], si64>](%arg0, %arg1) : (!mo.tensor<[7], f32>, !mo.tensor<[2], f32>) -> !mo.tensor<[9], f32>
  mo.output %0 : !mo.tensor<[9], f32>
}

// -----

mo.graph @concat_non_matching_input_ranks(%arg0: !mo.tensor<[7], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[9], f32> {
  %ax = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  // expected-error @below {{input and output ranks must be all dynamic, or all static and match}}
  %0 = mo.concat[%ax: !mo.tensor<[], si64>](%arg0, %arg1) : (!mo.tensor<[7], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[9], f32>
  mo.output %0 : !mo.tensor<[9], f32>
}

// -----

mo.graph @concat_non_matching_output_rank(%arg0: !mo.tensor<[2, 2], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[9], f32> {
  %ax = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  // expected-error @below {{input and output ranks must be all dynamic, or all static and match}}
  %0 = mo.concat[%ax: !mo.tensor<[], si64>](%arg0, %arg1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[9], f32>
  mo.output %0 : !mo.tensor<[9], f32>
}

// -----

mo.graph @concat_partial_dyn_rank(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<[2, 8], f32> {
  %ax = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // expected-error @below {{input and output ranks must be all dynamic, or all static and match}}
  %0 = mo.concat[%ax: !mo.tensor<[], si64>](%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<?, f32>) -> !mo.tensor<[2, 8], f32>
  mo.output %0 : !mo.tensor<[2, 8], f32>
}

// -----

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

mo.graph @gather_scalar_param(%input: !mo.tensor<[], f32>, %indices: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32> {
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  // expected-error@+1 {{'mo.gather' op operand #0 must be unranked or non-scalar tensor, but got '!mo.tensor<[], f32>'}}
  %0 = mo.gather(%input, %indices, %axis) : (!mo.tensor<[], f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

// -----

mo.graph @gather_float_indices(%input: !mo.tensor<[2, 2], f32>, %indices: !mo.tensor<[2], f32>) -> !mo.tensor<[2, 2], f32> {
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  // expected-error@+1 {{'mo.gather' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<[2], f32>'}}
  %0 = mo.gather(%input, %indices, %axis) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

// -----

mo.graph @gather_non_scalar_axis(%input: !mo.tensor<[2, 2], f32>, %indices: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32> {
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[2], si64>
  // expected-error @below {{'mo.gather' op operand #2 must be ranked scalar index tensor, but got '!mo.tensor<[2], si64>'}}
  %0 = mo.gather(%input, %indices, %axis) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

// -----

mo.graph @gather_non_int_axis(%input: !mo.tensor<[2, 2], f32>, %indices: !mo.tensor<[2], f32>) -> !mo.tensor<[2, 2], f32> {
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[2], si64>
  // expected-error @below {{'mo.gather' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<[2], f32>'}}
  %0 = mo.gather(%input, %indices, %axis) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

// -----

mo.graph @gather_unranked_axis(%input: !mo.tensor<[2, 2], f32>, %indices: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32> {
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<?, si64>
  // expected-error @below {{'mo.gather' op operand #2 must be ranked scalar index tensor, but got '!mo.tensor<?, si64>'}}
  %0 = mo.gather(%input, %indices, %axis) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>, !mo.tensor<?, si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

// -----

mo.graph @gather_result_rank_too_big(%input: !mo.tensor<[?, ?], f32>, %indices: !mo.tensor<[?], si64>) -> !mo.tensor<[?, ?, ?], f32> {
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  // expected-error @below {{result rank must be 1 less than the sum of param and indices ranks if they are all static}}
  %0 = mo.gather(%input, %indices, %axis) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?, ?], f32>
}

// -----

mo.graph @gather_result_rank_too_small(%input: !mo.tensor<[?, ?], f32>, %indices: !mo.tensor<[?], si64>) -> !mo.tensor<[?], f32> {
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  // expected-error @below {{result rank must be 1 less than the sum of param and indices ranks if they are all static}}
  %0 = mo.gather(%input, %indices, %axis) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[?], f32>
  mo.output %0 : !mo.tensor<[?], f32>
}

// -----

mo.graph @gather_unranked_output(%input: !mo.tensor<[2, 2], f32>, %indices: !mo.tensor<[2], si64>) -> !mo.tensor<?, f32> {
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  // expected-error @below {{result must be ranked if both param and indices are ranked}}
  %0 = mo.gather(%input, %indices, %axis) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

mo.graph @pad_non_scalar_constant(%input: !mo.tensor<[2, 3], f32>, %paddings: !mo.tensor<[2, 2], si64>, %constant: !mo.tensor<[2], f32>) -> !mo.tensor<[?, ?], f32> {
  // expected-error@+1 {{'mo.pad' op operand #2 must be ranked scalar tensor, but got '!mo.tensor<[2], f32>'}}
  %res = mo.pad(%input, %paddings, %constant) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 2], si64>, !mo.tensor<[2], f32>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// -----

mo.graph @pad_float_paddings(%input: !mo.tensor<[2, 3], f32>, %paddings: !mo.tensor<[2, 2], f32>, %constant: !mo.tensor<[1], f32>) -> !mo.tensor<[?, ?], f32> {
  // expected-error@+1 {{'mo.pad' op operand #1 must be shape [N, 2] and signed 64 or signed 32 integer elements, but got '!mo.tensor<[2, 2], f32>}}
  %res = mo.pad(%input, %paddings, %constant) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[1], f32>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// -----

mo.graph @pad_invalid_shape_paddings(%input: !mo.tensor<[2, 3], f32>, %paddings: !mo.tensor<[2, 3], si64>, %constant: !mo.tensor<[1], f32>) -> !mo.tensor<[?, ?], f32> {
  // expected-error@+1 {{'mo.pad' op operand #1 must be shape [N, 2] and signed 64 or signed 32 integer elements, but got '!mo.tensor<[2, 3], si64>'}}
  %res = mo.pad(%input, %paddings, %constant) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>, !mo.tensor<[1], f32>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// -----

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//

mo.graph @range_result_rank(
  %arg0: !mo.tensor<[], si32>, %arg1: !mo.tensor<[], si32>, %arg2: !mo.tensor<[], si32>
) -> !mo.tensor<[5, 1], si32> {
  // expected-error@+1 {{'mo.range' op result #0 must be rank 1 or unranked, but got '!mo.tensor<[5, 1], si32>'}}
  %res = mo.range(%arg0, %arg1, %arg2) : (
    !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[5, 1], si32>
  mo.output %res : !mo.tensor<[5, 1], si32>
}

// -----

mo.graph @range_arg_eltype_mismatch(
  %arg0: !mo.tensor<[], si32>, %arg1: !mo.tensor<[], si32>, %arg2: !mo.tensor<[], f32>
) -> !mo.tensor<[?], si32> {
  // expected-error@+1 {{'mo.range' op requires the same element type for all operands and results}}
  %res = mo.range(%arg0, %arg1, %arg2) : (
    !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], f32>) -> !mo.tensor<[?], si32>
  mo.output %res : !mo.tensor<[?], si32>
}

// -----

mo.graph @range_res_eltype_mismatch(
  %arg0: !mo.tensor<[], si32>, %arg1: !mo.tensor<[], si32>, %arg2: !mo.tensor<[], si32>
) -> !mo.tensor<[?], f32> {
  // expected-error@+1 {{'mo.range' op requires the same element type for all operands and results}}
  %res = mo.range(%arg0, %arg1, %arg2) : (
    !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], f32>
  mo.output %res : !mo.tensor<[?], f32>
}

// -----

mo.graph @range_too_many_params<() -> K>(
  %arg0: !mo.tensor<[], f32>, %arg1: !mo.tensor<[], f32>, %arg2: !mo.tensor<[], f32>
) -> !mo.tensor<[?], f32> {
  // expected-error@+1 {{cannot declare more than one parameter}}
  %res = mo.range<() -> M, N>(%arg0, %arg1, %arg2) : (
    !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[M], f32>
  mo.output<M> %res : !mo.tensor<[M], f32>
}

// -----

mo.graph @range_not_index_param<() -> K>(
  %arg0: !mo.tensor<[], si32>, %arg1: !mo.tensor<[], si32>, %arg2: !mo.tensor<[], si32>
) -> !mo.tensor<[?], f32> {
  // expected-error@+1 {{can only declare index parameters}}
  %res = mo.range<() -> Sh1: !mo.shape>(%arg0, %arg1, %arg2) : (
    !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[K], si32>
  mo.output<K> %res : !mo.tensor<[K], si32>
}

// -----

//===----------------------------------------------------------------------===//
// ShapeOfOp
//===----------------------------------------------------------------------===//

mo.graph @shape_of_f32_output(%arg: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], f32> {
  // expected-error@+1 {{'mo.shape_of' op result #0 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<[2], f32>'}}
  %res = mo.shape_of(%arg) : (!mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], f32>
  mo.output %res : !mo.tensor<[2], f32>
}

// -----

mo.graph @shape_of_dyn_rank(%arg: !mo.tensor<[1, 3], f32>) -> !mo.tensor<?, si64> {
  // expected-error@+1 {{'mo.shape_of' op result #0 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<?, si64>'}}
  %res = mo.shape_of(%arg) : (!mo.tensor<[1, 3], f32>) -> !mo.tensor<?, si64>
  mo.output %res : !mo.tensor<?, si64>
}

// -----

mo.graph @shape_of_rank_2(%arg: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2, 1], si64> {
  // expected-error@+1 {{'mo.shape_of' op result #0 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<[2, 1], si64>'}}
  %res = mo.shape_of(%arg) : (!mo.tensor<[1, 3], f32>) -> !mo.tensor<[2, 1], si64>
  mo.output %res : !mo.tensor<[2, 1], si64>
}

// -----

mo.graph @shape_of_dyn_rank_static_res(%arg: !mo.tensor<?, f32>) -> !mo.tensor<[2], si64> {
  // expected-error@+1 {{input must have static rank if result has known size}}
  %res = mo.shape_of(%arg) : (!mo.tensor<?, f32>) -> !mo.tensor<[2], si64>
  mo.output %res : !mo.tensor<[2], si64>
}

// -----

mo.graph @shape_of_rank_mismatch(%arg: !mo.tensor<[1, ?], f32>) -> !mo.tensor<[1], si64> {
  // expected-error@+1 {{input rank and output size must match}}
  %res = mo.shape_of(%arg) : (!mo.tensor<[1, ?], f32>) -> !mo.tensor<[1], si64>
  mo.output %res : !mo.tensor<[1], si64>
}

// -----

mo.graph @shape_of_dyn_rank_static_res(%arg: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?], si64> {
  // expected-error@+1 {{input must have dynamic rank if result has unknown size}}
  %res = mo.shape_of(%arg) : (!mo.tensor<[?, ?], f32>) -> !mo.tensor<[?], si64>
  mo.output %res : !mo.tensor<[?], si64>
}

// -----

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

mo.graph @slice_non_vector_index(
  %input: !mo.tensor<[10, 10], f32>,
  %start : !mo.tensor<[1, 2], si64>,
  %stop : !mo.tensor<[2], si32>,
  %step : !mo.tensor<[2], si64>
) -> !mo.tensor<[8, 5], f32> {
  // expected-error @below {{'mo.slice' op operand #1 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<[1, 2], si64>'}}
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 10], f32>, !mo.tensor<[1, 2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[8, 5], f32>
  mo.output %res : !mo.tensor<[8, 5], f32>
}

// -----

mo.graph @slice_f32_index(
  %input: !mo.tensor<[10, 10], f32>,
  %start : !mo.tensor<[2], si64>,
  %stop : !mo.tensor<[2], f32>,
  %step : !mo.tensor<[2], si64>
) -> !mo.tensor<[8, 5], f32> {
  // expected-error @below {{'mo.slice' op operand #2 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<[2], f32>'}}
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 10], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], f32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[8, 5], f32>
  mo.output %res : !mo.tensor<[8, 5], f32>
}

// -----

mo.graph @slice_start_invalid_rank(
  %input: !mo.tensor<[10, 10], f32>,
  %start : !mo.tensor<[1], si64>,
  %stop : !mo.tensor<[2], si32>,
  %step : !mo.tensor<[2], si64>
) -> !mo.tensor<[8, 5], f32> {
  // expected-error @below {{start shape must be [inputRank] if it's static}}
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 10], f32>, !mo.tensor<[1], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[8, 5], f32>
  mo.output %res : !mo.tensor<[8, 5], f32>
}

// -----

mo.graph @slice_end_invalid_rank(
  %input: !mo.tensor<[10, 10], f32>,
  %start : !mo.tensor<[2], si64>,
  %stop : !mo.tensor<[3], si32>,
  %step : !mo.tensor<[2], si64>
) -> !mo.tensor<[8, 5], f32> {
  // expected-error @below {{stop shape must be [inputRank] if it's static}}
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 10], f32>, !mo.tensor<[2], si64>, !mo.tensor<[3], si32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[8, 5], f32>
  mo.output %res : !mo.tensor<[8, 5], f32>
}

// -----

mo.graph @slice_step_invalid_rank(
  %input: !mo.tensor<[10, 10], f32>,
  %start : !mo.tensor<[2], si64>,
  %stop : !mo.tensor<[2], si32>,
  %step : !mo.tensor<[3], si64>
) -> !mo.tensor<[8, 5], f32> {
  // expected-error @below {{step shape must be [inputRank] if it's static}}
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 10], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[3], si64>
  ) -> !mo.tensor<[8, 5], f32>
  mo.output %res : !mo.tensor<[8, 5], f32>
}

// -----

mo.graph @slice_dyn_rank_input_static_start(
  %input: !mo.tensor<?, f32>,
  %start : !mo.tensor<[2], si64>,
  %stop : !mo.tensor<[?], si32>,
  %step : !mo.tensor<[?], si64>
) -> !mo.tensor<?, f32> {
  // expected-error @below {{start shape must be dynamic if input is has dynamic rank}}
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<?, f32>, !mo.tensor<[2], si64>, !mo.tensor<[?], si32>, !mo.tensor<[?], si64>
  ) -> !mo.tensor<?, f32>
  mo.output %res : !mo.tensor<?, f32>
}

// -----

mo.graph @slice_dyn_rank_input_static_stop(
  %input: !mo.tensor<?, f32>,
  %start : !mo.tensor<[?], si64>,
  %stop : !mo.tensor<[1], si32>,
  %step : !mo.tensor<[?], si64>
) -> !mo.tensor<?, f32> {
  // expected-error @below {{stop shape must be dynamic if input is has dynamic rank}}
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<?, f32>, !mo.tensor<[?], si64>, !mo.tensor<[1], si32>, !mo.tensor<[?], si64>
  ) -> !mo.tensor<?, f32>
  mo.output %res : !mo.tensor<?, f32>
}

// -----

mo.graph @slice_dyn_rank_input_static_step(
  %input: !mo.tensor<?, f32>,
  %start : !mo.tensor<[?], si64>,
  %stop : !mo.tensor<[?], si32>,
  %step : !mo.tensor<[3], si64>
) -> !mo.tensor<?, f32> {
  // expected-error @below {{step shape must be dynamic if input is has dynamic rank}}
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<?, f32>, !mo.tensor<[?], si64>, !mo.tensor<[?], si32>, !mo.tensor<[3], si64>
  ) -> !mo.tensor<?, f32>
  mo.output %res : !mo.tensor<?, f32>
}

// -----

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

mo.graph @transpose_fp32_perm(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 2], f32> {
  %perm = mo.constant {value = #M.dense_array<1.0, 0.0> : tensor<2xf32>} : !mo.tensor<[2], f32>
  // expected-error@+1 {{'mo.transpose' op operand #1 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<[2], f32>'}}
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], f32>) -> !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}

// -----

mo.graph @transpose_2d_perm(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 2], f32> {
  %perm = mo.constant {value = #M.dense_array<1, 0> : tensor<1x2xsi64>} : !mo.tensor<[1, 2], si64>
  // expected-error@+1 {{'mo.transpose' op operand #1 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<[1, 2], si64>'}}
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[1, 2], si64>) -> !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}

// -----

mo.graph @transpose_perm_size_static(%input: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %perm = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // expected-error@+1 {{argument and result must be statically ranked if perm has known shape}}
  %res = mo.transpose(%input, %perm) : (!mo.tensor<?, f32>, !mo.tensor<[2], si64>) -> !mo.tensor<?, f32>
  mo.output %res : !mo.tensor<?, f32>
}

// -----

mo.graph @transpose_perm_size_dynamic(%input: !mo.tensor<[?, ?], f32>, %perm: !mo.tensor<[?], si64>) -> !mo.tensor<[?, ?], f32> {
  // expected-error@+1 {{argument and result must be dynamically ranked if perm has unknown shape}}
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// -----

mo.graph @transpose_arg_res_rank_mismatch(%input: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?], f32> {
  %perm = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // expected-error@+1 {{input and result ranks do not match}}
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// -----

mo.graph @transpose_perm_rank_mismatch(%input: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?], f32> {
  %perm = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // expected-error@+1 {{input and result ranks do not match perm size}}
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?, ?], f32>
}

// -----

//===----------------------------------------------------------------------===//
// WhereOp
//===----------------------------------------------------------------------===//

mo.graph @where_bad_result_rank(%arg: !mo.tensor<[7, 7], f32>) -> !mo.tensor<[?, 1, 2], si64> {
  // expected-error@+1 {{mo.where: invalid output shape: rank must be 2}}
  %res = mo.where(%arg) : (!mo.tensor<[7, 7], f32>) -> !mo.tensor<[?, 1, 2], si64>
  mo.output %res : !mo.tensor<[?, 1, 2], si64>
}

// -----

mo.graph @where_dynamic_result_rank(%arg: !mo.tensor<[7, 7], f32>) -> !mo.tensor<?, si64> {
  // expected-error@+1 {{mo.where: invalid output shape: rank must be 2}}
  %res = mo.where(%arg) : (!mo.tensor<[7, 7], f32>) -> !mo.tensor<?, si64>
  mo.output %res : !mo.tensor<?, si64>
}

// -----

mo.graph @where_dynamic_input_rank(%arg: !mo.tensor<?, f32>) -> !mo.tensor<[?, 2], si64> {
  // expected-error@+1 {{mo.where: invalid output shape: inner_dim != input_rank}}
  %res = mo.where(%arg) : (!mo.tensor<?, f32>) -> !mo.tensor<[?, 2], si64>
  mo.output %res : !mo.tensor<[?, 2], si64>
}

// -----

mo.graph @where_bad_result_dim(%arg: !mo.tensor<[7, 7], f32>) -> !mo.tensor<[?, 1], si64> {
  // expected-error@+1 {{mo.where: invalid output shape: inner_dim != input_rank}}
  %res = mo.where(%arg) : (!mo.tensor<[7, 7], f32>) -> !mo.tensor<[?, 1], si64>
  mo.output %res : !mo.tensor<[?, 1], si64>
}

// -----

mo.graph @where_fp_result_type(%arg: !mo.tensor<[7, 7], f32>) -> !mo.tensor<[?, 2], si64> {
  // expected-error@+1 {{'mo.where' op result #0 must be signed 64 integer element type, but got '!mo.tensor<[?, 2], f32>'}}
  %res = mo.where(%arg) : (!mo.tensor<[7, 7], f32>) -> !mo.tensor<[?, 2], f32>
  mo.output %res : !mo.tensor<[?, 2], f32>
}
