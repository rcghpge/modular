// RUN: mo-opt %s --split-input-file --verify-diagnostics

// Each case isolates one GatherSumOp::verify() rejection. Positive cases
// (valid gather_sum ops) live in the shape-inference and fusion suites.

// Indices dtype: the builtin kernel is compiled for int32 only.
mo.graph @i64_indices(%input: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[2, 4], si64>) -> !mo.tensor<[2, 1, 3], f32> {
  // expected-error@below {{gather_sum indices must be int32}}
  %0 = "mo.gather_sum"(%input, %indices) : (!mo.tensor<[5, 3], f32>, !mo.tensor<[2, 4], si64>) -> !mo.tensor<[2, 1, 3], f32>
  mo.output %0 : !mo.tensor<[2, 1, 3], f32>
}

// -----

// Indices rank: the kernel strides the indices buffer as
// [batch, multi_hot].
mo.graph @rank1_indices(%input: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[4], si32>) -> !mo.tensor<[4, 1], f32> {
  // expected-error@below {{gather_sum indices must be rank 2: [batch, multi_hot]}}
  %0 = "mo.gather_sum"(%input, %indices) : (!mo.tensor<[5, 3], f32>, !mo.tensor<[4], si32>) -> !mo.tensor<[4, 1], f32>
  mo.output %0 : !mo.tensor<[4, 1], f32>
}

// -----

// Input rank: the kernel binds the table as [num_embeddings, emb_dim].
mo.graph @rank1_table(%input: !mo.tensor<[5], f32>, %indices: !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1], f32> {
  // expected-error@below {{gather_sum input must be rank 2: [num_embeddings, embedding_dim]}}
  %0 = "mo.gather_sum"(%input, %indices) : (!mo.tensor<[5], f32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1], f32>
  mo.output %0 : !mo.tensor<[2, 1], f32>
}

// -----

// Result rank: the fused op produces [batch, 1, emb] (or [batch, emb]).
mo.graph @rank4_result(%input: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, 3, 4], f32> {
  // expected-error@below {{gather_sum result must be rank 2 or 3}}
  %0 = "mo.gather_sum"(%input, %indices) : (!mo.tensor<[5, 3], f32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, 3, 4], f32>
  mo.output %0 : !mo.tensor<[2, 1, 3, 4], f32>
}

// -----

// Result shape: with static shapes the result is pinned to the kernel's
// derived strides (batch from the indices, kept axis at 1, emb dim from
// the table).
mo.graph @bad_result_shape(%input: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, 2], f32> {
  // expected-error@below {{gather_sum result must be [batch, 1, emb_dim] (or [batch, emb_dim]) matching the indices batch and table width}}
  %0 = "mo.gather_sum"(%input, %indices) : (!mo.tensor<[5, 3], f32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, 2], f32>
  mo.output %0 : !mo.tensor<[2, 1, 2], f32>
}

// -----

// Element type: the reduce keeps the table's dtype.
mo.graph @dtype_mismatch(%input: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, 3], si32> {
  // expected-error@below {{input and result must have the same element type}}
  %0 = "mo.gather_sum"(%input, %indices) : (!mo.tensor<[5, 3], f32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, 3], si32>
  mo.output %0 : !mo.tensor<[2, 1, 3], si32>
}
