// RUN: mo-opt %s --split-input-file --verify-diagnostics

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// ConvOp
//===----------------------------------------------------------------------===//

mo.graph @conv_window_param_size_mismatch<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(
  %arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>, %st: !mo.tensor<[3], si64>, %di: !mo.tensor<[2], si64>, %pa: !mo.tensor<[4], si64>, %ng: !mo.tensor<[], si64>
) -> !mo.tensor<Sh2, f32> {
  // expected-error@+1 {{window parameter sizes must match if static}}
  %0 = mo.conv(%arg0, %arg1) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
    !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<[3], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh2, f32>
  mo.output %0 : !mo.tensor<Sh2, f32>
}

// -----

//===----------------------------------------------------------------------===//
// ConvTransposeOp
//===----------------------------------------------------------------------===//

mo.graph @conv_transpose_window_param_size_mismatch<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(
  %arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>, %st: !mo.tensor<[2], si64>, %di: !mo.tensor<[2], si64>, %pa: !mo.tensor<[4], si64>, %op: !mo.tensor<[3], si64>
) -> !mo.tensor<Sh2, f32> {
  // expected-error@+1 {{window parameter sizes must match if static}}
  %0 = mo.conv_transpose(%arg0, %arg1) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<Sh2, f32>
  mo.output %0 : !mo.tensor<Sh2, f32>
}

// -----

//===----------------------------------------------------------------------===//
// MaxPoolOp
//===----------------------------------------------------------------------===//

mo.graph @max_pool_filter_rank(%arg0: !mo.tensor<[20, 10, 10, 32], f32>) -> !mo.tensor<[5, 5, 3], f32> {
  %fs = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %st = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  // expected-error @below {{input and output ranks must be all dynamic, or all static and match}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %st, paddings = %pa] : (
    !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[5, 5, 3], f32>
  mo.output %0 : !mo.tensor<[5, 5, 3], f32>
}

// -----

mo.graph @max_pool_filter_rank(%arg0: !mo.tensor<[20, 10, 10, 32], f32>) -> !mo.tensor<[5, 5], f32> {
  %fs = mo.constant {value = #M.dense_array<2, 2> : tensor<2x1xsi64>} : !mo.tensor<[2, 1], si64>
  %st = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  // expected-error@+1 {{'mo.max_pool' op operand #1 must be rank 1 or unranked with signed 64 or signed 32 integer elements}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %st, paddings = %pa] : (
    !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2, 1], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[5, 5], f32>
  mo.output %0 : !mo.tensor<[5, 5], f32>
}

// -----

mo.graph @max_pool_filter_size_mismatch(%arg0: !mo.tensor<[10, 10, 3], f32>) -> !mo.tensor<[5, 5, 3], f32> {
  %fs = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %c1 = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // expected-error@+1 {{filter_shape with static size must match input rank - 2}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %c1, dilations = %c1, paddings = %pa] : (
    !mo.tensor<[10, 10, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, 5, 3], f32>
  mo.output %0 : !mo.tensor<[5, 5, 3], f32>
}

// -----

mo.graph @max_pool_strides_size_mismatch(%arg0: !mo.tensor<[10, 10, 3], f32>) -> !mo.tensor<[5, 5, 3], f32> {
  %fs = mo.constant {value = #M.dense_array<2, 2, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %st = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // expected-error@+1 {{strides with static size must match input rank - 2}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %fs, paddings = %pa] : (
    !mo.tensor<[10, 10, 3], f32>, !mo.tensor<[3], si64>, !mo.tensor<[2], si64>, !mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, 5, 3], f32>
  mo.output %0 : !mo.tensor<[5, 5, 3], f32>
}

// -----

mo.graph @max_pool_dilations_size_mismatch(%arg0: !mo.tensor<[10, 10, 3], f32>) -> !mo.tensor<[5, 5, 3], f32> {
  %fs = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // expected-error@+1 {{dilations with static size must match input rank - 2}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %fs, dilations = %di, paddings = %pa] : (
    !mo.tensor<[10, 10, 3], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, 5, 3], f32>
  mo.output %0 : !mo.tensor<[5, 5, 3], f32>
}

// -----

mo.graph @max_pool_paddings_size_mismatch(%arg0: !mo.tensor<[10, 10, 3], f32>) -> !mo.tensor<[5, 5, 3], f32> {
  %fs = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %di = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %pa = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  // expected-error@+1 {{paddings with static size must match 2 * (input rank - 2)}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %fs, dilations = %di, paddings = %pa] : (
    !mo.tensor<[10, 10, 3], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[5, 5, 3], f32>
  mo.output %0 : !mo.tensor<[5, 5, 3], f32>
}

// -----

mo.graph @max_pool_window_param_size_mismatch<Sh0: !mosh.ape, Sh1: !mosh.ape>(
  %arg0: !mo.tensor<Sh0, f32>, %c1: !mo.tensor<[3], si64>, %di: !mo.tensor<[2], si64>, %pa: !mo.tensor<[4], si64>
) -> !mo.tensor<Sh1, f32> {
  // expected-error@+1 {{window parameter sizes must match if static}}
  %0 = mo.max_pool(%arg0) [filter_shape = %c1, strides = %c1, dilations = %di, paddings = %pa] : (
    !mo.tensor<Sh0, f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<Sh1, f32>
  mo.output %0 : !mo.tensor<Sh1, f32>
}

// -----

mo.graph @max_pool_partial_dyn_rank<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape>(
    %arg0: !mo.tensor<[20, 10, 10, 32], f32>, %fs: !mo.tensor<Sh0, si64>, %st: !mo.tensor<Sh1, si64>, %di: !mo.tensor<Sh2, si64>, %pa: !mo.tensor<Sh3, si64>
  ) -> !mo.tensor<Sh4, f32> {
  // expected-error @below {{input and output ranks must be all dynamic, or all static and match}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (
    !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<Sh0, si64>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32>
  mo.output %0 : !mo.tensor<Sh4, f32>
}
