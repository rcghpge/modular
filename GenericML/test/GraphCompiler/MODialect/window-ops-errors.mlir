// RUN: mo-opt %s --split-input-file --verify-diagnostics

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// ConvOp
//===----------------------------------------------------------------------===//

mo.graph @conv_window_param_size_mismatch(
  %arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>, %st: !mo.tensor<[3], si64>, %di: !mo.tensor<[2], si64>
) -> !mo.tensor<?, f32> {
  // expected-error@+1 {{window parameter sizes must match if static}}
  %0 = mo.conv(%arg0, %arg1) [strides = %st, dilations = %di] : (
    !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

//===----------------------------------------------------------------------===//
// MaxPoolOp
//===----------------------------------------------------------------------===//

mo.graph @max_pool_filter_rank(%arg0: !mo.tensor<[20, 10, 10, 32], f32>) -> !mo.tensor<[5, 5, 3], f32> {
  %fs = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %st = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // expected-error @below {{input and output ranks must be all dynamic, or all static and match}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %st] : (
    !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, 5, 3], f32>
  mo.output %0 : !mo.tensor<[5, 5, 3], f32>
}

// -----

mo.graph @max_pool_filter_rank(%arg0: !mo.tensor<[20, 10, 10, 32], f32>) -> !mo.tensor<[5, 5], f32> {
  %fs = mo.constant {value = #M.dense_array<2, 2> : tensor<2x1xsi64>} : !mo.tensor<[2, 1], si64>
  %st = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // expected-error@+1 {{'mo.max_pool' op operand #1 must be rank 1 or unranked with signed 64 integer elements}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %st] : (
    !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2, 1], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, 5], f32>
  mo.output %0 : !mo.tensor<[5, 5], f32>
}

// -----

mo.graph @max_pool_strides_el_type(%arg0: !mo.tensor<[20, 10, 10, 32], f32>) -> !mo.tensor<[5, 5], f32> {
  %fs = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %st = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // expected-error@+1 {{'mo.max_pool' op operand #2 must be rank 1 or unranked with signed 64 integer elements}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di] : (
    !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, 5], f32>
  mo.output %0 : !mo.tensor<[5, 5], f32>
}

// -----

mo.graph @max_pool_filter_size_mismatch(%arg0: !mo.tensor<[10, 10, 3], f32>) -> !mo.tensor<[5, 5, 3], f32> {
  %fs = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %c1 = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  // expected-error@+1 {{filter_shape with static size must match input rank - 2}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %c1, dilations = %c1] : (
    !mo.tensor<[10, 10, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[5, 5, 3], f32>
  mo.output %0 : !mo.tensor<[5, 5, 3], f32>
}

// -----

mo.graph @max_pool_strides_size_mismatch(%arg0: !mo.tensor<[10, 10, 3], f32>) -> !mo.tensor<[5, 5, 3], f32> {
  %fs = mo.constant {value = #M.dense_array<2, 2, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %st = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // expected-error@+1 {{strides with static size must match input rank - 2}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %fs] : (
    !mo.tensor<[10, 10, 3], f32>, !mo.tensor<[3], si64>, !mo.tensor<[2], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[5, 5, 3], f32>
  mo.output %0 : !mo.tensor<[5, 5, 3], f32>
}

// -----

mo.graph @max_pool_dilations_size_mismatch(%arg0: !mo.tensor<[10, 10, 3], f32>) -> !mo.tensor<[5, 5, 3], f32> {
  %fs = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // expected-error@+1 {{dilations with static size must match input rank - 2}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %fs, dilations = %di] : (
    !mo.tensor<[10, 10, 3], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, 5, 3], f32>
  mo.output %0 : !mo.tensor<[5, 5, 3], f32>
}

// -----

mo.graph @max_pool_window_param_size_mismatch(
  %arg0: !mo.tensor<?, f32>, %c1: !mo.tensor<[3], si64>, %di: !mo.tensor<[2], si64>
) -> !mo.tensor<?, f32> {
  // expected-error@+1 {{window parameter sizes must match if static}}
  %0 = mo.max_pool(%arg0) [filter_shape = %c1, strides = %c1, dilations = %di] : (
    !mo.tensor<?, f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @max_pool_partial_dyn_rank(
    %arg0: !mo.tensor<[20, 10, 10, 32], f32>, %fs: !mo.tensor<?, si64>, %st: !mo.tensor<?, si64>, %di: !mo.tensor<?, si64>
  ) -> !mo.tensor<?, f32> {
  // expected-error @below {{input and output ranks must be all dynamic, or all static and match}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di] : (
    !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
