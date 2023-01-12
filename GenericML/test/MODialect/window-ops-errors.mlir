// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @max_pool_filter_rank(%arg0: !mo.tensor<10x10xf32>) -> !mo.tensor<5x5x3xf32> {
  %fs = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<2xsi64>
  %st = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<2xsi64>
  // expected-error@+1 {{input and output ranks must match if static}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %st] : (
    !mo.tensor<10x10xf32>, !mo.tensor<2xsi64>, !mo.tensor<2xsi64>, !mo.tensor<2xsi64>) -> !mo.tensor<5x5x3xf32>
  mo.output %0 : !mo.tensor<5x5x3xf32>
}

// -----

mo.graph @max_pool_filter_rank(%arg0: !mo.tensor<10x10xf32>) -> !mo.tensor<5x5xf32> {
  %fs = mo.constant {value = #M.dense_array<2, 2> : tensor<2x1xsi64>} : !mo.tensor<2x1xsi64>
  %st = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<2xsi64>
  // expected-error@+1 {{'mo.max_pool' op operand #1 must be dim 1 or unranked with signed 64 integer elements}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %st] : (
    !mo.tensor<10x10xf32>, !mo.tensor<2x1xsi64>, !mo.tensor<2xsi64>, !mo.tensor<2xsi64>) -> !mo.tensor<5x5xf32>
  mo.output %0 : !mo.tensor<5x5xf32>
}

// -----

mo.graph @max_pool_strides_el_type(%arg0: !mo.tensor<10x10xf32>) -> !mo.tensor<5x5xf32> {
  %fs = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<2xsi64>
  %st = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi32>} : !mo.tensor<2xsi32>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<2xsi64>
  // expected-error@+1 {{'mo.max_pool' op operand #2 must be dim 1 or unranked with signed 64 integer elements}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di] : (
    !mo.tensor<10x10xf32>, !mo.tensor<2xsi64>, !mo.tensor<2xsi32>, !mo.tensor<2xsi64>) -> !mo.tensor<5x5xf32>
  mo.output %0 : !mo.tensor<5x5xf32>
}

// -----

mo.graph @max_pool_filter_size_mismatch(%arg0: !mo.tensor<10x10x3xf32>) -> !mo.tensor<5x5x3xf32> {
  %fs = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<2xsi64>
  %c1 = mo.constant {value = #M.dense_array<1, 1, 1> : tensor<3xsi64>} : !mo.tensor<3xsi64>
  // expected-error@+1 {{filter_shape with static size must match rank of input}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %c1, dilations = %c1] : (
    !mo.tensor<10x10x3xf32>, !mo.tensor<2xsi64>, !mo.tensor<3xsi64>, !mo.tensor<3xsi64>) -> !mo.tensor<5x5x3xf32>
  mo.output %0 : !mo.tensor<5x5x3xf32>
}

// -----

mo.graph @max_pool_strides_size_mismatch(%arg0: !mo.tensor<10x10x3xf32>) -> !mo.tensor<5x5x3xf32> {
  %fs = mo.constant {value = #M.dense_array<2, 2, 1> : tensor<3xsi64>} : !mo.tensor<3xsi64>
  %st = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<2xsi64>
  // expected-error@+1 {{strides with static size must match rank of input}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %fs] : (
    !mo.tensor<10x10x3xf32>, !mo.tensor<3xsi64>, !mo.tensor<2xsi64>, !mo.tensor<3xsi64>) -> !mo.tensor<5x5x3xf32>
  mo.output %0 : !mo.tensor<5x5x3xf32>
}

// -----

mo.graph @max_pool_dilations_size_mismatch(%arg0: !mo.tensor<10x10x3xf32>) -> !mo.tensor<5x5x3xf32> {
  %fs = mo.constant {value = #M.dense_array<2, 2, 1> : tensor<3xsi64>} : !mo.tensor<3xsi64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<2xsi64>
  // expected-error@+1 {{dilations with static size must match rank of input}}
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %fs, dilations = %di] : (
    !mo.tensor<10x10x3xf32>, !mo.tensor<3xsi64>, !mo.tensor<3xsi64>, !mo.tensor<2xsi64>) -> !mo.tensor<5x5x3xf32>
  mo.output %0 : !mo.tensor<5x5x3xf32>
}

// -----

mo.graph @max_pool_window_param_size_mismatch(
  %arg0: !mo.tensor<*f32>, %c1: !mo.tensor<3xsi64>, %di: !mo.tensor<2xsi64>
) -> !mo.tensor<*f32> {
  // expected-error@+1 {{window parameter sizes must match if static}}
  %0 = mo.max_pool(%arg0) [filter_shape = %c1, strides = %c1, dilations = %di] : (
    !mo.tensor<*f32>, !mo.tensor<3xsi64>, !mo.tensor<3xsi64>, !mo.tensor<2xsi64>) -> !mo.tensor<*f32>
  mo.output %0 : !mo.tensor<*f32>
}

// -----

mo.graph @conv_window_param_size_mismatch(
  %arg0: !mo.tensor<*f32>, %arg1: !mo.tensor<*f32>, %st: !mo.tensor<3xsi64>, %di: !mo.tensor<2xsi64>
) -> !mo.tensor<*f32> {
  // expected-error@+1 {{window parameter sizes must match if static}}
  %0 = mo.conv(%arg0, %arg1) [strides = %st, dilations = %di] : (
    !mo.tensor<*f32>, !mo.tensor<*f32>, !mo.tensor<3xsi64>, !mo.tensor<2xsi64>) -> !mo.tensor<*f32>
  mo.output %0 : !mo.tensor<*f32>
}
