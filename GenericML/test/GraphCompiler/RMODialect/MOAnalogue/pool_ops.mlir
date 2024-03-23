// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @max_pool_ok(%arg0: !mo.tensor<?, f32>, %fs: !mo.tensor<?, si64>, %st: !mo.tensor<?, si64>, %di: !mo.tensor<?, si64>, %pa: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @max_pool_bad_operand(%arg0: !mo.tensor<?, f32>, %fs: !mo.tensor<?, f32>, %st: !mo.tensor<?, si64>, %di: !mo.tensor<?, si64>, %pa: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  // expected-error @+1{{'rmo.max_pool' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @max_pool_bad_result(%arg0: !mo.tensor<?, f32>, %fs: !mo.tensor<?, si64>, %st: !mo.tensor<?, si64>, %di: !mo.tensor<?, si64>, %pa: !mo.tensor<?, si64>) -> !mo.tensor<?, f16> {
  // expected-error @+1{{'rmo.max_pool' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<?, f16>
  mo.output %0 : !mo.tensor<?, f16>
}

// -----

mo.graph @max_pool_ceil_ok(%arg0: !mo.tensor<?, f32>, %fs: !mo.tensor<?, si64>, %st: !mo.tensor<?, si64>, %di: !mo.tensor<?, si64>, %pa: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.max_pool_ceil_mode_true(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @avg_pool_ok(%arg0: !mo.tensor<?, f32>, %fs: !mo.tensor<?, si64>, %st: !mo.tensor<?, si64>, %di: !mo.tensor<?, si64>, %pa: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.avg_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] {count_boundary = false} : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}


// -----

mo.graph @avg_pool_ceil_ok(%arg0: !mo.tensor<?, f32>, %fs: !mo.tensor<?, si64>, %st: !mo.tensor<?, si64>, %di: !mo.tensor<?, si64>, %pa: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.avg_pool_ceil_mode_true(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] {count_boundary = false} : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
