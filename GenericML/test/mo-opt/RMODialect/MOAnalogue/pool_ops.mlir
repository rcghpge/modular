// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @max_pool_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape -> Sh5: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %fs: !mo.tensor<Sh1, si64>, %st: !mo.tensor<Sh2, si64>, %di: !mo.tensor<Sh3, si64>, %pa: !mo.tensor<Sh4, si64>) -> !mo.tensor<Sh5, f32> {
  %0 = rmo.mo.max_pool<() -> Sh6: !mosh.ape>(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, si64>, !mo.tensor<Sh4, si64>) -> !mo.tensor<Sh6, f32>
  mo.output<:!mosh.ape Sh6> %0 : !mo.tensor<Sh6, f32>
}

// -----

mo.graph @max_pool_bad_operand<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %fs: !mo.tensor<Sh1, f32>, %st: !mo.tensor<Sh2, si64>, %di: !mo.tensor<Sh3, si64>, %pa: !mo.tensor<Sh4, si64>) -> !mo.tensor<Sh5, f32> {
  // expected-error @+1{{'rmo.mo.max_pool' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<Sh1, f32>'}}
  %0 = rmo.mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, si64>, !mo.tensor<Sh4, si64>) -> !mo.tensor<Sh5, f32>
  mo.output %0 : !mo.tensor<Sh5, f32>
}

// -----

mo.graph @max_pool_bad_result<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %fs: !mo.tensor<Sh1, si64>, %st: !mo.tensor<Sh2, si64>, %di: !mo.tensor<Sh3, si64>, %pa: !mo.tensor<Sh4, si64>) -> !mo.tensor<Sh5, f16> {
  // expected-error @+1{{'rmo.mo.max_pool' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, si64>, !mo.tensor<Sh4, si64>) -> !mo.tensor<Sh5, f16>
  mo.output %0 : !mo.tensor<Sh5, f16>
}

// -----

mo.graph @max_pool_ceil_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape -> Sh5: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %fs: !mo.tensor<Sh1, si64>, %st: !mo.tensor<Sh2, si64>, %di: !mo.tensor<Sh3, si64>, %pa: !mo.tensor<Sh4, si64>) -> !mo.tensor<Sh5, f32> {
  %0 = rmo.mo.max_pool_ceil_mode_true<() -> Sh6: !mosh.ape>(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, si64>, !mo.tensor<Sh4, si64>) -> !mo.tensor<Sh6, f32>
  mo.output<:!mosh.ape Sh6> %0 : !mo.tensor<Sh6, f32>
}

// -----

mo.graph @avg_pool_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape -> Sh5: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %fs: !mo.tensor<Sh1, si64>, %st: !mo.tensor<Sh2, si64>, %di: !mo.tensor<Sh3, si64>, %pa: !mo.tensor<Sh4, si64>) -> !mo.tensor<Sh5, f32> {
  %0 = rmo.mo.avg_pool<() -> Sh6: !mosh.ape>(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] {count_boundary = false} : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, si64>, !mo.tensor<Sh4, si64>) -> !mo.tensor<Sh6, f32>
  mo.output<:!mosh.ape Sh6> %0 : !mo.tensor<Sh6, f32>
}


// -----

mo.graph @avg_pool_ceil_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape -> Sh5: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %fs: !mo.tensor<Sh1, si64>, %st: !mo.tensor<Sh2, si64>, %di: !mo.tensor<Sh3, si64>, %pa: !mo.tensor<Sh4, si64>) -> !mo.tensor<Sh5, f32> {
  %0 = rmo.mo.avg_pool_ceil_mode_true<() -> Sh6: !mosh.ape>(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] {count_boundary = false} : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, si64>, !mo.tensor<Sh4, si64>) -> !mo.tensor<Sh6, f32>
  mo.output<:!mosh.ape Sh6> %0 : !mo.tensor<Sh6, f32>
}
