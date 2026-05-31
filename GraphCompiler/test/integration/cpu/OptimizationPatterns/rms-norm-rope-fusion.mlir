// COM: Test RMSNorm + RoPE fusion pattern.
// RUN: mo-opt %s --mogg-pattern-fusion | FileCheck %s

// ============================================================
// Positive tests — pattern should fire and produce reduce.rms_norm.RoPE
// ============================================================

// Basic RoPE on GPU, no casts.
// CHECK-LABEL: mo.graph @rope_basic
// CHECK:       mo.reduce.rms_norm.RoPE
mo.graph @rope_basic<seq_len>(
  %input:    !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %weight:   !mo.tensor<[4], f32, gpu:0>,
  %epsilon:  !mo.tensor<[], f32>,
  %offset:   !mo.tensor<[], f32>,
  %cos_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %sin_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>
) -> !mo.tensor<[seq_len, 4], f32, gpu:0> {
  %split_sizes = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %normed = mo.reduce.rms_norm(%input, %weight, %epsilon, %offset)
    {multiply_before_cast = false} :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
     !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %x1, %x2 = mo.split(%normed, %split_sizes) {axis = 1 : index} :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[2], si64>)
    -> (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
  %neg_x2 = mo.negative(%x2) : !mo.tensor<[seq_len, 2], f32, gpu:0>
  %cat_axis = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  %rotated = mo.concat[1](%neg_x2, %x1) :
    (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
    -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %normed_cos  = mo.mul(%normed, %cos_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  %rotated_sin = mo.mul(%rotated, %sin_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  %result = mo.add(%normed_cos, %rotated_sin) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[seq_len, 4], f32, gpu:0>
}

// RoPE with commuted add operands (rotated*sin first, normed*cos second).
// CHECK-LABEL: mo.graph @rope_commuted_add
// CHECK:       mo.reduce.rms_norm.RoPE
mo.graph @rope_commuted_add<seq_len>(
  %input:    !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %weight:   !mo.tensor<[4], f32, gpu:0>,
  %epsilon:  !mo.tensor<[], f32>,
  %offset:   !mo.tensor<[], f32>,
  %cos_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %sin_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>
) -> !mo.tensor<[seq_len, 4], f32, gpu:0> {
  %split_sizes = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %normed = mo.reduce.rms_norm(%input, %weight, %epsilon, %offset)
    {multiply_before_cast = false} :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
     !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %x1, %x2 = mo.split(%normed, %split_sizes) {axis = 1 : index} :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[2], si64>)
    -> (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
  %neg_x2 = mo.negative(%x2) : !mo.tensor<[seq_len, 2], f32, gpu:0>
  %cat_axis = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  %rotated = mo.concat[1](%neg_x2, %x1) :
    (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
    -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %normed_cos  = mo.mul(%normed, %cos_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  %rotated_sin = mo.mul(%rotated, %sin_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  // commuted: rotated_sin comes first in the add
  %result = mo.add(%rotated_sin, %normed_cos) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[seq_len, 4], f32, gpu:0>
}

// RoPE with multiply_before_cast = true.
// CHECK-LABEL: mo.graph @rope_multiply_before_cast
// CHECK:       mo.reduce.rms_norm.RoPE
mo.graph @rope_multiply_before_cast<seq_len>(
  %input:    !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %weight:   !mo.tensor<[4], f32, gpu:0>,
  %epsilon:  !mo.tensor<[], f32>,
  %offset:   !mo.tensor<[], f32>,
  %cos_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %sin_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>
) -> !mo.tensor<[seq_len, 4], f32, gpu:0> {
  %split_sizes = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %normed = mo.reduce.rms_norm(%input, %weight, %epsilon, %offset)
    {multiply_before_cast = true} :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
     !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %x1, %x2 = mo.split(%normed, %split_sizes) {axis = 1 : index} :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[2], si64>)
    -> (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
  %neg_x2 = mo.negative(%x2) : !mo.tensor<[seq_len, 2], f32, gpu:0>
  %cat_axis = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  %rotated = mo.concat[1](%neg_x2, %x1) :
    (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
    -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %normed_cos  = mo.mul(%normed, %cos_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  %rotated_sin = mo.mul(%rotated, %sin_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  %result = mo.add(%normed_cos, %rotated_sin) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[seq_len, 4], f32, gpu:0>
}

// ============================================================
// Negative tests — pattern should NOT fire
// ============================================================

// Wrong concat order: concat(x1, -x2) instead of concat(-x2, x1).
// CHECK-LABEL: mo.graph @rope_wrong_concat_order
// CHECK:       mo.reduce.rms_norm
// CHECK-NOT:   mo.reduce.rms_norm.RoPE
mo.graph @rope_wrong_concat_order<seq_len>(
  %input:    !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %weight:   !mo.tensor<[4], f32, gpu:0>,
  %epsilon:  !mo.tensor<[], f32>,
  %offset:   !mo.tensor<[], f32>,
  %cos_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %sin_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>
) -> !mo.tensor<[seq_len, 4], f32, gpu:0> {
  %split_sizes = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %normed = mo.reduce.rms_norm(%input, %weight, %epsilon, %offset)
    {multiply_before_cast = false} :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
     !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %x1, %x2 = mo.split(%normed, %split_sizes) {axis = 1 : index} :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[2], si64>)
    -> (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
  %neg_x2 = mo.negative(%x2) : !mo.tensor<[seq_len, 2], f32, gpu:0>
  %cat_axis = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  // Wrong order: (x1, neg_x2) instead of (neg_x2, x1)
  %rotated = mo.concat[1](%x1, %neg_x2) :
    (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
    -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %normed_cos  = mo.mul(%normed, %cos_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  %rotated_sin = mo.mul(%rotated, %sin_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  %result = mo.add(%normed_cos, %rotated_sin) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[seq_len, 4], f32, gpu:0>
}

// Missing mul(normed, cos) — add uses normed directly; pattern should not fire.
// CHECK-LABEL: mo.graph @rope_missing_cos_mul
// CHECK:       mo.reduce.rms_norm
// CHECK-NOT:   mo.reduce.rms_norm.RoPE
mo.graph @rope_missing_cos_mul<seq_len>(
  %input:    !mo.tensor<[seq_len, 4], f32, gpu:0>,
  %weight:   !mo.tensor<[4], f32, gpu:0>,
  %epsilon:  !mo.tensor<[], f32>,
  %offset:   !mo.tensor<[], f32>,
  %sin_vals: !mo.tensor<[seq_len, 4], f32, gpu:0>
) -> !mo.tensor<[seq_len, 4], f32, gpu:0> {
  %split_sizes = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %normed = mo.reduce.rms_norm(%input, %weight, %epsilon, %offset)
    {multiply_before_cast = false} :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
     !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  %x1, %x2 = mo.split(%normed, %split_sizes) {axis = 1 : index} :
    (!mo.tensor<[seq_len, 4], f32, gpu:0>, !mo.tensor<[2], si64>)
    -> (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
  %neg_x2 = mo.negative(%x2) : !mo.tensor<[seq_len, 2], f32, gpu:0>
  %cat_axis = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  %rotated = mo.concat[1](%neg_x2, %x1) :
    (!mo.tensor<[seq_len, 2], f32, gpu:0>, !mo.tensor<[seq_len, 2], f32, gpu:0>)
    -> !mo.tensor<[seq_len, 4], f32, gpu:0>
  // No mul(normed, cos) — use normed directly in the add
  %rotated_sin = mo.mul(%rotated, %sin_vals) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  %result = mo.add(%normed, %rotated_sin) : !mo.tensor<[seq_len, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[seq_len, 4], f32, gpu:0>
}

// Split along axis 0 instead of the last axis: pattern should not fire.
// CHECK-LABEL: mo.graph @rope_wrong_split_axis
// CHECK:       mo.reduce.rms_norm
// CHECK-NOT:   mo.reduce.rms_norm.RoPE
mo.graph @rope_wrong_split_axis(
  %input:    !mo.tensor<[4, 4], f32, gpu:0>,
  %weight:   !mo.tensor<[4], f32, gpu:0>,
  %epsilon:  !mo.tensor<[], f32>,
  %offset:   !mo.tensor<[], f32>,
  %cos_vals: !mo.tensor<[4, 4], f32, gpu:0>,
  %sin_vals: !mo.tensor<[4, 4], f32, gpu:0>
) -> !mo.tensor<[4, 4], f32, gpu:0> {
  // Split along axis 0 (not the last axis == 1).
  %split_sizes = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %normed = mo.reduce.rms_norm(%input, %weight, %epsilon, %offset)
    {multiply_before_cast = false} :
    (!mo.tensor<[4, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
     !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[4, 4], f32, gpu:0>
  %x1, %x2 = mo.split(%normed, %split_sizes) {axis = 0 : index} :
    (!mo.tensor<[4, 4], f32, gpu:0>, !mo.tensor<[2], si64>)
    -> (!mo.tensor<[2, 4], f32, gpu:0>, !mo.tensor<[2, 4], f32, gpu:0>)
  %neg_x2 = mo.negative(%x2) : !mo.tensor<[2, 4], f32, gpu:0>
  %cat_axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %rotated = mo.concat[0](%neg_x2, %x1) :
    (!mo.tensor<[2, 4], f32, gpu:0>, !mo.tensor<[2, 4], f32, gpu:0>)
    -> !mo.tensor<[4, 4], f32, gpu:0>
  %normed_cos  = mo.mul(%normed, %cos_vals) : !mo.tensor<[4, 4], f32, gpu:0>
  %rotated_sin = mo.mul(%rotated, %sin_vals) : !mo.tensor<[4, 4], f32, gpu:0>
  %result = mo.add(%normed_cos, %rotated_sin) : !mo.tensor<[4, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[4, 4], f32, gpu:0>
}
