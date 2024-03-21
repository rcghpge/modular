// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @abs_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.abs(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @abs_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.abs([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @ceil_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.ceil(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @ceil_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.ceil([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @cos_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.cos(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @cos_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.cos([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @erf_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.erf(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @erf_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.erf([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @exp_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.exp(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @exp_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.exp([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @floor_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.floor(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @floor_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.floor([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @gelu_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.gelu(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @gelu_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.gelu([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @log_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.log(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @log_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.log([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @log1p_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.log1p(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @log1p_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.log1p([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @logsoftmax_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.logsoftmax(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @logsoftmax_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.logsoftmax([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @negative_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.negative(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @negative_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.negative([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @not_ok(%arg0: !mo.tensor<?, bool>) -> !mo.tensor<?, bool> {
  %1 = rmo.not(%arg0) : (!mo.tensor<?, bool>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @not_ok([[VARG0:[^ ]*]]: !mo.tensor<?, bool>) -> !mo.tensor<?, bool> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.not([[VARG0]]) : (!mo.tensor<?, bool>) -> !mo.tensor<?, bool>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, bool>

// -----

mo.graph @relu_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.relu(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @relu_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.relu([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @round_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.round(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @round_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.round([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @roundeven_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.roundeven(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @roundeven_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.roundeven([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @rsqrt_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.rsqrt(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @rsqrt_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.rsqrt([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @sigmoid_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.sigmoid(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @sigmoid_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.sigmoid([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @sin_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.sin(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @sin_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.sin([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @sqrt_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.sqrt(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @sqrt_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.sqrt([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @tanh_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.tanh(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @tanh_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.tanh([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @trunc_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.trunc(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @trunc_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.trunc([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @not_non_bool_dtypes(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1{{'rmo.not' op operand #0 must be boolean element type, but got '!mo.tensor<?, f32>'}}
  %1 = rmo.not(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

mo.graph @trunc_mismatched_dtype(%arg0: !mo.tensor<?, si32>) -> !mo.tensor<?, f32> {
  // expected-error @+1{{Not all TensorValues has the same dtype.}}
  %1 = rmo.trunc(%arg0) : (!mo.tensor<?, si32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
