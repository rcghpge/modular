// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @abs_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.abs(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32>
  mo.output %1 : !mo.tensor<Sh0, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @abs_ok<[[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.abs([[VARG0]]) : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH0]], f32>

// -----

mo.graph @ceil_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.ceil(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32>
  mo.output %1 : !mo.tensor<Sh0, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @ceil_ok<[[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.ceil([[VARG0]]) : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH0]], f32>

// -----

mo.graph @cos_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.cos(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32>
  mo.output %1 : !mo.tensor<Sh0, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @cos_ok<[[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.cos([[VARG0]]) : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH0]], f32>

// -----

mo.graph @erf_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.erf(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32>
  mo.output %1 : !mo.tensor<Sh0, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @erf_ok<[[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.erf([[VARG0]]) : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH0]], f32>

// -----

mo.graph @exp_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.exp(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32>
  mo.output %1 : !mo.tensor<Sh0, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @exp_ok<[[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.exp([[VARG0]]) : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH0]], f32>

// -----

mo.graph @floor_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.floor(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32>
  mo.output %1 : !mo.tensor<Sh0, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @floor_ok<[[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.floor([[VARG0]]) : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH0]], f32>

// -----

mo.graph @log_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.log(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32>
  mo.output %1 : !mo.tensor<Sh0, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @log_ok<[[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.log([[VARG0]]) : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH0]], f32>

// -----

mo.graph @log1p_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.log1p(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32>
  mo.output %1 : !mo.tensor<Sh0, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @log1p_ok<[[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.log1p([[VARG0]]) : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH0]], f32>

// -----

mo.graph @logsoftmax_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.logsoftmax(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32>
  mo.output %1 : !mo.tensor<Sh0, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @logsoftmax_ok<[[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.logsoftmax([[VARG0]], [[ARG1]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH0]], f32>

// -----

mo.graph @negative_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.negative(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32>
  mo.output %1 : !mo.tensor<Sh0, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @negative_ok<[[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.negative([[VARG0]]) : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH0]], f32>

// -----

mo.graph @not_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, bool>) -> !mo.tensor<Sh0, bool> {
  %1 = rmo.mo.not(%arg0) : (!mo.tensor<Sh0, bool>) -> !mo.tensor<Sh0, bool>
  mo.output %1 : !mo.tensor<Sh0, bool>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @not_ok<[[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], bool>) -> !mo.tensor<[[SH0]], bool> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.not([[VARG0]]) : (!mo.tensor<[[SH0]], bool>) -> !mo.tensor<[[SH0]], bool>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH0]], bool>

// -----

mo.graph @relu_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.relu(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32>
  mo.output %1 : !mo.tensor<Sh0, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @relu_ok<[[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.relu([[VARG0]]) : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH0]], f32>

// -----

mo.graph @round_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.round(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32>
  mo.output %1 : !mo.tensor<Sh0, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @round_ok<[[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.round([[VARG0]]) : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH0]], f32>

// -----

mo.graph @rsqrt_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.rsqrt(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32>
  mo.output %1 : !mo.tensor<Sh0, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @rsqrt_ok<[[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.rsqrt([[VARG0]]) : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH0]], f32>

// -----

mo.graph @sin_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.sin(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32>
  mo.output %1 : !mo.tensor<Sh0, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @sin_ok<[[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.sin([[VARG0]]) : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH0]], f32>

// -----

mo.graph @sqrt_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.sqrt(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32>
  mo.output %1 : !mo.tensor<Sh0, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @sqrt_ok<[[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.sqrt([[VARG0]]) : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH0]], f32>

// -----

mo.graph @tanh_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.tanh(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32>
  mo.output %1 : !mo.tensor<Sh0, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @tanh_ok<[[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.tanh([[VARG0]]) : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH0]], f32>

// -----

mo.graph @trunc_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.trunc(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32>
  mo.output %1 : !mo.tensor<Sh0, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @trunc_ok<[[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.trunc([[VARG0]]) : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH0]], f32>

// -----

mo.graph @not_non_bool_dtypes<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32> {
  // expected-error @+1{{'rmo.mo.not' op operand #0 must be boolean element type, but got '!mo.tensor<Sh0, f32>'}}
  %1 = rmo.mo.not(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, f32>
  mo.output %1 : !mo.tensor<Sh0, f32>
}

// -----

mo.graph @trunc_mismatched_dtype<Sh0: !mosh.ape, Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, si32>) -> !mo.tensor<Sh1, f32> {
  // expected-error @+1{{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.trunc(%arg0) : (!mo.tensor<Sh0, si32>) -> !mo.tensor<Sh1, f32>
  mo.output %1 : !mo.tensor<Sh1, f32>
}
