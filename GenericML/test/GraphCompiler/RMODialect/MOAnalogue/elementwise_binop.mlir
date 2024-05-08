// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @mod_ok(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.mod(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @mod_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>, [[VARG1:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.mod([[VARG0]], [[VARG1]]) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @min_ok(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.min(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @min_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>, [[VARG1:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.min([[VARG0]], [[VARG1]]) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @max_ok(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.max(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @max_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>, [[VARG1:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.max([[VARG0]], [[VARG1]]) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @div_ok(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.div(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @div_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>, [[VARG1:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.div([[VARG0]], [[VARG1]]) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @pow_ok(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.pow(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pow_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>, [[VARG1:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.pow([[VARG0]], [[VARG1]]) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @pow_mixed_precision(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, si32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.pow(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, si32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pow_mixed_precision([[VARG0:[^ ]*]]: !mo.tensor<?, f32>, [[VARG1:[^ ]*]]: !mo.tensor<?, si32>) -> !mo.tensor<?, f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.pow([[VARG0]], [[VARG1]]) : (!mo.tensor<?, f32>, !mo.tensor<?, si32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @sub_ok(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.sub(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @sub_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>, [[VARG1:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.sub([[VARG0]], [[VARG1]]) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>

// -----

mo.graph @mod_not_ok_bad_operands(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f64>) -> !mo.tensor<?, f32> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.mod(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

mo.graph @mod_not_ok_bad_results(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f64> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.mod(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f64>
  mo.output %1 : !mo.tensor<?, f64>
}

// -----

mo.graph @min_not_ok_bad_operands(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f64>) -> !mo.tensor<?, f32> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.min(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

mo.graph @min_not_ok_bad_results(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f64> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.min(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f64>
  mo.output %1 : !mo.tensor<?, f64>
}

// -----

mo.graph @max_not_ok_bad_operands(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f64>) -> !mo.tensor<?, f32> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.max(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

mo.graph @max_not_ok_bad_results(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f64> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.max(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f64>
  mo.output %1 : !mo.tensor<?, f64>
}

// -----

mo.graph @div_not_ok_bad_operands(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f64>) -> !mo.tensor<?, f32> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.div(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

mo.graph @div_not_ok_bad_results(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f64> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.div(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f64>
  mo.output %1 : !mo.tensor<?, f64>
}

// -----

mo.graph @sub_not_ok_bad_operands(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f64>) -> !mo.tensor<?, f32> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.sub(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

mo.graph @sub_not_ok_bad_results(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f64> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.sub(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f64>
  mo.output %1 : !mo.tensor<?, f64>
}
