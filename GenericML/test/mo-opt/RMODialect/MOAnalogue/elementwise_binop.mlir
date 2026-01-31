// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @mod_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.mo.mod(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @mod_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<Sh0, f32>, [[VARG1:[^ ]*]]: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.mod([[VARG0]], [[VARG1]]) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<Sh2, f32>

// -----

mo.graph @min_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.mo.min(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @min_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<Sh0, f32>, [[VARG1:[^ ]*]]: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.min([[VARG0]], [[VARG1]]) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<Sh2, f32>

// -----

mo.graph @max_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.mo.max(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @max_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<Sh0, f32>, [[VARG1:[^ ]*]]: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.max([[VARG0]], [[VARG1]]) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<Sh2, f32>

// -----

mo.graph @div_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.mo.div(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @div_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<Sh0, f32>, [[VARG1:[^ ]*]]: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.div([[VARG0]], [[VARG1]]) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<Sh2, f32>

// -----

mo.graph @pow_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.mo.pow(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pow_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<Sh0, f32>, [[VARG1:[^ ]*]]: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.pow([[VARG0]], [[VARG1]]) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<Sh2, f32>

// -----

mo.graph @pow_mixed_precision<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.mo.pow(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pow_mixed_precision<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<Sh0, f32>, [[VARG1:[^ ]*]]: !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.pow([[VARG0]], [[VARG1]]) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<Sh2, f32>

// -----

mo.graph @sub_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.mo.sub(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @sub_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<Sh0, f32>, [[VARG1:[^ ]*]]: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.sub([[VARG0]], [[VARG1]]) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<Sh2, f32>

// -----

mo.graph @mod_not_ok_bad_operands<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f64>) -> !mo.tensor<Sh2, f32> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.mod(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f64>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @mod_not_ok_bad_results<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f64> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.mod(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f64>
  mo.output %1 : !mo.tensor<Sh2, f64>
}

// -----

mo.graph @min_not_ok_bad_operands<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f64>) -> !mo.tensor<Sh2, f32> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.min(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f64>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @min_not_ok_bad_results<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f64> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.min(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f64>
  mo.output %1 : !mo.tensor<Sh2, f64>
}

// -----

mo.graph @max_not_ok_bad_operands<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f64>) -> !mo.tensor<Sh2, f32> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.max(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f64>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @max_not_ok_bad_results<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f64> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.max(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f64>
  mo.output %1 : !mo.tensor<Sh2, f64>
}

// -----

mo.graph @div_not_ok_bad_operands<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f64>) -> !mo.tensor<Sh2, f32> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.div(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f64>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @div_not_ok_bad_results<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f64> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.div(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f64>
  mo.output %1 : !mo.tensor<Sh2, f64>
}

// -----

mo.graph @sub_not_ok_bad_operands<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f64>) -> !mo.tensor<Sh2, f32> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.sub(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f64>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @sub_not_ok_bad_results<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f64> {
  // expected-error@below {{Not all TensorValues have the same dtype.}}
  %1 = rmo.mo.sub(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f64>
  mo.output %1 : !mo.tensor<Sh2, f64>
}
