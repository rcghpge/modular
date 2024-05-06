// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @binary_comparison_ok(%arg0: !mo.tensor<[?], f32>, %arg1: !mo.tensor<[?], f32>) -> !mo.tensor<[?], bool> {
  %1 = rmo.equal(%arg0, %arg1) : !mo.tensor<[?], f32>, !mo.tensor<[?], f32>
  mo.output %1 : !mo.tensor<[?], bool>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_comparison_ok([[VARG0:[^ ]*]]: !mo.tensor<[?], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[?], f32>) -> !mo.tensor<[?], bool> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.equal([[VARG0]], [[VARG1]]) : !mo.tensor<[?], f32>, !mo.tensor<[?], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[?], bool>

// -----

mo.graph @binary_comparison_ok_cast(%arg0: !mo.tensor<[?], si8>, %arg1: !mo.tensor<[?], bf16>) -> !mo.tensor<[?], bool> {
  %1 = rmo.not_equal(%arg0, %arg1) : !mo.tensor<[?], si8>, !mo.tensor<[?], bf16>
  mo.output %1 : !mo.tensor<[?], bool>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_comparison_ok_cast([[VARG0:[^ ]*]]: !mo.tensor<[?], si8>, [[VARG1:[^ ]*]]: !mo.tensor<[?], bf16>) -> !mo.tensor<[?], bool> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.not_equal([[VARG0]], [[VARG1]]) : !mo.tensor<[?], si8>, !mo.tensor<[?], bf16>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[?], bool>

// -----

// ui8 fits within a bf16 exactly. bf16 has 7 significand bits plus leading implicit 1 bit.
mo.graph @binary_comparison_ok_cast_ui8_bf16(%arg0: !mo.tensor<[7], ui8>, %arg1: !mo.tensor<[7], bf16>) -> !mo.tensor<[7], bool> {
    %1 = rmo.greater(%arg0, %arg1) : !mo.tensor<[7], ui8>, !mo.tensor<[7], bf16>
    mo.output %1: !mo.tensor<[7], bool>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_comparison_ok_cast_ui8_bf16([[VARG0:[^ ]*]]: !mo.tensor<[7], ui8>, [[VARG1:[^ ]*]]: !mo.tensor<[7], bf16>) -> !mo.tensor<[7], bool> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.greater([[VARG0]], [[VARG1]]) : !mo.tensor<[7], ui8>, !mo.tensor<[7], bf16>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[7], bool>

// -----

mo.graph @binary_comparison_not_ok_unranked(%arg0: !mo.tensor<?, si64>, %arg1: !mo.tensor<?, si64>) -> !mo.tensor<?, bool> {
    // expected-error@below {{rmo.greater_equal ops do not support dynamic rank. Use the MO analogue ops instead.}}
    %1 = rmo.greater_equal(%arg0, %arg1) : !mo.tensor<?, si64>, !mo.tensor<?, si64>
    mo.output %1: !mo.tensor<?, bool>
}

// -----

mo.graph @binary_comparison_not_ok_bad_cast_ui8_si8(%arg0: !mo.tensor<[7], si8>, %arg1: !mo.tensor<[7], ui8>) -> !mo.tensor<[7], bool> {
    // expected-error@below {{Failed to resolve valid dtype: Unsafe cast from ui8 to si8. Insert an explicit cast op if this conversion is wanted.}}
    %1 = rmo.equal(%arg0, %arg1) : !mo.tensor<[7], si8>, !mo.tensor<[7], ui8>
    mo.output %1: !mo.tensor<[7], bool>
}

// -----

mo.graph @binary_elemwise_dyn_static_bad_shape(%arg0: !mo.tensor<[1, ?], si64>, %arg1: !mo.tensor<[?, 5], si64>) -> !mo.tensor<[?, 5], bool> {
    // expected-error@below {{Inputs shapes [1, ?] and [?, 5] are not compatible. Dims "?" and "5" are neither equivalent nor broadcastable. Insert an explicit rebind if these dims will be equivalent at runtime.}}
    %1 = rmo.greater(%arg0, %arg1) : !mo.tensor<[1, ?], si64>, !mo.tensor<[?, 5], si64>
    mo.output %1: !mo.tensor<[?, 5], bool>
}
