// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @binary_elemwise_ok(%arg0: !mo.tensor<[?], f32>, %arg1: !mo.tensor<[?], f32>) -> !mo.tensor<[?], f32> {
  %1 = rmo.add(%arg0, %arg1) : !mo.tensor<[?], f32>, !mo.tensor<[?], f32>
  mo.output %1 : !mo.tensor<[?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_elemwise_ok([[VARG0:[^ ]*]]: !mo.tensor<[?], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[?], f32>) -> !mo.tensor<[?], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.add([[VARG0]], [[VARG1]]) : !mo.tensor<[?], f32>, !mo.tensor<[?], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[?], f32>

// -----

mo.graph @binary_elemwise_ok_cast(%arg0: !mo.tensor<[?], si8>, %arg1: !mo.tensor<[?], bf16>) -> !mo.tensor<[?], bf16> {
  %1 = rmo.sub(%arg0, %arg1) : !mo.tensor<[?], si8>, !mo.tensor<[?], bf16>
  mo.output %1 : !mo.tensor<[?], bf16>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_elemwise_ok_cast([[VARG0:[^ ]*]]: !mo.tensor<[?], si8>, [[VARG1:[^ ]*]]: !mo.tensor<[?], bf16>) -> !mo.tensor<[?], bf16> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.sub([[VARG0]], [[VARG1]]) : !mo.tensor<[?], si8>, !mo.tensor<[?], bf16>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[?], bf16>

// -----

// ui8 fits within a bf16 exactly. bf16 has 7 significand bits plus leading implicit 1 bit.
mo.graph @binary_elemwise_ok_cast_ui8_bf16(%arg0: !mo.tensor<[7], ui8>, %arg1: !mo.tensor<[7], bf16>) -> !mo.tensor<[7], bf16> {
    %1 = rmo.mul(%arg0, %arg1) : !mo.tensor<[7], ui8>, !mo.tensor<[7], bf16>
    mo.output %1: !mo.tensor<[7], bf16>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_elemwise_ok_cast_ui8_bf16([[VARG0:[^ ]*]]: !mo.tensor<[7], ui8>, [[VARG1:[^ ]*]]: !mo.tensor<[7], bf16>) -> !mo.tensor<[7], bf16> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mul([[VARG0]], [[VARG1]]) : !mo.tensor<[7], ui8>, !mo.tensor<[7], bf16>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[7], bf16>

// -----

mo.graph @binary_elemwise_not_ok_unranked(%arg0: !mo.tensor<?, si64>, %arg1: !mo.tensor<?, si64>) -> !mo.tensor<?, si64> {
    // expected-error@below {{rmo.div ops do not support dynamic rank. Use the MO analogue ops instead.}}
    %1 = rmo.div(%arg0, %arg1) : !mo.tensor<?, si64>, !mo.tensor<?, si64>
    mo.output %1: !mo.tensor<?, si64>
}

// -----

mo.graph @binary_elemwise_not_ok_bad_cast_ui8_si8(%arg0: !mo.tensor<[7], si8>, %arg1: !mo.tensor<[7], ui8>) -> !mo.tensor<[7], si8> {
    // expected-error@below {{Failed to resolve valid dtype: Unsafe cast from ui8 to si8. Insert an explicit cast op if this conversion is wanted.}}
    %1 = rmo.mod(%arg0, %arg1) : !mo.tensor<[7], si8>, !mo.tensor<[7], ui8>
    mo.output %1: !mo.tensor<[7], si8>
}

// -----

mo.graph @binary_elemwise_not_ok_bad_cast_f16_bf16(%arg0: !mo.tensor<[7], bf16>, %arg1: !mo.tensor<[7], f16>) -> !mo.tensor<[7], bf16> {
    // expected-error@below {{Failed to resolve valid dtype: Unsafe cast from f16 to bf16. Insert an explicit cast op if this conversion is wanted.}}
    %1 = rmo.min(%arg0, %arg1) : !mo.tensor<[7], bf16>, !mo.tensor<[7], f16>
    mo.output %1: !mo.tensor<[7], bf16>
}

// -----

mo.graph @binary_elemwise_not_ok_bad_cast_tf32_f32(%arg0: !mo.tensor<[7], tf32>, %arg1: !mo.tensor<[7], f32>) -> !mo.tensor<[7], f32> {
    // expected-error@below {{Failed to resolve valid dtype: Unsafe cast from tf32 to f32. Insert an explicit cast op if this conversion is wanted.}}
    %1 = rmo.max(%arg0, %arg1) : !mo.tensor<[7], tf32>, !mo.tensor<[7], f32>
    mo.output %1: !mo.tensor<[7], f32>
}

// -----

mo.graph @binary_elemwise_dyn_static_bad_shape(%arg0: !mo.tensor<[1, ?], si64>, %arg1: !mo.tensor<[?, 5], si64>) -> !mo.tensor<[?, 5], si64> {
    // expected-error@below {{Inputs shapes [1, ?] and [?, 5] are not compatible. Dims "?" and "5" are neither equivalent nor broadcastable. Insert an explicit rebind if these dims will be equivalent at runtime.}}
    %1 = rmo.mul(%arg0, %arg1) : !mo.tensor<[1, ?], si64>, !mo.tensor<[?, 5], si64>
    mo.output %1: !mo.tensor<[?, 5], si64>
}
