// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope


mo.graph @range_folded_away() -> (!mo.tensor<[?], ui32>) {
  %start = mo.constant {value = #M.dense_array<1> : tensor<ui32>} : !mo.tensor<[], ui32>
  %limit = mo.constant {value = #M.dense_array<11> : tensor<ui32>} : !mo.tensor<[], ui32>
  %step = mo.constant {value = #M.dense_array<2> : tensor<ui32>} : !mo.tensor<[], ui32>
  %0 = rmo.mo.range(%start, %limit, %step) : (
    !mo.tensor<[], ui32>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>) -> !mo.tensor<[?], ui32>

  mo.output %0 : !mo.tensor<[?], ui32>
}
// CHECK-LABEL: mo.graph @range_folded_away
// CHECK-SAME: <() -> [[D2:[^ ]*]]>() -> !mo.tensor<[[[D2]]], ui32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1, 3, 5, 7, 9> : tensor<5xui32>} : !mo.tensor<[5], ui32>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D1:[^ ]*]]>([[V0]]) {{.*}} : (!mo.tensor<[5], ui32>) -> !mo.tensor<[[[D1]]], ui32>
// CHECK-NEXT: mo.output<[[D1]]> [[V1]] : !mo.tensor<[[[D1]]], ui32>

// -----

mo.graph @non_constant_range_input(%step: !mo.tensor<[], ui32>) -> (!mo.tensor<[?], ui32>) {
  %start = mo.constant {value = #M.dense_array<1> : tensor<ui32>} : !mo.tensor<[], ui32>
  %limit = mo.constant {value = #M.dense_array<11> : tensor<ui32>} : !mo.tensor<[], ui32>
  %0 = rmo.mo.range(%start, %limit, %step) : (
    !mo.tensor<[], ui32>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>) -> !mo.tensor<[?], ui32>

  mo.output %0 : !mo.tensor<[?], ui32>
}
// CHECK-LABEL: mo.graph @non_constant_range_input
// CHECK-SAME: <() -> [[D2:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[], ui32>) -> !mo.tensor<[[[D2]]], ui32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<ui32>} : !mo.tensor<[], ui32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant {value = #M.dense_array<11> : tensor<ui32>} : !mo.tensor<[], ui32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.range<() -> [[RMO_D3:[^ ]*]]>([[V0]], [[V1]], [[VARG0]]) : (!mo.tensor<[], ui32>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>) -> !mo.tensor<[[[RMO_D3]]], ui32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D1:[^ ]*]]>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D3]]], ui32>) -> !mo.tensor<[[[D1]]], ui32>
// CHECK-NEXT: mo.output<[[D1]]> [[V3]] : !mo.tensor<[[[D1]]], ui32>

// -----


mo.graph @range_floats() -> (!mo.tensor<[5], f32>) {
  %start = mo.constant {value = #M.dense_array<1.0> : tensor<f32>} : !mo.tensor<[], f32>
  %limit = mo.constant {value = #M.dense_array<11.0> : tensor<f32>} : !mo.tensor<[], f32>
  %step = mo.constant {value = #M.dense_array<2.0> : tensor<f32>} : !mo.tensor<[], f32>
  %0 = rmo.mo.range(%start, %limit, %step) : (
    !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[5], f32>

  mo.output %0 : !mo.tensor<[5], f32>
}

// CHECK-LABEL: mo.graph @range_floats
// CHECK-SAME: () -> !mo.tensor<[5], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant {value = #M.dense_array<1.100000e+01> : tensor<f32>} : !mo.tensor<[], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.constant {value = #M.dense_array<2.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.range([[V0]], [[V1]], [[V2]]) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[5], f32>
// CHECK-NEXT: mo.output [[V3]] : !mo.tensor<[5], f32>

// -----

mo.graph @range_floats_unknown_output_shape() -> (!mo.tensor<[?], f32>) {
  %start = mo.constant {value = #M.dense_array<1.0> : tensor<f32>} : !mo.tensor<[], f32>
  %limit = mo.constant {value = #M.dense_array<11.0> : tensor<f32>} : !mo.tensor<[], f32>
  %step = mo.constant {value = #M.dense_array<2.0> : tensor<f32>} : !mo.tensor<[], f32>
  %0 = rmo.mo.range(%start, %limit, %step) : (
    !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[?], f32>

  mo.output %0 : !mo.tensor<[?], f32>
}
// CHECK-LABEL: mo.graph @range_floats_unknown_output_shape
// CHECK-SAME: <() -> [[D2:[^ ]*]]>() -> !mo.tensor<[[[D2]]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant {value = #M.dense_array<1.100000e+01> : tensor<f32>} : !mo.tensor<[], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.constant {value = #M.dense_array<2.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.range([[V0]], [[V1]], [[V2]]) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[5], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D1:[^ ]*]]>([[V3]]) {{.*}} : (!mo.tensor<[5], f32>) -> !mo.tensor<[[[D1]]], f32>
// CHECK-NEXT: mo.output<[[D1]]> [[V4]] : !mo.tensor<[[[D1]]], f32>
