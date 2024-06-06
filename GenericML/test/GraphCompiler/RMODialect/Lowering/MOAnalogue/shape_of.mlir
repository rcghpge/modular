// RUN: mo-opt %s --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" | FileCheck %s

mo.graph @shape_of_dyn(%input: !mo.tensor<?, f32>) -> !mo.tensor<[?], si64> {
    %1 = rmo.mo.shape_of(%input): (!mo.tensor<?, f32>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @shape_of_dyn<[[SH1:[^ ]*]]: !mosh.ape -> [[D2:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[[D2]]], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.shape_of<() -> [[RMO_D5:[^ ]*]]>([[V0]]) : (!mo.tensor<[[SH3]], f32>) -> !mo.tensor<[[[RMO_D5]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D4:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D5]]], si64>) -> !mo.tensor<[[[D4]]], si64>
// CHECK-NEXT: mo.output<[[D4]]> [[V2]] : !mo.tensor<[[[D4]]], si64>

// -----

mo.graph @shape_of_params(%input: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?], si64> {
    %1 = rmo.mo.shape_of(%input): (!mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}

// CHECK-LABEL: mo.graph
// CHECK-SAME: @shape_of_params<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]] -> [[D4:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D4]]], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]], [[D6:[^ ]*]], [[D7:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D5]], [[D6]], [[D7]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.shape_of([[V0]]) : (!mo.tensor<[[[D5]], [[D6]], [[D7]]], f32>) -> !mo.tensor<[3], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D8:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[3], si64>) -> !mo.tensor<[[[D8]]], si64>
// CHECK-NEXT: mo.output<[[D8]]> [[V2]] : !mo.tensor<[[[D8]]], si64>
