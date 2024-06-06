// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @cast_dyn(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<[?, ?, ?], si32> {
  %1 = rmo.mo.cast(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<[?, ?, ?], si32>
  mo.output %1 : !mo.tensor<[?, ?, ?], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @cast_dyn<[[SH1:[^ ]*]]: !mosh.ape -> [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[[D2]], [[D3]], [[D4]]], si32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.cast([[V0]]) : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH5]], si32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D6:[^ ]*]], [[D7:[^ ]*]], [[D8:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], si32>) -> !mo.tensor<[[[D6]], [[D7]], [[D8]]], si32>
// CHECK-NEXT: mo.output<[[D6]], [[D7]], [[D8]]> [[V2]] : !mo.tensor<[[[D6]], [[D7]], [[D8]]], si32>

// -----

mo.graph @cast_params(%arg0: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?], si32> {
  %1 = rmo.mo.cast(%arg0) : (!mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?], si32>
  mo.output %1 : !mo.tensor<[?, ?, ?], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @cast_params<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]] -> [[D4:[^ ]*]], [[D5:[^ ]*]], [[D6:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]]], si32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.cast([[V0]]) : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], si32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D10:[^ ]*]], [[D11:[^ ]*]], [[D12:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], si32>) -> !mo.tensor<[[[D10]], [[D11]], [[D12]]], si32>
// CHECK-NEXT: mo.output<[[D10]], [[D11]], [[D12]]> [[V2]] : !mo.tensor<[[[D10]], [[D11]], [[D12]]], si32>
// -----

mo.graph @cast_static(%arg0: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[?, ?, ?], si32> {
  %1 = rmo.mo.cast(%arg0) : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[?, ?, ?], si32>
  mo.output %1 : !mo.tensor<[?, ?, ?], si32>
}

// CHECK-LABEL: mo.graph
// CHECK-SAME: @cast_static<() -> [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], si32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.cast([[VARG0]]) : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], si32>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D4:[^ ]*]], [[D5:[^ ]*]], [[D6:[^ ]*]]>([[V0]]) {{.*}} : (!mo.tensor<[1, 2, 3], si32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]]], si32>
// CHECK-NEXT: mo.output<[[D4]], [[D5]], [[D6]]> [[V1]] : !mo.tensor<[[[D4]], [[D5]], [[D6]]], si32>
