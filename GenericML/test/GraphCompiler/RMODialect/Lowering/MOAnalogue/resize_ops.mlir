// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @resize_linear_dyn(%arg0: !mo.tensor<?, f32>, %size: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.resize.linear(%arg0, %size) {coordinate_transform_mode = #mo<coord_trans_mode align_corners>, antialias = false}: (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @resize_linear_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH5]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], si64>) -> !mo.tensor<[[[RMO_D7]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.resize.linear<() -> [[RMO_SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]], [[V2]]) {antialias = false, coordinate_transform_mode = #mo<coord_trans_mode align_corners>} : (!mo.tensor<[[SH4]], f32>, !mo.tensor<[[[RMO_D7]]], si64>) -> !mo.tensor<[[RMO_SH8]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[RMO_SH8]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @resize_linear_params(%arg0: !mo.tensor<[?, ?, ?, ?], f32>, %size: !mo.tensor<[?], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.resize.linear(%arg0, %size) {coordinate_transform_mode = #mo<coord_trans_mode align_corners>, antialias = false}: (!mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[?], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @resize_linear_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]] -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D5]]], si64>) -> !mo.tensor<[[SH6]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D7:[A-Za-z0-9_]*]], [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]], [[D10]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D11:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D5]]], si64> -> !mo.tensor<[[[D11]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.resize.linear<() -> [[RMO_D13:[A-Za-z0-9_]*]], [[RMO_D14:[A-Za-z0-9_]*]], [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]]>([[V0]], [[V1]]) {antialias = false, coordinate_transform_mode = #mo<coord_trans_mode align_corners>} : (!mo.tensor<[[[D7]], [[D8]], [[D9]], [[D10]]], f32>, !mo.tensor<[[[D11]]], si64>) -> !mo.tensor<[[[RMO_D13]], [[RMO_D14]], [[RMO_D15]], [[RMO_D16]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH12:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D13]], [[RMO_D14]], [[RMO_D15]], [[RMO_D16]]], f32>) -> !mo.tensor<[[SH12]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH12]]> [[V3]] : !mo.tensor<[[SH12]], f32>

// -----

mo.graph @resize_mix(%arg0: !mo.tensor<[?, 8, ?, 9], f32>, %size: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.resize.linear(%arg0, %size) {coordinate_transform_mode = #mo<coord_trans_mode align_corners>, antialias = false}: (!mo.tensor<[?, 8, ?, 9], f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @resize_mix<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[SH3:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], 8, [[D2]], 9], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D5:[A-Za-z0-9_]*]], [[D6:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], 8, [[D2]], 9], f32> -> !mo.tensor<[[[D5]], 8, [[D6]], 9], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH7]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D9:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH7]], si64>) -> !mo.tensor<[[[RMO_D9]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.resize.linear<() -> [[RMO_D10:[A-Za-z0-9_]*]], [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]], [[RMO_D13:[A-Za-z0-9_]*]]>([[V0]], [[V2]]) {antialias = false, coordinate_transform_mode = #mo<coord_trans_mode align_corners>} : (!mo.tensor<[[[D5]], 8, [[D6]], 9], f32>, !mo.tensor<[[[RMO_D9]]], si64>) -> !mo.tensor<[[[RMO_D10]], [[RMO_D11]], [[RMO_D12]], [[RMO_D13]]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[RMO_D10]], [[RMO_D11]], [[RMO_D12]], [[RMO_D13]]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V4]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @resize_static(%arg0: !mo.tensor<[2, 8, 13, 9], f32>, %size: !mo.tensor<[4], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.resize.linear(%arg0, %size) {coordinate_transform_mode = #mo<coord_trans_mode align_corners>, antialias = false}: (!mo.tensor<[2, 8, 13, 9], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @resize_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, 8, 13, 9], f32>, [[ARG1:%.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.resize.linear<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) {antialias = false, coordinate_transform_mode = #mo<coord_trans_mode align_corners>} : (!mo.tensor<[2, 8, 13, 9], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]], [[RMO_D6]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @resize_nearest_dyn(%arg0: !mo.tensor<?, f32>, %size: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.resize.nearest(%arg0, %size) {coordinate_transform_mode = #mo<coord_trans_mode half_pixel>, round_mode = 1}: (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @resize_nearest_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH5]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], si64>) -> !mo.tensor<[[[RMO_D7]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.resize.nearest<() -> [[RMO_SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]], [[V2]]) {coordinate_transform_mode = #mo<coord_trans_mode half_pixel>, round_mode = 1 : i64} : (!mo.tensor<[[SH4]], f32>, !mo.tensor<[[[RMO_D7]]], si64>) -> !mo.tensor<[[RMO_SH8]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[RMO_SH8]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], f32>
