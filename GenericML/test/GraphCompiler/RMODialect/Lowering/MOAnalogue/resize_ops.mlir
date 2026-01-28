// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @resize_linear_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %size: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.resize.linear<() -> Sh3: !mosh.ape>(%arg0, %size) {coordinate_transform_mode = #mo<coord_trans_mode align_corners>, antialias = false}: (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @resize_linear_dyn<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[SH2]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
//  CHECK-NEXT:   [[V1:%.*]] = mo.resize.linear<() -> [[RMO_SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]], [[V0]]) {antialias = false, coordinate_transform_mode = #mo<coord_trans_mode align_corners>} : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[[RMO_D1]]], si64>) -> !mo.tensor<[[RMO_SH2]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[RMO_SH2]], f32>) -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> [[V2]] : !mo.tensor<[[SH3]], f32>

// -----

mo.graph @resize_linear_params<D0, D1, D2, D3, D4 -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>, %size: !mo.tensor<[D4], si64>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.resize.linear<() -> Sh1: !mosh.ape>(%arg0, %size) {coordinate_transform_mode = #mo<coord_trans_mode align_corners>, antialias = false}: (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[D4], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @resize_linear_params<[[D0:[A-Za-z0-9_]*]], [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]] -> [[SH0:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D4]]], si64>) -> !mo.tensor<[[SH0]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.resize.linear<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) {antialias = false, coordinate_transform_mode = #mo<coord_trans_mode align_corners>} : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]]], si64>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V1]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @resize_mix<D0, D1, Sh0: !mosh.ape -> Sh1: !mosh.ape>(%arg0: !mo.tensor<[D0, 8, D1, 9], f32>, %size: !mo.tensor<Sh0, si64>) -> !mo.tensor<Sh1, f32> {
  %0 = rmo.mo.resize.linear<() -> Sh2: !mosh.ape>(%arg0, %size) {coordinate_transform_mode = #mo<coord_trans_mode align_corners>, antialias = false}: (!mo.tensor<[D0, 8, D1, 9], f32>, !mo.tensor<Sh0, si64>) -> !mo.tensor<Sh2, f32>
  mo.output<:!mosh.ape Sh2> %0 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @resize_mix<[[D0:[A-Za-z0-9_]*]], [[D1:[A-Za-z0-9_]*]], [[SH0:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D0]], 8, [[D1]], 9], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH0]], si64>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH0]], si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
//  CHECK-NEXT:   [[V1:%.*]] = mo.resize.linear<() -> [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG0]], [[V0]]) {antialias = false, coordinate_transform_mode = #mo<coord_trans_mode align_corners>} : (!mo.tensor<[[[D0]], 8, [[D1]], 9], f32>, !mo.tensor<[[[RMO_D1]]], si64>) -> !mo.tensor<[[[RMO_D2]], [[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D2]], [[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V2]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @resize_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[2, 8, 13, 9], f32>, %size: !mo.tensor<[4], si64>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.resize.linear<() -> Sh1: !mosh.ape>(%arg0, %size) {coordinate_transform_mode = #mo<coord_trans_mode align_corners>, antialias = false}: (!mo.tensor<[2, 8, 13, 9], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @resize_static<() -> [[SH0:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, 8, 13, 9], f32>, [[ARG1:%.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<[[SH0]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.resize.linear<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) {antialias = false, coordinate_transform_mode = #mo<coord_trans_mode align_corners>} : (!mo.tensor<[2, 8, 13, 9], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V1]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @resize_nearest_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %size: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.resize.nearest<() -> Sh3: !mosh.ape>(%arg0, %size) {coordinate_transform_mode = #mo<coord_trans_mode half_pixel>, round_mode = 1}: (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @resize_nearest_dyn<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[SH2]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
//  CHECK-NEXT:   [[V1:%.*]] = mo.resize.nearest<() -> [[RMO_SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]], [[V0]]) {coordinate_transform_mode = #mo<coord_trans_mode half_pixel>, round_mode = 1 : i64} : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[[RMO_D1]]], si64>) -> !mo.tensor<[[RMO_SH2]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[RMO_SH2]], f32>) -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> [[V2]] : !mo.tensor<[[SH3]], f32>

// -----

mo.graph @resize_bicubic_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[1, 3, 224, 224], f32>, %size: !mo.tensor<[4], si64>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.resize.bicubic<() -> Sh1: !mosh.ape>(%arg0, %size) : (!mo.tensor<[1, 3, 224, 224], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @resize_bicubic_static<() -> [[SH0:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[1, 3, 224, 224], f32>, [[ARG1:%.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<[[SH0]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.resize.bicubic<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) : (!mo.tensor<[1, 3, 224, 224], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V1]] : !mo.tensor<[[SH1]], f32>
