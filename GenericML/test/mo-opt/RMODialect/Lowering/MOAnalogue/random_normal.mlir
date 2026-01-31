// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @random_normal_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape -> Sh4: !mosh.ape>(%size: !mo.tensor<Sh0, si64>, %mean: !mo.tensor<Sh1, f32>, %variance: !mo.tensor<Sh2, f32>, %seed: !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32> {
  %0 = rmo.mo.random.normal<() -> Sh5: !mosh.ape>(%size, %mean, %variance, %seed) : (!mo.tensor<Sh0, si64>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f32>, !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh5, f32>
  mo.output<:!mosh.ape Sh5> %0 : !mo.tensor<Sh5, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @random_normal_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape -> Sh4: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, si64>, %[[ARG1:.*]]: !mo.tensor<Sh1, f32>, %[[ARG2:.*]]: !mo.tensor<Sh2, f32>, %[[ARG3:.*]]: !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D0:[A-Za-z0-9_]*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, si64>) -> !mo.tensor<[[[RMO_D0]]], si64>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<Sh2, f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape(%[[ARG3]]) {{.*}} : (!mo.tensor<Sh3, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V4:.*]] = mo.random.normal<() -> [[RMO_SH1:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V0]], %[[V1]], %[[V2]], %[[V3]]) : (!mo.tensor<[[[RMO_D0]]], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[RMO_SH1]], f32>
//  CHECK-NEXT:   %[[V5:.*]] = rmo.rebind_tensor_shape<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V4]]) {{.*}} : (!mo.tensor<[[RMO_SH1]], f32>) -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH5]]> %[[V5]] : !mo.tensor<[[SH5]], f32>

// -----

mo.graph @random_normal_dyn_out_param<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape -> D0, D1>(%size: !mo.tensor<Sh0, si64>, %mean: !mo.tensor<Sh1, f32>, %variance: !mo.tensor<Sh2, f32>, %seed: !mo.tensor<Sh3, si64>) -> !mo.tensor<[D0, D1], f32> {
  %0 = rmo.mo.random.normal<() -> D2, D3>(%size, %mean, %variance, %seed) : (!mo.tensor<Sh0, si64>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f32>, !mo.tensor<Sh3, si64>) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %0 : !mo.tensor<[D2, D3], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @random_normal_dyn_out_param<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape -> D0, D1>(%[[ARG0:.*]]: !mo.tensor<Sh0, si64>, %[[ARG1:.*]]: !mo.tensor<Sh1, f32>, %[[ARG2:.*]]: !mo.tensor<Sh2, f32>, %[[ARG3:.*]]: !mo.tensor<Sh3, si64>) -> !mo.tensor<[D0, D1], f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<Sh2, f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape(%[[ARG3]]) {{.*}} : (!mo.tensor<Sh3, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V4:.*]] = mo.random.normal<() -> [[RMO_D0:[A-Za-z0-9_]*]], [[RMO_D1:[A-Za-z0-9_]*]]>(%[[V0]], %[[V1]], %[[V2]], %[[V3]]) : (!mo.tensor<[2], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>
//  CHECK-NEXT:   %[[V5:.*]] = rmo.rebind_tensor_shape<() -> [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]]>(%[[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>) -> !mo.tensor<[[[D2]], [[D3]]], f32>
//  CHECK-NEXT:   mo.output<[[D2]], [[D3]]> %[[V5]] : !mo.tensor<[[[D2]], [[D3]]], f32>

// -----

mo.graph @random_normal_params<D0 -> Sh0: !mosh.ape>(%size: !mo.tensor<[D0], si64>, %mean: !mo.tensor<[], f32>, %variance: !mo.tensor<[], f32>, %seed: !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.random.normal<() -> Sh1: !mosh.ape>(%size, %mean, %variance, %seed) : (!mo.tensor<[D0], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @random_normal_params<D0 -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0], si64>, %[[ARG1:.*]]: !mo.tensor<[], f32>, %[[ARG2:.*]]: !mo.tensor<[], f32>, %[[ARG3:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D0:[A-Za-z0-9_]*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<[D0], si64>) -> !mo.tensor<[[[RMO_D0]]], si64>
//  CHECK-NEXT:   %[[V1:.*]] = mo.random.normal<() -> [[RMO_SH1:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]) : (!mo.tensor<[[[RMO_D0]]], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[RMO_SH1]], f32>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<[[RMO_SH1]], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V2]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @random_normal_static<() -> Sh0: !mosh.ape>(%size: !mo.tensor<[2], si64>, %mean: !mo.tensor<[], f32>, %variance: !mo.tensor<[], f32>, %seed: !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.random.normal<() -> Sh1: !mosh.ape>(%size, %mean, %variance, %seed) : (!mo.tensor<[2], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @random_normal_static<() -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[2], si64>, %[[ARG1:.*]]: !mo.tensor<[], f32>, %[[ARG2:.*]]: !mo.tensor<[], f32>, %[[ARG3:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = mo.random.normal<() -> [[RMO_D0:[A-Za-z0-9_]*]], [[RMO_D1:[A-Za-z0-9_]*]]>(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]) : (!mo.tensor<[2], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V1]] : !mo.tensor<[[SH1]], f32>
