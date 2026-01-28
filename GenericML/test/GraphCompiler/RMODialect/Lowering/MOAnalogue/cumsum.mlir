// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @cumsum_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %axis: !mo.tensor<Sh1, si64>) -> (!mo.tensor<Sh2, f32>) {
  %0 = rmo.mo.cumsum<() -> Sh3: !mosh.ape>(%input, %axis) {reverse = 0 : index, exclusive = 0 : index} : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @cumsum_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V1:.*]] = mo.cumsum(%[[ARG0]], %[[V0]]) {exclusive = 0 : index, reverse = 0 : index} : (!mo.tensor<Sh0, f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> %[[V2]] : !mo.tensor<[[SH3]], f32>

// -----

mo.graph @cumsum_dyn_param_out<Sh0: !mosh.ape, Sh1: !mosh.ape -> D0, D1>(%input: !mo.tensor<Sh0, f32>, %axis: !mo.tensor<Sh1, si64>) -> (!mo.tensor<[D0, D1], f32>) {
  %0 = rmo.mo.cumsum<() -> D2, D3>(%input, %axis) {reverse = 0 : index, exclusive = 0 : index} : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %0 : !mo.tensor<[D2, D3], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @cumsum_dyn_param_out<Sh0: !mosh.ape, Sh1: !mosh.ape -> D0, D1>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, si64>) -> !mo.tensor<[D0, D1], f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D0:[A-Za-z0-9_]*]], [[RMO_D1:[A-Za-z0-9_]*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V2:.*]] = mo.cumsum(%[[V0]], %[[V1]]) {exclusive = 0 : index, reverse = 0 : index} : (!mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]]>(%[[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>) -> !mo.tensor<[[[D2]], [[D3]]], f32>
//  CHECK-NEXT:   mo.output<[[D2]], [[D3]]> %[[V3]] : !mo.tensor<[[[D2]], [[D3]]], f32>

// -----

mo.graph @cumsum_dyn_param<D0, D1 -> Sh0: !mosh.ape>(%input: !mo.tensor<[D0, D1], f32>, %axis: !mo.tensor<[], si64>) -> (!mo.tensor<Sh0, f32>) {
  %0 = rmo.mo.cumsum<() -> Sh1: !mosh.ape>(%input, %axis) {reverse = 0 : index, exclusive = 0 : index} : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @cumsum_dyn_param<D0, D1 -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, D1], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = mo.cumsum(%[[ARG0]], %[[ARG1]]) {exclusive = 0 : index, reverse = 0 : index} : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[D0, D1], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V1]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @cumsum_static<() -> Sh0: !mosh.ape>(%input: !mo.tensor<[3, 5], f32>, %axis: !mo.tensor<[], si64>) -> (!mo.tensor<Sh0, f32>) {
  %0 = rmo.mo.cumsum<() -> Sh1: !mosh.ape>(%input, %axis) {reverse = 0 : index, exclusive = 0 : index} : (!mo.tensor<[3, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @cumsum_static<() -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[3, 5], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = mo.cumsum(%[[ARG0]], %[[ARG1]]) {exclusive = 0 : index, reverse = 0 : index} : (!mo.tensor<[3, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 5], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[3, 5], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V1]] : !mo.tensor<[[SH1]], f32>
