// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @argnonzero_dyn<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, si64> {
  %0 = rmo.mo.arg_nonzero<() -> Sh2: !mosh.ape>(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh2, si64>
  mo.output<:!mosh.ape Sh2> %0 : !mo.tensor<Sh2, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @argnonzero_dyn<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, si64> {
//  CHECK-NEXT:   %[[V0:.*]] = mo.arg_nonzero<() -> [[RMO_D0:[A-Za-z0-9_]*]], [[RMO_D1:[A-Za-z0-9_]*]]>(%[[ARG0]]) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[[RMO_D0]], [[RMO_D1]]], si64>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D0]], [[RMO_D1]]], si64>) -> !mo.tensor<[[SH2]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> %[[V1]] : !mo.tensor<[[SH2]], si64>



// -----

mo.graph @argnonzero_dyn_out_param<Sh0: !mosh.ape -> D0>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<[D0, 3], si64> {
  %0 = rmo.mo.arg_nonzero<() -> D1>(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[D1, 3], si64>
  mo.output<D1> %0 : !mo.tensor<[D1, 3], si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @argnonzero_dyn_out_param<Sh0: !mosh.ape -> D0>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>) -> !mo.tensor<[D0, 3], si64> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D0:[A-Za-z0-9_]*]], [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[[RMO_D0]], [[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = mo.arg_nonzero<() -> [[RMO_D3:[A-Za-z0-9_]*]]>(%[[V0]]) : (!mo.tensor<[[[RMO_D0]], [[RMO_D1]], [[RMO_D2]]], f32>) -> !mo.tensor<[[[RMO_D3]], 3], si64>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[D1:[A-Za-z0-9_]*]]>(%[[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], 3], si64>) -> !mo.tensor<[[[D1]], 3], si64>
//  CHECK-NEXT:   mo.output<[[D1]]> %[[V2]] : !mo.tensor<[[[D1]], 3], si64>

// -----

mo.graph @argnonzero_param<D0, D1 -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<Sh0, si64> {
  %0 = rmo.mo.arg_nonzero<() -> Sh1: !mosh.ape>(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<Sh1, si64>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @argnonzero_param<D0, D1 -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<Sh0, si64> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D0:[A-Za-z0-9_]*]], [[RMO_D1:[A-Za-z0-9_]*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = mo.arg_nonzero<() -> [[RMO_D2:[A-Za-z0-9_]*]]>(%[[V0]]) : (!mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>) -> !mo.tensor<[[[RMO_D2]], 2], si64>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D2]], 2], si64>) -> !mo.tensor<[[SH1]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V2]] : !mo.tensor<[[SH1]], si64>

// -----

mo.graph @argnonzero_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[2, 7], f32>) -> !mo.tensor<Sh0, si64> {
  %0 = rmo.mo.arg_nonzero<() -> Sh1: !mosh.ape>(%arg0) : (!mo.tensor<[2, 7], f32>) -> !mo.tensor<Sh1, si64>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @argnonzero_static<() -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[2, 7], f32>) -> !mo.tensor<Sh0, si64> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D0:[A-Za-z0-9_]*]], [[RMO_D1:[A-Za-z0-9_]*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<[2, 7], f32>) -> !mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = mo.arg_nonzero<() -> [[RMO_D2:[A-Za-z0-9_]*]]>(%[[V0]]) : (!mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>) -> !mo.tensor<[[[RMO_D2]], 2], si64>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D2]], 2], si64>) -> !mo.tensor<[[SH1]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V2]] : !mo.tensor<[[SH1]], si64>
