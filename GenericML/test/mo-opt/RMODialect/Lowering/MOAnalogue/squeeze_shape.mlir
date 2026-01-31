// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s

mo.graph @squeeze_shape_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%shape: !mo.tensor<Sh0, si64>, %indices: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, si64> {
  %1 = rmo.mo.squeeze_shape<() -> Sh3: !mosh.ape>(%shape, %indices) : (!mo.tensor<Sh0, si64>, !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh3, si64>
  mo.output<:!mosh.ape Sh3> %1: !mo.tensor<Sh3, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @squeeze_shape_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, si64>, %[[ARG1:.*]]: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, si64> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D2:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, si64>) -> !mo.tensor<[[[RMO_D2]]], si64>
//  CHECK-NEXT:   %[[V2:.*]] = mo.squeeze_shape<() -> [[RMO_D3:.*]]>(%[[V0]], %[[V1]]) : (!mo.tensor<[[[RMO_D1]]], si64>, !mo.tensor<[[[RMO_D2]]], si64>) -> !mo.tensor<[[[RMO_D3]]], si64>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D3]]], si64>) -> !mo.tensor<[[SH3]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> %[[V3]] : !mo.tensor<[[SH3]], si64>

// -----

mo.graph @squeeze_shape_dyn_params<Sh0: !mosh.ape, D0 -> Sh1: !mosh.ape>(%shape: !mo.tensor<Sh0, si64>, %indices: !mo.tensor<[D0], si64>) -> !mo.tensor<Sh1, si64> {
  %1 = rmo.mo.squeeze_shape<() -> Sh2: !mosh.ape>(%shape, %indices) : (!mo.tensor<Sh0, si64>, !mo.tensor<[D0], si64>) -> !mo.tensor<Sh2, si64>
  mo.output<:!mosh.ape Sh2> %1: !mo.tensor<Sh2, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @squeeze_shape_dyn_params<Sh0: !mosh.ape, D0 -> Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, si64>, %[[ARG1:.*]]: !mo.tensor<[D0], si64>) -> !mo.tensor<Sh1, si64> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D2:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<[D0], si64>) -> !mo.tensor<[[[RMO_D2]]], si64>
//  CHECK-NEXT:   %[[V2:.*]] = mo.squeeze_shape<() -> [[RMO_D3:.*]]>(%[[V0]], %[[V1]]) : (!mo.tensor<[[[RMO_D1]]], si64>, !mo.tensor<[[[RMO_D2]]], si64>) -> !mo.tensor<[[[RMO_D3]]], si64>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:.*]]: !mosh.ape>(%[[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D3]]], si64>) -> !mo.tensor<[[SH2]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> %[[V3]] : !mo.tensor<[[SH2]], si64>

// -----

mo.graph @squeeze_shape_params_params<D0, D1 -> Sh0: !mosh.ape>(%shape: !mo.tensor<[D0], si64>, %indices: !mo.tensor<[D1], si64>) -> !mo.tensor<Sh0, si64> {
  %1 = rmo.mo.squeeze_shape<() -> Sh1: !mosh.ape>(%shape, %indices) : (!mo.tensor<[D0], si64>, !mo.tensor<[D1], si64>) -> !mo.tensor<Sh1, si64>
  mo.output<:!mosh.ape Sh1> %1: !mo.tensor<Sh1, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @squeeze_shape_params_params<D0, D1 -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0], si64>, %[[ARG1:.*]]: !mo.tensor<[D1], si64>) -> !mo.tensor<Sh0, si64> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D2:.*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<[D0], si64>) -> !mo.tensor<[[[RMO_D2]]], si64>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<[D1], si64>) -> !mo.tensor<[[[RMO_D3]]], si64>
//  CHECK-NEXT:   %[[V2:.*]] = mo.squeeze_shape<() -> [[RMO_D4:.*]]>(%[[V0]], %[[V1]]) : (!mo.tensor<[[[RMO_D2]]], si64>, !mo.tensor<[[[RMO_D3]]], si64>) -> !mo.tensor<[[[RMO_D4]]], si64>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D4]]], si64>) -> !mo.tensor<[[SH1]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V3]] : !mo.tensor<[[SH1]], si64>

// -----

mo.graph @squeeze_shape_static_params<D0 -> Sh0: !mosh.ape>(%shape: !mo.tensor<[2], si64>, %indices: !mo.tensor<[D0], si64>) -> !mo.tensor<Sh0, si64> {
  %1 = rmo.mo.squeeze_shape<() -> Sh1: !mosh.ape>(%shape, %indices) : (!mo.tensor<[2], si64>, !mo.tensor<[D0], si64>) -> !mo.tensor<Sh1, si64>
  mo.output<:!mosh.ape Sh1> %1: !mo.tensor<Sh1, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @squeeze_shape_static_params<D0 -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[2], si64>, %[[ARG1:.*]]: !mo.tensor<[D0], si64>) -> !mo.tensor<Sh0, si64> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D2:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<[D0], si64>) -> !mo.tensor<[[[RMO_D2]]], si64>
//  CHECK-NEXT:   %[[V2:.*]] = mo.squeeze_shape<() -> [[RMO_D3:.*]]>(%[[V0]], %[[V1]]) : (!mo.tensor<[[[RMO_D1]]], si64>, !mo.tensor<[[[RMO_D2]]], si64>) -> !mo.tensor<[[[RMO_D3]]], si64>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D3]]], si64>) -> !mo.tensor<[[SH1]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V3]] : !mo.tensor<[[SH1]], si64>

// -----

mo.graph @squeeze_shape_static_static<() -> Sh0: !mosh.ape>(%shape: !mo.tensor<[4], si64>, %indices: !mo.tensor<[2], si64>) -> !mo.tensor<Sh0, si64> {
  %1 = rmo.mo.squeeze_shape<() -> Sh1: !mosh.ape>(%shape, %indices) : (!mo.tensor<[4], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<Sh1, si64>
  mo.output<:!mosh.ape Sh1> %1: !mo.tensor<Sh1, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @squeeze_shape_static_static<() -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[4], si64>, %[[ARG1:.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<Sh0, si64> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<[4], si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D2:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D2]]], si64>
//  CHECK-NEXT:   %[[V2:.*]] = mo.squeeze_shape<() -> [[RMO_D3:.*]]>(%[[V0]], %[[V1]]) : (!mo.tensor<[[[RMO_D1]]], si64>, !mo.tensor<[[[RMO_D2]]], si64>) -> !mo.tensor<[[[RMO_D3]]], si64>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D3]]], si64>) -> !mo.tensor<[[SH1]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V3]] : !mo.tensor<[[SH1]], si64>
