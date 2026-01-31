// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @transpose_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %perm: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.transpose<() -> Sh3: !mosh.ape>(%arg0, %perm) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @transpose_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH1:.*]]: !mosh.ape>(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[RMO_SH1]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D2:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, si64>) -> !mo.tensor<[[[RMO_D2]]], si64>
//  CHECK-NEXT:   %[[V2:.*]] = mo.transpose<() -> [[RMO_SH3:.*]]: !mosh.ape>(%[[V0]], %[[V1]]) : (!mo.tensor<[[RMO_SH1]], f32>, !mo.tensor<[[[RMO_D2]]], si64>) -> !mo.tensor<[[RMO_SH3]], f32>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V2]]) {{.*}} : (!mo.tensor<[[RMO_SH3]], f32>) -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> %[[V3]] : !mo.tensor<[[SH3]], f32>

// -----

mo.graph @transpose_dyn_params<Sh0: !mosh.ape, D0 -> Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %perm: !mo.tensor<[D0], si64>) -> !mo.tensor<Sh1, f32> {
  %0 = rmo.mo.transpose<() -> Sh2: !mosh.ape>(%arg0, %perm) : (!mo.tensor<Sh0, f32>, !mo.tensor<[D0], si64>) -> !mo.tensor<Sh2, f32>
  mo.output<:!mosh.ape Sh2> %0 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @transpose_dyn_params<Sh0: !mosh.ape, D0 -> Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<[D0], si64>) -> !mo.tensor<Sh1, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH1:.*]]: !mosh.ape>(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[RMO_SH1]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D2:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<[D0], si64>) -> !mo.tensor<[[[RMO_D2]]], si64>
//  CHECK-NEXT:   %[[V2:.*]] = mo.transpose<() -> [[RMO_SH3:.*]]: !mosh.ape>(%[[V0]], %[[V1]]) : (!mo.tensor<[[RMO_SH1]], f32>, !mo.tensor<[[[RMO_D2]]], si64>) -> !mo.tensor<[[RMO_SH3]], f32>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:.*]]: !mosh.ape>(%[[V2]]) {{.*}} : (!mo.tensor<[[RMO_SH3]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> %[[V3]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @transpose_params_dyn<D0, D1, D2, Sh0: !mosh.ape -> Sh1: !mosh.ape>(%arg0: !mo.tensor<[D0, D1, D2], f32>, %perm: !mo.tensor<Sh0, si64>) -> !mo.tensor<Sh1, f32> {
  %0 = rmo.mo.transpose<() -> Sh2: !mosh.ape>(%arg0, %perm) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<Sh0, si64>) -> !mo.tensor<Sh2, f32>
  mo.output<:!mosh.ape Sh2> %0 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @transpose_params_dyn<D0, D1, D2, Sh0: !mosh.ape -> Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, D1, D2], f32>, %[[ARG1:.*]]: !mo.tensor<Sh0, si64>) -> !mo.tensor<Sh1, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]], [[RMO_D2:.*]], [[RMO_D3:.*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh0, si64>) -> !mo.tensor<[3], si64>
//  CHECK-NEXT:   %[[V2:.*]] = mo.transpose<() -> [[RMO_D4:.*]], [[RMO_D5:.*]], [[RMO_D6:.*]]>(%[[V0]], %[[V1]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[RMO_D4]], [[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:.*]]: !mosh.ape>(%[[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D4]], [[RMO_D5]], [[RMO_D6]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> %[[V3]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @transpose_mix<D0, Sh0: !mosh.ape -> Sh1: !mosh.ape>(%arg0: !mo.tensor<[2, D0, 3], f32>, %perm: !mo.tensor<Sh0, si64>) -> !mo.tensor<Sh1, f32> {
  %0 = rmo.mo.transpose<() -> Sh2: !mosh.ape>(%arg0, %perm) : (!mo.tensor<[2, D0, 3], f32>, !mo.tensor<Sh0, si64>) -> !mo.tensor<Sh2, f32>
  mo.output<:!mosh.ape Sh2> %0 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @transpose_mix<D0, Sh0: !mosh.ape -> Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[2, D0, 3], f32>, %[[ARG1:.*]]: !mo.tensor<Sh0, si64>) -> !mo.tensor<Sh1, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]], [[RMO_D2:.*]], [[RMO_D3:.*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<[2, D0, 3], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh0, si64>) -> !mo.tensor<[3], si64>
//  CHECK-NEXT:   %[[V2:.*]] = mo.transpose<() -> [[RMO_D4:.*]], [[RMO_D5:.*]], [[RMO_D6:.*]]>(%[[V0]], %[[V1]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[RMO_D4]], [[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:.*]]: !mosh.ape>(%[[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D4]], [[RMO_D5]], [[RMO_D6]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> %[[V3]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @transpose_dyn_out_params<Sh0: !mosh.ape, Sh1: !mosh.ape -> D0, D1>(%arg0: !mo.tensor<Sh0, f32>, %perm: !mo.tensor<Sh1, si64>) -> !mo.tensor<[D0, D1], f32> {
  %0 = rmo.mo.transpose<() -> D2, D3>(%arg0, %perm) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %0 : !mo.tensor<[D2, D3], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @transpose_dyn_out_params<Sh0: !mosh.ape, Sh1: !mosh.ape -> D0, D1>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, si64>) -> !mo.tensor<[D0, D1], f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]], [[RMO_D2:.*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   %[[V2:.*]] = mo.transpose<() -> [[RMO_D3:.*]], [[RMO_D4:.*]]>(%[[V0]], %[[V1]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[D2:.*]], [[D3:.*]]>(%[[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[[D2]], [[D3]]], f32>
//  CHECK-NEXT:   mo.output<[[D2]], [[D3]]> %[[V3]] : !mo.tensor<[[[D2]], [[D3]]], f32>

// -----

mo.graph @transpose_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[2, 4, 3], f32>, %perm: !mo.tensor<[3], si64>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.transpose<() -> Sh1: !mosh.ape>(%arg0, %perm) : (!mo.tensor<[2, 4, 3], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @transpose_static<() -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[2, 4, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[3], si64>) -> !mo.tensor<Sh0, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]], [[RMO_D2:.*]], [[RMO_D3:.*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<[2, 4, 3], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = mo.transpose<() -> [[RMO_D4:.*]], [[RMO_D5:.*]], [[RMO_D6:.*]]>(%[[V0]], %[[ARG1]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[RMO_D4]], [[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D4]], [[RMO_D5]], [[RMO_D6]]], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V2]] : !mo.tensor<[[SH1]], f32>
