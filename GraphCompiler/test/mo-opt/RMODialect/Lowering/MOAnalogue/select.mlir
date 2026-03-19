// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @select_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, bool>, %arg1: !mo.tensor<Sh1, f32>, %arg2: !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, f32> {
  %1 = rmo.mo.select<() -> Sh4: !mosh.ape>(%arg0, %arg1, %arg2) : (!mo.tensor<Sh0, bool>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %1 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @select_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, bool>, %[[ARG1:.*]]: !mo.tensor<Sh1, f32>, %[[ARG2:.*]]: !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, bool>) -> !mo.tensor<Sh1, bool>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<Sh2, f32>) -> !mo.tensor<Sh1, f32>
//  CHECK-NEXT:   %[[V2:.*]] = mo.select(%[[V0]], %[[ARG1]], %[[V1]]) : (!mo.tensor<Sh1, bool>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh1, f32>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH4:.*]]: !mosh.ape>(%[[V2]]) {{.*}} : (!mo.tensor<Sh1, f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> %[[V3]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @select_dyn_params<D0, D1, Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<[D0, D1], bool>, %arg1: !mo.tensor<Sh0, f32>, %arg2: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.mo.select<() -> Sh3: !mosh.ape>(%arg0, %arg1, %arg2) : (!mo.tensor<[D0, D1], bool>, !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @select_dyn_params<D0, D1, Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, D1], bool>, %[[ARG1:.*]]: !mo.tensor<Sh0, f32>, %[[ARG2:.*]]: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[D0, D1], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<Sh1, f32>) -> !mo.tensor<[D0, D1], f32>
//  CHECK-NEXT:   %[[V2:.*]] = mo.select(%[[ARG0]], %[[V0]], %[[V1]]) : (!mo.tensor<[D0, D1], bool>, !mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V2]]) {{.*}} : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> %[[V3]] : !mo.tensor<[[SH3]], f32>

// -----

mo.graph @select_params<D0, D1, D2, D3, D4, D5 -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[D0, D1], bool>, %arg1: !mo.tensor<[D2, D3], f32>, %arg2: !mo.tensor<[D4, D5], f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.select<() -> Sh1: !mosh.ape>(%arg0, %arg1, %arg2) : (!mo.tensor<[D0, D1], bool>, !mo.tensor<[D2, D3], f32>, !mo.tensor<[D4, D5], f32>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @select_params<D0, D1, D2, D3, D4, D5 -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, D1], bool>, %[[ARG1:.*]]: !mo.tensor<[D2, D3], f32>, %[[ARG2:.*]]: !mo.tensor<[D4, D5], f32>) -> !mo.tensor<Sh0, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG0]]) {{.*}} : (!mo.tensor<[D0, D1], bool>) -> !mo.tensor<[D2, D3], bool>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<[D4, D5], f32>) -> !mo.tensor<[D2, D3], f32>
//  CHECK-NEXT:   %[[V2:.*]] = mo.select(%[[V0]], %[[ARG1]], %[[V1]]) : (!mo.tensor<[D2, D3], bool>, !mo.tensor<[D2, D3], f32>, !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D2, D3], f32>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V2]]) {{.*}} : (!mo.tensor<[D2, D3], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V3]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @select_mix<D0, D1, D2, D3, D4 -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[2, D0, D1], bool>, %arg1: !mo.tensor<[2, D2, 5], f32>, %arg2: !mo.tensor<[D3, D4, 5], f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.select<() -> Sh1: !mosh.ape>(%arg0, %arg1, %arg2) : (!mo.tensor<[2, D0, D1], bool>, !mo.tensor<[2, D2, 5], f32>, !mo.tensor<[D3, D4, 5], f32>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @select_mix<D0, D1, D2, D3, D4 -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[2, D0, D1], bool>, %[[ARG1:.*]]: !mo.tensor<[2, D2, 5], f32>, %[[ARG2:.*]]: !mo.tensor<[D3, D4, 5], f32>) -> !mo.tensor<Sh0, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG0]]) {{.*}} : (!mo.tensor<[2, D0, D1], bool>) -> !mo.tensor<[2, D2, 5], bool>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<[D3, D4, 5], f32>) -> !mo.tensor<[2, D2, 5], f32>
//  CHECK-NEXT:   %[[V2:.*]] = mo.select(%[[V0]], %[[ARG1]], %[[V1]]) : (!mo.tensor<[2, D2, 5], bool>, !mo.tensor<[2, D2, 5], f32>, !mo.tensor<[2, D2, 5], f32>) -> !mo.tensor<[2, D2, 5], f32>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V2]]) {{.*}} : (!mo.tensor<[2, D2, 5], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V3]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @select_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[2, 8, 5], bool>, %arg1: !mo.tensor<[2, 8, 5], f32>, %arg2: !mo.tensor<[2, 8, 5], f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.select<() -> Sh1: !mosh.ape>(%arg0, %arg1, %arg2) : (!mo.tensor<[2, 8, 5], bool>, !mo.tensor<[2, 8, 5], f32>, !mo.tensor<[2, 8, 5], f32>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @select_static<() -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[2, 8, 5], bool>, %[[ARG1:.*]]: !mo.tensor<[2, 8, 5], f32>, %[[ARG2:.*]]: !mo.tensor<[2, 8, 5], f32>) -> !mo.tensor<Sh0, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = mo.select(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (!mo.tensor<[2, 8, 5], bool>, !mo.tensor<[2, 8, 5], f32>, !mo.tensor<[2, 8, 5], f32>) -> !mo.tensor<[2, 8, 5], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[2, 8, 5], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V1]] : !mo.tensor<[[SH1]], f32>
