// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @is_nan_dyn<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, bool> {
  %1 = rmo.mo.is_nan<() -> Sh2: !mosh.ape>(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh2, bool>
  mo.output<:!mosh.ape Sh2> %1 : !mo.tensor<Sh2, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @is_nan_dyn<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = mo.is_nan(%[[ARG0]]) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, bool>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<Sh0, bool>) -> !mo.tensor<[[SH2]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> %[[V1]] : !mo.tensor<[[SH2]], bool>

// -----

mo.graph @is_nan_params<D0 -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[D0], f32>) -> !mo.tensor<Sh0, bool> {
  %1 = rmo.mo.is_nan<() -> Sh1: !mosh.ape>(%arg0) : (!mo.tensor<[D0], f32>) -> !mo.tensor<Sh1, bool>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @is_nan_params<D0 -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0], f32>) -> !mo.tensor<Sh0, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = mo.is_nan(%[[ARG0]]) : (!mo.tensor<[D0], f32>) -> !mo.tensor<[D0], bool>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[D0], bool>) -> !mo.tensor<[[SH1]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V1]] : !mo.tensor<[[SH1]], bool>

// -----

mo.graph @is_nan_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[2, 6], f32>) -> !mo.tensor<Sh0, bool> {
  %1 = rmo.mo.is_nan<() -> Sh1: !mosh.ape>(%arg0) : (!mo.tensor<[2, 6], f32>) -> !mo.tensor<Sh1, bool>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @is_nan_static<() -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[2, 6], f32>) -> !mo.tensor<Sh0, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = mo.is_nan(%[[ARG0]]) : (!mo.tensor<[2, 6], f32>) -> !mo.tensor<[2, 6], bool>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[2, 6], bool>) -> !mo.tensor<[[SH1]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V1]] : !mo.tensor<[[SH1]], bool>

// -----

mo.graph @is_inf_dyn<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, bool> {
  %1 = rmo.mo.is_inf<() -> Sh2: !mosh.ape>(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh2, bool>
  mo.output<:!mosh.ape Sh2> %1 : !mo.tensor<Sh2, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @is_inf_dyn<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = mo.is_inf(%[[ARG0]]) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, bool>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<Sh0, bool>) -> !mo.tensor<[[SH2]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> %[[V1]] : !mo.tensor<[[SH2]], bool>
