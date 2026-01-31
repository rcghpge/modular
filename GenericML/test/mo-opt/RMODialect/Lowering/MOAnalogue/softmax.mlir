// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @softmax_dyn<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f32> {
  %1 = rmo.mo.softmax<() -> Sh2: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh2, f32>
  mo.output<:!mosh.ape Sh2> %1 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @softmax_dyn<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = mo.softmax(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<Sh0, f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> %[[V1]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @softmax_params<D0, D1 -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.softmax<() -> Sh1: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @softmax_params<D0, D1 -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, D1], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = mo.softmax(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[D0, D1], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V1]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @softmax_params_mix<D0 -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[3, D0], f32>, %arg1: !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.softmax<() -> Sh1: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[3, D0], f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @softmax_params_mix<D0 -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[3, D0], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = mo.softmax(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[3, D0], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[3, D0], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[3, D0], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V1]] : !mo.tensor<[[SH1]], f32>


// -----

mo.graph @softmax_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[3, 7], f32>, %arg1: !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.softmax<() -> Sh1: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[3, 7], f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @softmax_static<() -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[3, 7], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = mo.softmax(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[3, 7], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 7], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[3, 7], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V1]] : !mo.tensor<[[SH1]], f32>
