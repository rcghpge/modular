// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @and_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, bool>, %arg1: !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh2, bool> {
  %1 = rmo.mo.and<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, bool>, !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh3, bool>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @and_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, bool>, %[[ARG1:.*]]: !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh2, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, bool>) -> !mo.tensor<Sh0, bool>
//  CHECK-NEXT:   %[[V1:.*]] = mo.and(%[[ARG0]], %[[V0]]) : !mo.tensor<Sh0, bool>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<Sh0, bool>) -> !mo.tensor<[[SH3]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> %[[V2]] : !mo.tensor<[[SH3]], bool>

// -----

mo.graph @and_dyn_params<Sh0: !mosh.ape, D0, D1 -> Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, bool>, %arg1: !mo.tensor<[D0, D1], bool>) -> !mo.tensor<Sh1, bool> {
  %1 = rmo.mo.and<() -> Sh2: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, bool>, !mo.tensor<[D0, D1], bool>) -> !mo.tensor<Sh2, bool>
  mo.output<:!mosh.ape Sh2> %1 : !mo.tensor<Sh2, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @and_dyn_params<Sh0: !mosh.ape, D0, D1 -> Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, bool>, %[[ARG1:.*]]: !mo.tensor<[D0, D1], bool>) -> !mo.tensor<Sh1, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, bool>) -> !mo.tensor<[D0, D1], bool>
//  CHECK-NEXT:   %[[V1:.*]] = mo.and(%[[V0]], %[[ARG1]]) : !mo.tensor<[D0, D1], bool>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<[D0, D1], bool>) -> !mo.tensor<[[SH2]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> %[[V2]] : !mo.tensor<[[SH2]], bool>

// -----

mo.graph @and_params_params<D0, D1, D2, D3 -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[D0, D1], bool>, %arg1: !mo.tensor<[D2, D3], bool>) -> !mo.tensor<Sh0, bool> {
  %1 = rmo.mo.and<() -> Sh1: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[D0, D1], bool>, !mo.tensor<[D2, D3], bool>) -> !mo.tensor<Sh1, bool>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @and_params_params<D0, D1, D2, D3 -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, D1], bool>, %[[ARG1:.*]]: !mo.tensor<[D2, D3], bool>) -> !mo.tensor<Sh0, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<[D2, D3], bool>) -> !mo.tensor<[D0, D1], bool>
//  CHECK-NEXT:   %[[V1:.*]] = mo.and(%[[ARG0]], %[[V0]]) : !mo.tensor<[D0, D1], bool>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<[D0, D1], bool>) -> !mo.tensor<[[SH1]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V2]] : !mo.tensor<[[SH1]], bool>

// -----

mo.graph @and_mix<D0, D1, D2, D3 -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[D0, 2, D1], bool>, %arg1: !mo.tensor<[D2, D3, 3], bool>) -> !mo.tensor<Sh0, bool> {
  %1 = rmo.mo.and<() -> Sh1: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[D0, 2, D1], bool>, !mo.tensor<[D2, D3, 3], bool>) -> !mo.tensor<Sh1, bool>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @and_mix<D0, D1, D2, D3 -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, 2, D1], bool>, %[[ARG1:.*]]: !mo.tensor<[D2, D3, 3], bool>) -> !mo.tensor<Sh0, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG0]]) {{.*}} : (!mo.tensor<[D0, 2, D1], bool>) -> !mo.tensor<[D0, 2, 3], bool>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<[D2, D3, 3], bool>) -> !mo.tensor<[D0, 2, 3], bool>
//  CHECK-NEXT:   %[[V2:.*]] = mo.and(%[[V0]], %[[V1]]) : !mo.tensor<[D0, 2, 3], bool>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V2]]) {{.*}} : (!mo.tensor<[D0, 2, 3], bool>) -> !mo.tensor<[[SH1]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V3]] : !mo.tensor<[[SH1]], bool>

// -----

mo.graph @and_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[3, 2], bool>, %arg1: !mo.tensor<[3, 2], bool>) -> !mo.tensor<Sh0, bool> {
  %1 = rmo.mo.and<() -> Sh1: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[3, 2], bool>, !mo.tensor<[3, 2], bool>) -> !mo.tensor<Sh1, bool>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @and_static<() -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[3, 2], bool>, %[[ARG1:.*]]: !mo.tensor<[3, 2], bool>) -> !mo.tensor<Sh0, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = mo.and(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[3, 2], bool>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[3, 2], bool>) -> !mo.tensor<[[SH1]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V1]] : !mo.tensor<[[SH1]], bool>

// -----

mo.graph @or_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, bool>, %arg1: !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh2, bool> {
  %1 = rmo.mo.or<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, bool>, !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh3, bool>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @or_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, bool>, %[[ARG1:.*]]: !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh2, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, bool>) -> !mo.tensor<Sh0, bool>
//  CHECK-NEXT:   %[[V1:.*]] = mo.or(%[[ARG0]], %[[V0]]) : !mo.tensor<Sh0, bool>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<Sh0, bool>) -> !mo.tensor<[[SH3]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> %[[V2]] : !mo.tensor<[[SH3]], bool>

// -----

mo.graph @xor_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, bool>, %arg1: !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh2, bool> {
  %1 = rmo.mo.xor<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, bool>, !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh3, bool>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @xor_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, bool>, %[[ARG1:.*]]: !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh2, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, bool>) -> !mo.tensor<Sh0, bool>
//  CHECK-NEXT:   %[[V1:.*]] = mo.xor(%[[ARG0]], %[[V0]]) : !mo.tensor<Sh0, bool>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<Sh0, bool>) -> !mo.tensor<[[SH3]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> %[[V2]] : !mo.tensor<[[SH3]], bool>
