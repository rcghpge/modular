// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @equal_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, bool> {
  %1 = rmo.mo.equal<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh3, bool>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @equal_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, f32>) -> !mo.tensor<Sh0, f32>
//  CHECK-NEXT:   %[[V1:.*]] = mo.equal(%[[ARG0]], %[[V0]]) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, bool>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<Sh0, bool>) -> !mo.tensor<[[SH3]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> %[[V2]] : !mo.tensor<[[SH3]], bool>

// -----

mo.graph @equal_dyn_params<Sh0: !mosh.ape, D0, D1 -> Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<Sh1, bool> {
  %1 = rmo.mo.equal<() -> Sh2: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<Sh2, bool>
  mo.output<:!mosh.ape Sh2> %1 : !mo.tensor<Sh2, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @equal_dyn_params<Sh0: !mosh.ape, D0, D1 -> Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<Sh1, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[D0, D1], f32>
//  CHECK-NEXT:   %[[V1:.*]] = mo.equal(%[[V0]], %[[ARG1]]) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], bool>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<[D0, D1], bool>) -> !mo.tensor<[[SH2]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> %[[V2]] : !mo.tensor<[[SH2]], bool>

// -----

mo.graph @equal_params<D0, D1, D2, D3 -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D2, D3], f32>) -> !mo.tensor<Sh0, bool> {
  %1 = rmo.mo.equal<() -> Sh1: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>) -> !mo.tensor<Sh1, bool>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @equal_params<D0, D1, D2, D3 -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, D1], f32>, %[[ARG1:.*]]: !mo.tensor<[D2, D3], f32>) -> !mo.tensor<Sh0, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D0, D1], f32>
//  CHECK-NEXT:   %[[V1:.*]] = mo.equal(%[[ARG0]], %[[V0]]) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], bool>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<[D0, D1], bool>) -> !mo.tensor<[[SH1]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V2]] : !mo.tensor<[[SH1]], bool>

// -----

mo.graph @equal_mix<D0, D1, D2, D3 -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[D0, 2, D1], f32>, %arg1: !mo.tensor<[D2, D3, 3], f32>) -> !mo.tensor<Sh0, bool> {
  %1 = rmo.mo.equal<() -> Sh1: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[D0, 2, D1], f32>, !mo.tensor<[D2, D3, 3], f32>) -> !mo.tensor<Sh1, bool>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @equal_mix<D0, D1, D2, D3 -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, 2, D1], f32>, %[[ARG1:.*]]: !mo.tensor<[D2, D3, 3], f32>) -> !mo.tensor<Sh0, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG0]]) {{.*}} : (!mo.tensor<[D0, 2, D1], f32>) -> !mo.tensor<[D0, 2, 3], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<[D2, D3, 3], f32>) -> !mo.tensor<[D0, 2, 3], f32>
//  CHECK-NEXT:   %[[V2:.*]] = mo.equal(%[[V0]], %[[V1]]) : (!mo.tensor<[D0, 2, 3], f32>, !mo.tensor<[D0, 2, 3], f32>) -> !mo.tensor<[D0, 2, 3], bool>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V2]]) {{.*}} : (!mo.tensor<[D0, 2, 3], bool>) -> !mo.tensor<[[SH1]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V3]] : !mo.tensor<[[SH1]], bool>

// -----

mo.graph @equal_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[1, 2, 3], f32>, %arg1: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<Sh0, bool> {
  %1 = rmo.mo.equal<() -> Sh1: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<Sh1, bool>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @equal_static<() -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[1, 2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<Sh0, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = mo.equal(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], bool>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[1, 2, 3], bool>) -> !mo.tensor<[[SH1]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V1]] : !mo.tensor<[[SH1]], bool>

// -----

mo.graph @greater_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, bool> {
  %1 = rmo.mo.greater<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh3, bool>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @greater_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, f32>) -> !mo.tensor<Sh0, f32>
//  CHECK-NEXT:   %[[V1:.*]] = mo.greater(%[[ARG0]], %[[V0]]) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, bool>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<Sh0, bool>) -> !mo.tensor<[[SH3]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> %[[V2]] : !mo.tensor<[[SH3]], bool>

// -----

mo.graph @greater_equal_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, bool> {
  %1 = rmo.mo.greater_equal<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh3, bool>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @greater_equal_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, f32>) -> !mo.tensor<Sh0, f32>
//  CHECK-NEXT:   %[[V1:.*]] = mo.greater_equal(%[[ARG0]], %[[V0]]) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, bool>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<Sh0, bool>) -> !mo.tensor<[[SH3]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> %[[V2]] : !mo.tensor<[[SH3]], bool>

// -----

mo.graph @not_equal_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, bool> {
  %1 = rmo.mo.not_equal<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh3, bool>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @not_equal_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, bool> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, f32>) -> !mo.tensor<Sh0, f32>
//  CHECK-NEXT:   %[[V1:.*]] = mo.not_equal(%[[ARG0]], %[[V0]]) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh0, bool>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<Sh0, bool>) -> !mo.tensor<[[SH3]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> %[[V2]] : !mo.tensor<[[SH3]], bool>
