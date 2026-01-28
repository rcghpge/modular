// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @add_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.mo.add<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @add_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, f32>) -> !mo.tensor<Sh0, f32>
// CHECK-NEXT: %[[V1:.*]] = mo.add(%[[ARG0]], %[[V0]]) : !mo.tensor<Sh0, f32>
// CHECK-NEXT: %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH3]]> %[[V2]] : !mo.tensor<[[SH3]], f32>

// -----

mo.graph @add_dyn_params<Sh0: !mosh.ape, D0, D1, D2 -> Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<Sh1, f32> {
  %1 = rmo.mo.add<() -> Sh2: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<Sh2, f32>
  mo.output<:!mosh.ape Sh2> %1 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @add_dyn_params<Sh0: !mosh.ape, D0, D1, D2 -> Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<Sh1, f32> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[D0, D1, D2], f32>
// CHECK-NEXT: %[[V1:.*]] = mo.add(%[[V0]], %[[ARG1]]) : !mo.tensor<[D0, D1, D2], f32>
// CHECK-NEXT: %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> %[[V2]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @add_params_params<D0, D1, D2, D3, D4, D5 -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[D0, D1, D2], f32>, %arg1: !mo.tensor<[D3, D4, D5], f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.add<() -> Sh1: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[D3, D4, D5], f32>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @add_params_params<D0, D1, D2, D3, D4, D5 -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, D1, D2], f32>, %[[ARG1:.*]]: !mo.tensor<[D3, D4, D5], f32>) -> !mo.tensor<Sh0, f32> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<[D3, D4, D5], f32>) -> !mo.tensor<[D0, D1, D2], f32>
// CHECK-NEXT: %[[V1:.*]] = mo.add(%[[ARG0]], %[[V0]]) : !mo.tensor<[D0, D1, D2], f32>
// CHECK-NEXT: %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> %[[V2]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @add_params_mix<D0, D1, D2, D3, D4 -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[D0, D1, D2], f32>, %arg1: !mo.tensor<[D3, D4, 2], f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.add<() -> Sh1: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[D3, D4, 2], f32>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @add_params_mix<D0, D1, D2, D3, D4 -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, D1, D2], f32>, %[[ARG1:.*]]: !mo.tensor<[D3, D4, 2], f32>) -> !mo.tensor<Sh0, f32> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG0]]) {{.*}} : (!mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[D0, D1, 2], f32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<[D3, D4, 2], f32>) -> !mo.tensor<[D0, D1, 2], f32>
// CHECK-NEXT: %[[V2:.*]] = mo.add(%[[V0]], %[[V1]]) : !mo.tensor<[D0, D1, 2], f32>
// CHECK-NEXT: %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V2]]) {{.*}} : (!mo.tensor<[D0, D1, 2], f32>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> %[[V3]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @add_static_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[1, 2, 3], f32>, %arg1: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.add<() -> Sh1: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @add_static_static<() -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[1, 2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<Sh0, f32> {
// CHECK-NEXT: %[[V0:.*]] = mo.add(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> %[[V1]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @div_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.mo.div<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @div_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, f32>) -> !mo.tensor<Sh0, f32>
// CHECK-NEXT: %[[V1:.*]] = mo.div(%[[ARG0]], %[[V0]]) : !mo.tensor<Sh0, f32>
// CHECK-NEXT: %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH3]]> %[[V2]] : !mo.tensor<[[SH3]], f32>

// -----

mo.graph @max_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.mo.max<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @max_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, f32>) -> !mo.tensor<Sh0, f32>
// CHECK-NEXT: %[[V1:.*]] = mo.max(%[[ARG0]], %[[V0]]) : !mo.tensor<Sh0, f32>
// CHECK-NEXT: %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH3]]> %[[V2]] : !mo.tensor<[[SH3]], f32>

// -----

mo.graph @min_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.mo.min<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @min_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, f32>) -> !mo.tensor<Sh0, f32>
// CHECK-NEXT: %[[V1:.*]] = mo.min(%[[ARG0]], %[[V0]]) : !mo.tensor<Sh0, f32>
// CHECK-NEXT: %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH3]]> %[[V2]] : !mo.tensor<[[SH3]], f32>

// -----

mo.graph @mul_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.mo.mul<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @mul_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, f32>) -> !mo.tensor<Sh0, f32>
// CHECK-NEXT: %[[V1:.*]] = mo.mul(%[[ARG0]], %[[V0]]) : !mo.tensor<Sh0, f32>
// CHECK-NEXT: %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH3]]> %[[V2]] : !mo.tensor<[[SH3]], f32>

// -----

mo.graph @pow_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.mo.pow<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pow_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, f32>) -> !mo.tensor<Sh0, f32>
// CHECK-NEXT: %[[V1:.*]] = mo.pow(%[[ARG0]] : !mo.tensor<Sh0, f32>, %[[V0]] : !mo.tensor<Sh0, f32>) : !mo.tensor<Sh0, f32>
// CHECK-NEXT: %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH3]]> %[[V2]] : !mo.tensor<[[SH3]], f32>

// -----

mo.graph @sub_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.mo.sub<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @sub_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, f32>) -> !mo.tensor<Sh0, f32>
// CHECK-NEXT: %[[V1:.*]] = mo.sub(%[[ARG0]], %[[V0]]) : !mo.tensor<Sh0, f32>
// CHECK-NEXT: %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH3]]> %[[V2]] : !mo.tensor<[[SH3]], f32>

// -----

mo.graph @sub_static_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[1, 2, 3], f32>, %arg1: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.sub<() -> Sh1: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @sub_static_static<() -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[1, 2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<Sh0, f32> {
// CHECK-NEXT: %[[V0:.*]] = mo.sub(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> %[[V1]] : !mo.tensor<[[SH1]], f32>
