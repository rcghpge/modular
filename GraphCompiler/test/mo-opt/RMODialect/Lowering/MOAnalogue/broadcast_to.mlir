// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s

mo.graph @broadcast_to_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> D0>(%data: !mo.tensor<Sh0, si64>, %shape: !mo.tensor<Sh1, si64>) -> !mo.tensor<[D0], si64> {
    %1 = rmo.mo.broadcast_to<() -> D1>(%data, %shape) : (!mo.tensor<Sh0, si64>, !mo.tensor<Sh1, si64>) -> !mo.tensor<[D1], si64>
    mo.output<D1> %1: !mo.tensor<[D1], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> D0>(%[[ARG0:.*]]: !mo.tensor<Sh0, si64>, %[[ARG1:.*]]: !mo.tensor<Sh1, si64>) -> !mo.tensor<[D0], si64> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
// CHECK-NEXT: %[[V1:.*]] = mo.broadcast_to<() -> [[RMO_SH2:.*]]: !mosh.ape>(%[[ARG0]], %[[V0]]) : (!mo.tensor<Sh0, si64>, !mo.tensor<[[[RMO_D1]]], si64>) -> !mo.tensor<[[RMO_SH2]], si64>
// CHECK-NEXT: %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[D1:.*]]>(%[[V1]]) {{.*}} : (!mo.tensor<[[RMO_SH2]], si64>) -> !mo.tensor<[[[D1]]], si64>
// CHECK-NEXT: mo.output<[[D1]]> %[[V2]] : !mo.tensor<[[[D1]]], si64>

// -----

mo.graph @broadcast_to_dyn_params<Sh0: !mosh.ape, D0 -> D1>(%data: !mo.tensor<Sh0, si64>, %shape: !mo.tensor<[D0], si64>) -> !mo.tensor<[D1], si64> {
    %1 = rmo.mo.broadcast_to<() -> D2>(%data, %shape) : (!mo.tensor<Sh0, si64>, !mo.tensor<[D0], si64>) -> !mo.tensor<[D2], si64>
    mo.output<D2> %1: !mo.tensor<[D2], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_dyn_params<Sh0: !mosh.ape, D0 -> D1>(%[[ARG0:.*]]: !mo.tensor<Sh0, si64>, %[[ARG1:.*]]: !mo.tensor<[D0], si64>) -> !mo.tensor<[D1], si64> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<[D0], si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
// CHECK-NEXT: %[[V1:.*]] = mo.broadcast_to<() -> [[RMO_SH2:.*]]: !mosh.ape>(%[[ARG0]], %[[V0]]) : (!mo.tensor<Sh0, si64>, !mo.tensor<[[[RMO_D1]]], si64>) -> !mo.tensor<[[RMO_SH2]], si64>
// CHECK-NEXT: %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[D2:.*]]>(%[[V1]]) {{.*}} : (!mo.tensor<[[RMO_SH2]], si64>) -> !mo.tensor<[[[D2]]], si64>
// CHECK-NEXT: mo.output<[[D2]]> %[[V2]] : !mo.tensor<[[[D2]]], si64>

// -----

mo.graph @broadcast_to_dyn_static<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%data: !mo.tensor<Sh0, si64>, %shape: !mo.tensor<[5], si64>) -> !mo.tensor<Sh1, si64> {
    %1 = rmo.mo.broadcast_to<() -> Sh2: !mosh.ape>(%data, %shape) : (!mo.tensor<Sh0, si64>, !mo.tensor<[5], si64>) -> !mo.tensor<Sh2, si64>
    mo.output<:!mosh.ape Sh2> %1: !mo.tensor<Sh2, si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_dyn_static<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, si64>, %[[ARG1:.*]]: !mo.tensor<[5], si64>) -> !mo.tensor<Sh1, si64> {
// CHECK-NEXT: %[[V0:.*]] = mo.broadcast_to<() -> [[RMO_D1:.*]], [[RMO_D2:.*]], [[RMO_D3:.*]], [[RMO_D4:.*]], [[RMO_D5:.*]]>(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<Sh0, si64>, !mo.tensor<[5], si64>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], si64>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], si64>) -> !mo.tensor<[[SH2]], si64>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> %[[V1]] : !mo.tensor<[[SH2]], si64>

// -----

mo.graph @broadcast_to_dyn_dyn_f32<Sh0: !mosh.ape, Sh1: !mosh.ape -> D0>(%data: !mo.tensor<Sh0, f32>, %shape: !mo.tensor<Sh1, si64>) -> !mo.tensor<[D0], f32> {
    %1 = rmo.mo.broadcast_to<() -> D1>(%data, %shape) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<[D1], f32>
    mo.output<D1> %1: !mo.tensor<[D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_dyn_dyn_f32<Sh0: !mosh.ape, Sh1: !mosh.ape -> D0>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, si64>) -> !mo.tensor<[D0], f32> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
// CHECK-NEXT: %[[V1:.*]] = mo.broadcast_to<() -> [[RMO_SH2:.*]]: !mosh.ape>(%[[ARG0]], %[[V0]]) : (!mo.tensor<Sh0, f32>, !mo.tensor<[[[RMO_D1]]], si64>) -> !mo.tensor<[[RMO_SH2]], f32>
// CHECK-NEXT: %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[D1:.*]]>(%[[V1]]) {{.*}} : (!mo.tensor<[[RMO_SH2]], f32>) -> !mo.tensor<[[[D1]]], f32>
// CHECK-NEXT: mo.output<[[D1]]> %[[V2]] : !mo.tensor<[[[D1]]], f32>

// -----

mo.graph @broadcast_to_dyn_params_f32<Sh0: !mosh.ape, D0 -> D1>(%data: !mo.tensor<Sh0, f32>, %shape: !mo.tensor<[D0], si64>) -> !mo.tensor<[D1], f32> {
    %1 = rmo.mo.broadcast_to<() -> D2>(%data, %shape) : (!mo.tensor<Sh0, f32>, !mo.tensor<[D0], si64>) -> !mo.tensor<[D2], f32>
    mo.output<D2> %1: !mo.tensor<[D2], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_dyn_params_f32<Sh0: !mosh.ape, D0 -> D1>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<[D0], si64>) -> !mo.tensor<[D1], f32> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<[D0], si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
// CHECK-NEXT: %[[V1:.*]] = mo.broadcast_to<() -> [[RMO_SH2:.*]]: !mosh.ape>(%[[ARG0]], %[[V0]]) : (!mo.tensor<Sh0, f32>, !mo.tensor<[[[RMO_D1]]], si64>) -> !mo.tensor<[[RMO_SH2]], f32>
// CHECK-NEXT: %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[D2:.*]]>(%[[V1]]) {{.*}} : (!mo.tensor<[[RMO_SH2]], f32>) -> !mo.tensor<[[[D2]]], f32>
// CHECK-NEXT: mo.output<[[D2]]> %[[V2]] : !mo.tensor<[[[D2]]], f32>

// -----

mo.graph @broadcast_to_dyn_static_f32<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%data: !mo.tensor<Sh0, f32>, %shape: !mo.tensor<[5], si64>) -> !mo.tensor<Sh1, f32> {
    %1 = rmo.mo.broadcast_to<() -> Sh2: !mosh.ape>(%data, %shape) : (!mo.tensor<Sh0, f32>, !mo.tensor<[5], si64>) -> !mo.tensor<Sh2, f32>
    mo.output<:!mosh.ape Sh2> %1: !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_dyn_static_f32<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<[5], si64>) -> !mo.tensor<Sh1, f32> {
// CHECK-NEXT: %[[V0:.*]] = mo.broadcast_to<() -> [[RMO_D1:.*]], [[RMO_D2:.*]], [[RMO_D3:.*]], [[RMO_D4:.*]], [[RMO_D5:.*]]>(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<Sh0, f32>, !mo.tensor<[5], si64>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> %[[V1]] : !mo.tensor<[[SH2]], f32>
