// RUN: mo-opt %s --mo-legalize-rmo-operators="skip-rmo-rebinds" | FileCheck %s

// COM: note rmo.mo.reshape also has information on rank information potentially stored in the output rank of the original op

mo.graph @reshape_dyn_dyn_out_params<Sh0: !mosh.ape, Sh1: !mosh.ape -> D0>(%shape0: !mo.tensor<Sh0, si64>, %shape1: !mo.tensor<Sh1, si64>) -> !mo.tensor<[D0], si64> {
    %1 = rmo.mo.reshape<() -> D1>(%shape0, %shape1) : (!mo.tensor<Sh0, si64>, !mo.tensor<Sh1, si64>) -> !mo.tensor<[D1], si64>
    mo.output<D1> %1: !mo.tensor<[D1], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reshape_dyn_dyn_out_params<Sh0: !mosh.ape, Sh1: !mosh.ape -> D0>(%[[ARG0:.*]]: !mo.tensor<Sh0, si64>, %[[ARG1:.*]]: !mo.tensor<Sh1, si64>) -> !mo.tensor<[D0], si64> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, si64>) -> !mo.tensor<[1], si64>
// CHECK-NEXT: %[[V1:.*]] = mo.reshape<() -> [[RMO_D1:.*]]>(%[[ARG0]], %[[V0]]) : (!mo.tensor<Sh0, si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
// CHECK-NEXT: %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[D1:.*]]>(%[[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D1]]], si64>) -> !mo.tensor<[[[D1]]], si64>
// CHECK-NEXT: mo.output<[[D1]]> %[[V2]] : !mo.tensor<[[[D1]]], si64>

// -----

mo.graph @reshape_dyn_params_out_params<Sh0: !mosh.ape, D0 -> D1>(%shape0: !mo.tensor<Sh0, si64>, %shape1: !mo.tensor<[D0], si64>) -> !mo.tensor<[D1], si64> {
    %1 = rmo.mo.reshape<() -> D2>(%shape0, %shape1) : (!mo.tensor<Sh0, si64>, !mo.tensor<[D0], si64>) -> !mo.tensor<[D2], si64>
    mo.output<D2> %1: !mo.tensor<[D2], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reshape_dyn_params_out_params<Sh0: !mosh.ape, D0 -> D1>(%[[ARG0:.*]]: !mo.tensor<Sh0, si64>, %[[ARG1:.*]]: !mo.tensor<[D0], si64>) -> !mo.tensor<[D1], si64> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<[D0], si64>) -> !mo.tensor<[1], si64>
// CHECK-NEXT: %[[V1:.*]] = mo.reshape<() -> [[RMO_D1:.*]]>(%[[ARG0]], %[[V0]]) : (!mo.tensor<Sh0, si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
// CHECK-NEXT: %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[D2:.*]]>(%[[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D1]]], si64>) -> !mo.tensor<[[[D2]]], si64>
// CHECK-NEXT: mo.output<[[D2]]> %[[V2]] : !mo.tensor<[[[D2]]], si64>

// -----

mo.graph @reshape_dyn_params_out_dyn<Sh0: !mosh.ape, D0 -> Sh1: !mosh.ape>(%shape0: !mo.tensor<Sh0, si64>, %shape1: !mo.tensor<[D0], si64>) -> !mo.tensor<Sh1, si64> {
    %1 = rmo.mo.reshape<() -> Sh2: !mosh.ape>(%shape0, %shape1) : (!mo.tensor<Sh0, si64>, !mo.tensor<[D0], si64>) -> !mo.tensor<Sh2, si64>
    mo.output<:!mosh.ape Sh2> %1: !mo.tensor<Sh2, si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reshape_dyn_params_out_dyn<Sh0: !mosh.ape, D0 -> Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, si64>, %[[ARG1:.*]]: !mo.tensor<[D0], si64>) -> !mo.tensor<Sh1, si64> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<[D0], si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
// CHECK-NEXT: %[[V1:.*]] = mo.reshape<() -> [[RMO_SH2:.*]]: !mosh.ape>(%[[ARG0]], %[[V0]]) : (!mo.tensor<Sh0, si64>, !mo.tensor<[[[RMO_D1]]], si64>) -> !mo.tensor<[[RMO_SH2]], si64>
// CHECK-NEXT: %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<[[RMO_SH2]], si64>) -> !mo.tensor<[[SH2]], si64>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> %[[V2]] : !mo.tensor<[[SH2]], si64>

// -----

mo.graph @reshape_dyn_static<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%shape0: !mo.tensor<Sh0, si64>, %shape1: !mo.tensor<[5], si64>) -> !mo.tensor<Sh1, si64> {
    %1 = rmo.mo.reshape<() -> Sh2: !mosh.ape>(%shape0, %shape1) : (!mo.tensor<Sh0, si64>, !mo.tensor<[5], si64>) -> !mo.tensor<Sh2, si64>
    mo.output<:!mosh.ape Sh2> %1: !mo.tensor<Sh2, si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reshape_dyn_static<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, si64>, %[[ARG1:.*]]: !mo.tensor<[5], si64>) -> !mo.tensor<Sh1, si64> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<[5], si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
// CHECK-NEXT: %[[V1:.*]] = mo.reshape<() -> [[RMO_D2:.*]], [[RMO_D3:.*]], [[RMO_D4:.*]], [[RMO_D5:.*]], [[RMO_D6:.*]]>(%[[ARG0]], %[[V0]]) : (!mo.tensor<Sh0, si64>, !mo.tensor<[[[RMO_D1]]], si64>) -> !mo.tensor<[[[RMO_D2]], [[RMO_D3]], [[RMO_D4]], [[RMO_D5]], [[RMO_D6]]], si64>
// CHECK-NEXT: %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:.*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D2]], [[RMO_D3]], [[RMO_D4]], [[RMO_D5]], [[RMO_D6]]], si64>) -> !mo.tensor<[[SH2]], si64>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> %[[V2]] : !mo.tensor<[[SH2]], si64>
