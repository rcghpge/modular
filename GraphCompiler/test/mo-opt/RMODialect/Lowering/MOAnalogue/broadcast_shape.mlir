// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s

mo.graph @broadcast_shape_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> D0>(%shape0: !mo.tensor<Sh0, si64>, %shape1: !mo.tensor<Sh1, si64>) -> !mo.tensor<[D0], si64> {
    %1 = rmo.mo.broadcast_shape<() -> D1>(%shape0, %shape1) : (!mo.tensor<Sh0, si64>, !mo.tensor<Sh1, si64>) -> !mo.tensor<[D1], si64>
    mo.output<D1> %1: !mo.tensor<[D1], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_shape_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> D0>(%[[ARG0:.*]]: !mo.tensor<Sh0, si64>, %[[ARG1:.*]]: !mo.tensor<Sh1, si64>) -> !mo.tensor<[D0], si64> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D2:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, si64>) -> !mo.tensor<[[[RMO_D2]]], si64>
// CHECK-NEXT: %[[V2:.*]] = mo.broadcast_shape<() -> [[RMO_D3:.*]]>(%[[V0]], %[[V1]]) : (!mo.tensor<[[[RMO_D1]]], si64>, !mo.tensor<[[[RMO_D2]]], si64>) -> !mo.tensor<[[[RMO_D3]]], si64>
// CHECK-NEXT: %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[D1:.*]]>(%[[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D3]]], si64>) -> !mo.tensor<[[[D1]]], si64>
// CHECK-NEXT: mo.output<[[D1]]> %[[V3]] : !mo.tensor<[[[D1]]], si64>

// -----

mo.graph @broadcast_shape_dyn_params<Sh0: !mosh.ape, D0 -> D1>(%shape0: !mo.tensor<Sh0, si64>, %shape1: !mo.tensor<[D0], si64>) -> !mo.tensor<[D1], si64> {
    %1 = rmo.mo.broadcast_shape<() -> D2>(%shape0, %shape1) : (!mo.tensor<Sh0, si64>, !mo.tensor<[D0], si64>) -> !mo.tensor<[D2], si64>
    mo.output<D2> %1: !mo.tensor<[D2], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_shape_dyn_params<Sh0: !mosh.ape, D0 -> D1>(%[[ARG0:.*]]: !mo.tensor<Sh0, si64>, %[[ARG1:.*]]: !mo.tensor<[D0], si64>) -> !mo.tensor<[D1], si64> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D2:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<[D0], si64>) -> !mo.tensor<[[[RMO_D2]]], si64>
// CHECK-NEXT: %[[V2:.*]] = mo.broadcast_shape<() -> [[RMO_D3:.*]]>(%[[V0]], %[[V1]]) : (!mo.tensor<[[[RMO_D1]]], si64>, !mo.tensor<[[[RMO_D2]]], si64>) -> !mo.tensor<[[[RMO_D3]]], si64>
// CHECK-NEXT: %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[D2:.*]]>(%[[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D3]]], si64>) -> !mo.tensor<[[[D2]]], si64>
// CHECK-NEXT: mo.output<[[D2]]> %[[V3]] : !mo.tensor<[[[D2]]], si64>

// -----

mo.graph @broadcast_shape_params_params<D0, D1 -> D2>(%shape0: !mo.tensor<[D0], si64>, %shape1: !mo.tensor<[D1], si64>) -> !mo.tensor<[D2], si64> {
    %1 = rmo.mo.broadcast_shape<() -> D3>(%shape0, %shape1) : (!mo.tensor<[D0], si64>, !mo.tensor<[D1], si64>) -> !mo.tensor<[D3], si64>
    mo.output<D3> %1: !mo.tensor<[D3], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_shape_params_params<D0, D1 -> D2>(%[[ARG0:.*]]: !mo.tensor<[D0], si64>, %[[ARG1:.*]]: !mo.tensor<[D1], si64>) -> !mo.tensor<[D2], si64> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<[D0], si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D2:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<[D1], si64>) -> !mo.tensor<[[[RMO_D2]]], si64>
// CHECK-NEXT: %[[V2:.*]] = mo.broadcast_shape<() -> [[RMO_D3:.*]]>(%[[V0]], %[[V1]]) : (!mo.tensor<[[[RMO_D1]]], si64>, !mo.tensor<[[[RMO_D2]]], si64>) -> !mo.tensor<[[[RMO_D3]]], si64>
// CHECK-NEXT: %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[D3:.*]]>(%[[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D3]]], si64>) -> !mo.tensor<[[[D3]]], si64>
// CHECK-NEXT: mo.output<[[D3]]> %[[V3]] : !mo.tensor<[[[D3]]], si64>

// -----

mo.graph @broadcast_shape_static_static<() -> D0>(%shape0: !mo.tensor<[3], si64>, %shape1: !mo.tensor<[2], si64>) -> !mo.tensor<[D0], si64> {
    %1 = rmo.mo.broadcast_shape<() -> D1>(%shape0, %shape1) : (!mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D1], si64>
    mo.output<D1> %1: !mo.tensor<[D1], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_shape_static_static<() -> D0>(%[[ARG0:.*]]: !mo.tensor<[3], si64>, %[[ARG1:.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[D0], si64> {
// CHECK-NEXT: %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<[3], si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D2:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D2]]], si64>
// CHECK-NEXT: %[[V2:.*]] = mo.broadcast_shape<() -> [[RMO_D3:.*]]>(%[[V0]], %[[V1]]) : (!mo.tensor<[[[RMO_D1]]], si64>, !mo.tensor<[[[RMO_D2]]], si64>) -> !mo.tensor<[[[RMO_D3]]], si64>
// CHECK-NEXT: %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[D1:.*]]>(%[[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D3]]], si64>) -> !mo.tensor<[[[D1]]], si64>
// CHECK-NEXT: mo.output<[[D1]]> %[[V3]] : !mo.tensor<[[[D1]]], si64>
