// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s

mo.graph @broadcast_to_dyn_dyn(%data: !mo.tensor<?, si64>, %shape: !mo.tensor<?, si64>) -> !mo.tensor<[?], si64> {
    %1 = rmo.mo.broadcast_to(%data, %shape) : (!mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_dyn_dyn<[[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[[D3]]], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH5]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], si64>) -> !mo.tensor<[[[RMO_D7]]], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.broadcast_to<() -> [[RMO_SH8:[^ ]*]]: !mosh.ape>([[V0]], [[V2]]) : (!mo.tensor<[[SH4]], si64>, !mo.tensor<[[[RMO_D7]]], si64>) -> !mo.tensor<[[RMO_SH8]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D6:[^ ]*]]>([[V3]]) {{.*}} : (!mo.tensor<[[RMO_SH8]], si64>) -> !mo.tensor<[[[D6]]], si64>
// CHECK-NEXT: mo.output<[[D6]]> [[V4]] : !mo.tensor<[[[D6]]], si64>

// -----

mo.graph @broadcast_to_dyn_params(%data: !mo.tensor<?, si64>, %shape: !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64> {
    %1 = rmo.mo.broadcast_to(%data, %shape) : (!mo.tensor<?, si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_dyn_params<[[SH1:[^ ]*]]: !mosh.ape, [[D2:[^ ]*]] -> [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[[[D3]]], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]]], si64> -> !mo.tensor<[[[D5]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[D5]]], si64>) -> !mo.tensor<[[[RMO_D7]]], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.broadcast_to<() -> [[RMO_SH8:[^ ]*]]: !mosh.ape>([[V0]], [[V2]]) : (!mo.tensor<[[SH4]], si64>, !mo.tensor<[[[RMO_D7]]], si64>) -> !mo.tensor<[[RMO_SH8]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D6:[^ ]*]]>([[V3]]) {{.*}} : (!mo.tensor<[[RMO_SH8]], si64>) -> !mo.tensor<[[[D6]]], si64>
// CHECK-NEXT: mo.output<[[D6]]> [[V4]] : !mo.tensor<[[[D6]]], si64>

// -----

mo.graph @broadcast_to_dyn_static(%data: !mo.tensor<?, si64>, %shape: !mo.tensor<[5], si64>) -> !mo.tensor<?, si64> {
    %1 = rmo.mo.broadcast_to(%data, %shape) : (!mo.tensor<?, si64>, !mo.tensor<[5], si64>) -> !mo.tensor<?, si64>
    mo.output %1: !mo.tensor<?, si64>
}

// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_dyn_static<[[SH1:[^ ]*]]: !mosh.ape -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[5], si64>) -> !mo.tensor<[[SH2]], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH3]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.broadcast_to<() -> [[RMO_D5:[^ ]*]], [[RMO_D6:[^ ]*]], [[RMO_D7:[^ ]*]], [[RMO_D8:[^ ]*]], [[RMO_D9:[^ ]*]]>([[V0]], [[VARG1]]) : (!mo.tensor<[[SH3]], si64>, !mo.tensor<[5], si64>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]], [[RMO_D7]], [[RMO_D8]], [[RMO_D9]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH4:[^ ]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D5]], [[RMO_D6]], [[RMO_D7]], [[RMO_D8]], [[RMO_D9]]], si64>) -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V2]] : !mo.tensor<[[SH4]], si64>

// -----

mo.graph @broadcast_to_dyn_dyn_f32(%data: !mo.tensor<?, f32>, %shape: !mo.tensor<?, si64>) -> !mo.tensor<[?], f32> {
    %1 = rmo.mo.broadcast_to(%data, %shape) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<[?], f32>
    mo.output %1: !mo.tensor<[?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_dyn_dyn_f32<[[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[[D3]]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH5]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], si64>) -> !mo.tensor<[[[RMO_D7]]], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.broadcast_to<() -> [[RMO_SH8:[^ ]*]]: !mosh.ape>([[V0]], [[V2]]) : (!mo.tensor<[[SH4]], f32>, !mo.tensor<[[[RMO_D7]]], si64>) -> !mo.tensor<[[RMO_SH8]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D6:[^ ]*]]>([[V3]]) {{.*}} : (!mo.tensor<[[RMO_SH8]], f32>) -> !mo.tensor<[[[D6]]], f32>
// CHECK-NEXT: mo.output<[[D6]]> [[V4]] : !mo.tensor<[[[D6]]], f32>

// -----

mo.graph @broadcast_to_dyn_params_f32(%data: !mo.tensor<?, f32>, %shape: !mo.tensor<[?], si64>) -> !mo.tensor<[?], f32> {
    %1 = rmo.mo.broadcast_to(%data, %shape) : (!mo.tensor<?, f32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], f32>
    mo.output %1: !mo.tensor<[?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_dyn_params_f32<[[SH1:[^ ]*]]: !mosh.ape, [[D2:[^ ]*]] -> [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[[[D3]]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]]], si64> -> !mo.tensor<[[[D5]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[D5]]], si64>) -> !mo.tensor<[[[RMO_D7]]], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.broadcast_to<() -> [[RMO_SH8:[^ ]*]]: !mosh.ape>([[V0]], [[V2]]) : (!mo.tensor<[[SH4]], f32>, !mo.tensor<[[[RMO_D7]]], si64>) -> !mo.tensor<[[RMO_SH8]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D6:[^ ]*]]>([[V3]]) {{.*}} : (!mo.tensor<[[RMO_SH8]], f32>) -> !mo.tensor<[[[D6]]], f32>
// CHECK-NEXT: mo.output<[[D6]]> [[V4]] : !mo.tensor<[[[D6]]], f32>

// -----

mo.graph @broadcast_to_dyn_static_f32(%data: !mo.tensor<?, f32>, %shape: !mo.tensor<[5], si64>) -> !mo.tensor<?, f32> {
    %1 = rmo.mo.broadcast_to(%data, %shape) : (!mo.tensor<?, f32>, !mo.tensor<[5], si64>) -> !mo.tensor<?, f32>
    mo.output %1: !mo.tensor<?, f32>
}

// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_dyn_static_f32<[[SH1:[^ ]*]]: !mosh.ape -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[5], si64>) -> !mo.tensor<[[SH2]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.broadcast_to<() -> [[RMO_D5:[^ ]*]], [[RMO_D6:[^ ]*]], [[RMO_D7:[^ ]*]], [[RMO_D8:[^ ]*]], [[RMO_D9:[^ ]*]]>([[V0]], [[VARG1]]) : (!mo.tensor<[[SH3]], f32>, !mo.tensor<[5], si64>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]], [[RMO_D7]], [[RMO_D8]], [[RMO_D9]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH4:[^ ]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D5]], [[RMO_D6]], [[RMO_D7]], [[RMO_D8]], [[RMO_D9]]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V2]] : !mo.tensor<[[SH4]], f32>
