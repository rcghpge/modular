// RUN: mo-opt %s --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" | FileCheck %s

// COM: note rmo.mo.reshape also has information on rank information potentially stored in the output rank of the original op

mo.graph @reshape_dyn_dyn_out_params(%shape0: !mo.tensor<?, si64>, %shape1: !mo.tensor<?, si64>) -> !mo.tensor<[?], si64> {
    %1 = rmo.mo.reshape(%shape0, %shape1) : (!mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reshape_dyn_dyn_out_params<[[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[[D3]]], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH5]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], si64>) -> !mo.tensor<[1], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.reshape<() -> [[RMO_D7:[^ ]*]]>([[V0]], [[V2]]) : (!mo.tensor<[[SH4]], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[[[RMO_D7]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D6:[^ ]*]]>([[V3]]) {{.*}} : (!mo.tensor<[[[RMO_D7]]], si64>) -> !mo.tensor<[[[D6]]], si64>
// CHECK-NEXT: mo.output<[[D6]]> [[V4]] : !mo.tensor<[[[D6]]], si64>

// -----

mo.graph @reshape_dyn_params_out_params(%shape0: !mo.tensor<?, si64>, %shape1: !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64> {
    %1 = rmo.mo.reshape(%shape0, %shape1) : (!mo.tensor<?, si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reshape_dyn_params_out_params<[[SH1:[^ ]*]]: !mosh.ape, [[D2:[^ ]*]] -> [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[[[D3]]], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]]], si64> -> !mo.tensor<[[[D5]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D5]]], si64>) -> !mo.tensor<[1], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.reshape<() -> [[RMO_D7:[^ ]*]]>([[V0]], [[V2]]) : (!mo.tensor<[[SH4]], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[[[RMO_D7]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D6:[^ ]*]]>([[V3]]) {{.*}} : (!mo.tensor<[[[RMO_D7]]], si64>) -> !mo.tensor<[[[D6]]], si64>
// CHECK-NEXT: mo.output<[[D6]]> [[V4]] : !mo.tensor<[[[D6]]], si64>

// -----

mo.graph @reshape_dyn_params_out_dyn(%shape0: !mo.tensor<?, si64>, %shape1: !mo.tensor<[?], si64>) -> !mo.tensor<?, si64> {
    %1 = rmo.mo.reshape(%shape0, %shape1) : (!mo.tensor<?, si64>, !mo.tensor<[?], si64>) -> !mo.tensor<?, si64>
    mo.output %1: !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reshape_dyn_params_out_dyn<[[SH1:[^ ]*]]: !mosh.ape, [[D2:[^ ]*]] -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[[SH3]], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]]], si64> -> !mo.tensor<[[[D5]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[D5]]], si64>) -> !mo.tensor<[[[RMO_D7]]], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.reshape<() -> [[RMO_SH8:[^ ]*]]: !mosh.ape>([[V0]], [[V2]]) : (!mo.tensor<[[SH4]], si64>, !mo.tensor<[[[RMO_D7]]], si64>) -> !mo.tensor<[[RMO_SH8]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[RMO_SH8]], si64>) -> !mo.tensor<[[SH6]], si64>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], si64>

// -----

mo.graph @reshape_dyn_static(%shape0: !mo.tensor<?, si64>, %shape1: !mo.tensor<[5], si64>) -> !mo.tensor<?, si64> {
    %1 = rmo.mo.reshape(%shape0, %shape1) : (!mo.tensor<?, si64>, !mo.tensor<[5], si64>) -> !mo.tensor<?, si64>
    mo.output %1: !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reshape_dyn_static<[[SH1:[^ ]*]]: !mosh.ape -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[5], si64>) -> !mo.tensor<[[SH2]], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH3]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D5:[^ ]*]]>([[VARG1]]) {{.*}} : (!mo.tensor<[5], si64>) -> !mo.tensor<[[[RMO_D5]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.reshape<() -> [[RMO_D6:[^ ]*]], [[RMO_D7:[^ ]*]], [[RMO_D8:[^ ]*]], [[RMO_D9:[^ ]*]], [[RMO_D10:[^ ]*]]>([[V0]], [[V1]]) : (!mo.tensor<[[SH3]], si64>, !mo.tensor<[[[RMO_D5]]], si64>) -> !mo.tensor<[[[RMO_D6]], [[RMO_D7]], [[RMO_D8]], [[RMO_D9]], [[RMO_D10]]], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH4:[^ ]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D6]], [[RMO_D7]], [[RMO_D8]], [[RMO_D9]], [[RMO_D10]]], si64>) -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V3]] : !mo.tensor<[[SH4]], si64>
