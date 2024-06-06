// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" --verify-diagnostics %s | FileCheck %s --enable-var-scope

mo.graph @bcast_like_dyn_dyn(%shape0: !mo.tensor<?, si64>, %shape1: !mo.tensor<?, si64>) -> !mo.tensor<[?], si64> {
    %1 = rmo.broadcast_like(%shape0, %shape1) : (!mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @bcast_like_dyn_dyn<[[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[[D3]]], si64> {
// CHECK-DAG: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH5]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.shape_of<() -> [[RMO_D12:[^ ]*]]>([[V1]]) : (!mo.tensor<[[SH5]], si64>) -> !mo.tensor<[[[RMO_D12]]], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[^ ]*]]>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D12]]], si64>) -> !mo.tensor<[[[RMO_D7]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D10:[^ ]*]]>([[V3]]) {{.*}} : (!mo.tensor<[[[RMO_D7]]], si64>) -> !mo.tensor<[[[RMO_D10]]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.broadcast_to<() -> [[RMO_SH11:[^ ]*]]: !mosh.ape>([[V0]], [[V4]]) : (!mo.tensor<[[SH4]], si64>, !mo.tensor<[[[RMO_D10]]], si64>) -> !mo.tensor<[[RMO_SH11]], si64>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH9:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[RMO_SH11]], si64>) -> !mo.tensor<[[RMO_SH9]], si64>
// CHECK-NEXT: [[V7:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D6:[^ ]*]]>([[V6]]) {{.*}} : (!mo.tensor<[[RMO_SH9]], si64>) -> !mo.tensor<[[[D6]]], si64>
// CHECK-NEXT: mo.output<[[D6]]> [[V7]] : !mo.tensor<[[[D6]]], si64>

// -----

mo.graph @bcast_like_dyn_params(%shape0: !mo.tensor<?, si64>, %shape1: !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64> {
    %1 = rmo.broadcast_like(%shape0, %shape1) : (!mo.tensor<?, si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @bcast_like_dyn_params<[[SH1:[^ ]*]]: !mosh.ape, [[D2:[^ ]*]] -> [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[[[D3]]], si64> {
// CHECK-DAG: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH4]], si64>
// CHECK-DAG: [[V1:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]]], si64> -> !mo.tensor<[[[D5]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.shape_of([[V1]]) : (!mo.tensor<[[[D5]]], si64>) -> !mo.tensor<[1], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.broadcast_to<() -> [[RMO_D8:[^ ]*]]>([[V0]], [[V2]]) : (!mo.tensor<[[SH4]], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[[[RMO_D8]]], si64>
// CHECK-DAG: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[^ ]*]]>([[V3]]) {{.*}} : (!mo.tensor<[[[RMO_D8]]], si64>) -> !mo.tensor<[[[RMO_D7]]], si64>
// CHECK-DAG: [[V5:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D6:[^ ]*]]>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D7]]], si64>) -> !mo.tensor<[[[D6]]], si64>
// CHECK-NEXT: mo.output<[[D6]]> [[V5]] : !mo.tensor<[[[D6]]], si64>

// -----

mo.graph @bcast_like_dyn_static(%shape0: !mo.tensor<?, si64>, %shape1: !mo.tensor<[5], si64>) -> !mo.tensor<[?], si64> {
    %1 = rmo.broadcast_like(%shape0, %shape1) : (!mo.tensor<?, si64>, !mo.tensor<[5], si64>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @bcast_like_dyn_static<[[SH1:[^ ]*]]: !mosh.ape -> [[D2:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[5], si64>) -> !mo.tensor<[[[D2]]], si64> {
// CHECK-DAG: [[V0:[^ ]*]] = mo.rebind<() -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH3]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.shape_of([[VARG1]]) : (!mo.tensor<[5], si64>) -> !mo.tensor<[1], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.broadcast_to<() -> [[RMO_D6:[^ ]*]]>([[V0]], [[V1]]) : (!mo.tensor<[[SH3]], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[[[RMO_D6]]], si64>
// CHECK-DAG: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D5:[^ ]*]]>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D6]]], si64>) -> !mo.tensor<[[[RMO_D5]]], si64>
// CHECK-DAG: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D4:[^ ]*]]>([[V3]]) {{.*}} : (!mo.tensor<[[[RMO_D5]]], si64>) -> !mo.tensor<[[[D4]]], si64>
// CHECK-NEXT: mo.output<[[D4]]> [[V4]] : !mo.tensor<[[[D4]]], si64>

// -----

mo.graph @bcast_like_real_world_condition(%shape0: !mo.tensor<[1, 10, 1, 10], si64>, %shape1: !mo.tensor<[50, 20, 10, 20, 10], si64>) -> !mo.tensor<?, si64> {
    %1 = rmo.broadcast_like(%shape0, %shape1) : (!mo.tensor<[1, 10, 1, 10], si64>, !mo.tensor<[50, 20, 10, 20, 10], si64>) -> !mo.tensor<?, si64>
    mo.output %1: !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @bcast_like_real_world_condition<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[1, 10, 1, 10], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[50, 20, 10, 20, 10], si64>) -> !mo.tensor<[[SH1]], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.shape_of([[VARG1]]) : (!mo.tensor<[50, 20, 10, 20, 10], si64>) -> !mo.tensor<[5], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.broadcast_to<() -> [[RMO_D8:[^ ]*]], [[RMO_D9:[^ ]*]], [[RMO_D10:[^ ]*]], [[RMO_D11:[^ ]*]], [[RMO_D12:[^ ]*]]>([[VARG0]], [[V0]]) : (!mo.tensor<[1, 10, 1, 10], si64>, !mo.tensor<[5], si64>) -> !mo.tensor<[[[RMO_D8]], [[RMO_D9]], [[RMO_D10]], [[RMO_D11]], [[RMO_D12]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[^ ]*]], [[RMO_D4:[^ ]*]], [[RMO_D5:[^ ]*]], [[RMO_D6:[^ ]*]], [[RMO_D7:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D8]], [[RMO_D9]], [[RMO_D10]], [[RMO_D11]], [[RMO_D12]]], si64>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]], [[RMO_D6]], [[RMO_D7]]], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH2:[^ ]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]], [[RMO_D6]], [[RMO_D7]]], si64>) -> !mo.tensor<[[SH2]], si64>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> [[V3]] : !mo.tensor<[[SH2]], si64>
