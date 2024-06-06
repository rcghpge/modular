// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s

mo.graph @broadcast_shape_dyn_dyn(%shape0: !mo.tensor<?, si64>, %shape1: !mo.tensor<?, si64>) -> !mo.tensor<[?], si64> {
    %1 = rmo.mo.broadcast_shape(%shape0, %shape1) : (!mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_shape_dyn_dyn<[[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[[D3]]], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH5]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[^ ]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH4]], si64>) -> !mo.tensor<[[[RMO_D7]]], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D8:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], si64>) -> !mo.tensor<[[[RMO_D8]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.broadcast_shape<() -> [[RMO_D9:[^ ]*]]>([[V2]], [[V3]]) : (!mo.tensor<[[[RMO_D7]]], si64>, !mo.tensor<[[[RMO_D8]]], si64>) -> !mo.tensor<[[[RMO_D9]]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D6:[^ ]*]]>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D9]]], si64>) -> !mo.tensor<[[[D6]]], si64>
// CHECK-NEXT: mo.output<[[D6]]> [[V5]] : !mo.tensor<[[[D6]]], si64>

// -----

mo.graph @broadcast_shape_dyn_params(%shape0: !mo.tensor<?, si64>, %shape1: !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64> {
    %1 = rmo.mo.broadcast_shape(%shape0, %shape1) : (!mo.tensor<?, si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_shape_dyn_params<[[SH1:[^ ]*]]: !mosh.ape, [[D2:[^ ]*]] -> [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[[[D3]]], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]]], si64> -> !mo.tensor<[[[D5]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[^ ]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH4]], si64>) -> !mo.tensor<[[[RMO_D7]]], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D8:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[D5]]], si64>) -> !mo.tensor<[[[RMO_D8]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.broadcast_shape<() -> [[RMO_D9:[^ ]*]]>([[V2]], [[V3]]) : (!mo.tensor<[[[RMO_D7]]], si64>, !mo.tensor<[[[RMO_D8]]], si64>) -> !mo.tensor<[[[RMO_D9]]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D6:[^ ]*]]>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D9]]], si64>) -> !mo.tensor<[[[D6]]], si64>
// CHECK-NEXT: mo.output<[[D6]]> [[V5]] : !mo.tensor<[[[D6]]], si64>

// -----

mo.graph @broadcast_shape_params_params(%shape0: !mo.tensor<[?], si64>, %shape1: !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64> {
    %1 = rmo.mo.broadcast_shape(%shape0, %shape1) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_shape_params_params<[[D1:[^ ]*]], [[D2:[^ ]*]] -> [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[[[D3]]], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]]], si64> -> !mo.tensor<[[[D4]]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]]], si64> -> !mo.tensor<[[[D5]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[^ ]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[D4]]], si64>) -> !mo.tensor<[[[RMO_D7]]], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D8:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[D5]]], si64>) -> !mo.tensor<[[[RMO_D8]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.broadcast_shape<() -> [[RMO_D9:[^ ]*]]>([[V2]], [[V3]]) : (!mo.tensor<[[[RMO_D7]]], si64>, !mo.tensor<[[[RMO_D8]]], si64>) -> !mo.tensor<[[[RMO_D9]]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D6:[^ ]*]]>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D9]]], si64>) -> !mo.tensor<[[[D6]]], si64>
// CHECK-NEXT: mo.output<[[D6]]> [[V5]] : !mo.tensor<[[[D6]]], si64>

// -----

mo.graph @broadcast_shape_static_static(%shape0: !mo.tensor<[3], si64>, %shape1: !mo.tensor<[2], si64>) -> !mo.tensor<[?], si64> {
    %1 = rmo.mo.broadcast_shape(%shape0, %shape1) : (!mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_shape_static_static<() -> [[D1:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[3], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[[[D1]]], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[^ ]*]]>([[VARG0]]) {{.*}} : (!mo.tensor<[3], si64>) -> !mo.tensor<[[[RMO_D3]]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D4:[^ ]*]]>([[VARG1]]) {{.*}} : (!mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D4]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.broadcast_shape<() -> [[RMO_D5:[^ ]*]]>([[V0]], [[V1]]) : (!mo.tensor<[[[RMO_D3]]], si64>, !mo.tensor<[[[RMO_D4]]], si64>) -> !mo.tensor<[[[RMO_D5]]], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D2:[^ ]*]]>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D5]]], si64>) -> !mo.tensor<[[[D2]]], si64>
// CHECK-NEXT: mo.output<[[D2]]> [[V3]] : !mo.tensor<[[[D2]]], si64>
