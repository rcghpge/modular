// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators %s | FileCheck %s --enable-var-scope

mo.graph @rebind_tensor_shape_dyn_dyn<Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
    %0 = mo.rebind<() -> Sh3: !mosh.ape>(%arg0) : !mo.tensor<Sh1, f32> -> !mo.tensor<Sh3, f32>
    kgen.param.declare RMO_SH_EQ5: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", Sh3, Sh4)>
    kgen.param.declare RMO_SH_CH6: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "rebind shape check failed with error: \22RMO rebind shape check failed.\22" : !kgen.string}, RMO_SH_EQ5)>
    %1 = mo.rebind<() -> Sh4: !mosh.ape>(%0) : !mo.tensor<Sh3, f32> -> !mo.tensor<Sh4, f32>
    mo.output<:!mosh.ape Sh4> %1 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @rebind_tensor_shape_dyn_dyn<[[SH1:[^ ]*]]: !mosh.ape -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH2]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ5:[^ ]*]]: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [[SH3]], [[SH4:[^ ]*]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH6:[^ ]*]]: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "{{.*}}" : !kgen.string}, [[RMO_SH_EQ5]])>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH4]]: !mosh.ape>([[V0]]) : !mo.tensor<[[SH3]], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V1]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @rebind_tensor_shape_dyn_mix<Sh1: !mosh.ape -> D2>(%arg0: !mo.tensor<Sh1, f32>) -> !mo.tensor<[D2, 3], f32> {
    %0 = mo.rebind<() -> Sh3: !mosh.ape>(%arg0) : !mo.tensor<Sh1, f32> -> !mo.tensor<Sh3, f32>
    kgen.param.declare RMO_SH_EQ5: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", Sh3, [D4, 3])>
    kgen.param.declare RMO_SH_CH6: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "rebind shape check failed with error: \22RMO rebind shape check failed.\22" : !kgen.string}, RMO_SH_EQ5)>
    %1 = mo.rebind<() -> D4>(%0) : !mo.tensor<Sh3, f32> -> !mo.tensor<[D4, 3], f32>
    mo.output<D4> %1 : !mo.tensor<[D4, 3], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @rebind_tensor_shape_dyn_mix<[[SH1:[^ ]*]]: !mosh.ape -> [[D2:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[[D2]], 3], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ5:[^ ]*]]: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [[SH3]], [[[D4:[^ ]*]], 3])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH6:[^ ]*]]: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "{{.*}}" : !kgen.string}, [[RMO_SH_EQ5]])>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D4]]>([[V0]]) : !mo.tensor<[[SH3]], f32> -> !mo.tensor<[[[D4]], 3], f32>
// CHECK-NEXT: mo.output<[[D4]]> [[V1]] : !mo.tensor<[[[D4]], 3], f32>

// -----

mo.graph @rebind_tensor_shape_mix_dyn<D1 -> Sh2: !mosh.ape>(%arg0: !mo.tensor<[D1, 3], f32>) -> !mo.tensor<Sh2, f32> {
    %0 = mo.rebind<() -> D3>(%arg0) : !mo.tensor<[D1, 3], f32> -> !mo.tensor<[D3, 3], f32>
    kgen.param.declare RMO_SH_EQ5: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [D3, 3], Sh4)>
    kgen.param.declare RMO_SH_CH6: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "rebind shape check failed with error: \22RMO rebind shape check failed.\22" : !kgen.string}, RMO_SH_EQ5)>
    %1 = mo.rebind<() -> Sh4: !mosh.ape>(%0) : !mo.tensor<[D3, 3], f32> -> !mo.tensor<Sh4, f32>
    mo.output<:!mosh.ape Sh4> %1 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @rebind_tensor_shape_mix_dyn<[[D1:[^ ]*]] -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], 3], f32>) -> !mo.tensor<[[SH2]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], 3], f32> -> !mo.tensor<[[[D3]], 3], f32>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ5:[^ ]*]]: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [[[D3]], 3], [[SH4:[^ ]*]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH6:[^ ]*]]: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "{{.*}}" : !kgen.string}, [[RMO_SH_EQ5]])>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH4]]: !mosh.ape>([[V0]]) : !mo.tensor<[[[D3]], 3], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V1]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @rebind_tensor_shape_mix_mix<D1 -> D2>(%arg0: !mo.tensor<[4, D1], f32>) -> !mo.tensor<[D2, 3], f32> {
    %0 = mo.rebind<() -> D3>(%arg0) : !mo.tensor<[4, D1], f32> -> !mo.tensor<[4, D3], f32>
    kgen.param.declare RMO_SH_EQ5: i1 = <and(eq(:si64 D3, 3), eq(:si64 D4, 4))>
    kgen.param.declare RMO_SH_CH6: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "rebind shape check failed with error: \22RMO rebind shape check failed.\22" : !kgen.string}, RMO_SH_EQ5)>
    %1 = mo.rebind<() -> D4>(%0) : !mo.tensor<[4, D3], f32> -> !mo.tensor<[D4, 3], f32>
    mo.output<D4> %1 : !mo.tensor<[D4, 3], f32>
}

// CHECK-LABEL: mo.graph
// CHECK-SAME: @rebind_tensor_shape_mix_mix<[[D1:[^ ]*]] -> [[D2:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[4, [[D1]]], f32>) -> !mo.tensor<[[[D2]], 3], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[4, [[D1]]], f32> -> !mo.tensor<[4, [[D3]]], f32>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ5:[^ ]*]]: i1 = <and(eq(:si64 [[D3]], 3), eq(:si64 [[D4:[^ ]*]], 4))>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH6:[^ ]*]]: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "{{.*}}" : !kgen.string}, [[RMO_SH_EQ5]])>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D4]]>([[V0]]) : !mo.tensor<[4, [[D3]]], f32> -> !mo.tensor<[[[D4]], 3], f32>
// CHECK-NEXT: mo.output<[[D4]]> [[V1]] : !mo.tensor<[[[D4]], 3], f32>
