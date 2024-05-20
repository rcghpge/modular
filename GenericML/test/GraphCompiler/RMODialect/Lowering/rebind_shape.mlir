// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators %s | FileCheck %s --enable-var-scope

mo.graph @rebind_tensor_shape_dyn_dyn(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.rebind_tensor_shape(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @rebind_tensor_shape_dyn_dyn<[[SH1:[^ ]*]]: !mosh.ape -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH2]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ5:[^ ]*]]: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [[SH3]], [[SH4:[^ ]*]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH6:[^ ]*]]: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "RMO rebind shape check failed." : !kgen.string}, [[RMO_SH_EQ5]])>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH4]]: !mosh.ape>([[V0]]) : !mo.tensor<[[SH3]], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V1]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @rebind_tensor_shape_dyn_mix(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<[?, 3], f32> {
  %1 = rmo.rebind_tensor_shape(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<[?, 3], f32>
  mo.output %1 : !mo.tensor<[?, 3], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @rebind_tensor_shape_dyn_mix<[[SH1:[^ ]*]]: !mosh.ape -> [[D2:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[[D2]], 3], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ5:[^ ]*]]: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [[SH3]], [[[D4:[^ ]*]], 3])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH6:[^ ]*]]: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "RMO rebind shape check failed." : !kgen.string}, [[RMO_SH_EQ5]])>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D4]]>([[V0]]) : !mo.tensor<[[SH3]], f32> -> !mo.tensor<[[[D4]], 3], f32>
// CHECK-NEXT: mo.output<[[D4]]> [[V1]] : !mo.tensor<[[[D4]], 3], f32>

// -----

mo.graph @rebind_tensor_shape_mix_dyn(%arg0: !mo.tensor<[?, 3], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.rebind_tensor_shape(%arg0) : (!mo.tensor<[?, 3], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @rebind_tensor_shape_mix_dyn<[[D1:[^ ]*]] -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], 3], f32>) -> !mo.tensor<[[SH2]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], 3], f32> -> !mo.tensor<[[[D3]], 3], f32>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ5:[^ ]*]]: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [[[D3]], 3], [[SH4:[^ ]*]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH6:[^ ]*]]: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "RMO rebind shape check failed." : !kgen.string}, [[RMO_SH_EQ5]])>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH4]]: !mosh.ape>([[V0]]) : !mo.tensor<[[[D3]], 3], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V1]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @rebind_tensor_shape_mix_mix(%arg0: !mo.tensor<[4, ?], f32>) -> !mo.tensor<[?, 3], f32> {
  %1 = rmo.rebind_tensor_shape(%arg0) : (!mo.tensor<[4, ?], f32>) -> !mo.tensor<[?, 3], f32>
  mo.output %1 : !mo.tensor<[?, 3], f32>
}

// CHECK-LABEL: mo.graph
// CHECK-SAME: @rebind_tensor_shape_mix_mix<[[D1:[^ ]*]] -> [[D2:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[4, [[D1]]], f32>) -> !mo.tensor<[[[D2]], 3], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[4, [[D1]]], f32> -> !mo.tensor<[4, [[D3]]], f32>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ5:[^ ]*]]: i1 = <and(eq([[D3]], 3), eq([[D4:[^ ]*]], 4))>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH6:[^ ]*]]: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "RMO rebind shape check failed." : !kgen.string}, [[RMO_SH_EQ5]])>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D4]]>([[V0]]) : !mo.tensor<[4, [[D3]]], f32> -> !mo.tensor<[[[D4]], 3], f32>
// CHECK-NEXT: mo.output<[[D4]]> [[V1]] : !mo.tensor<[[[D4]], 3], f32>
