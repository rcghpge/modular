// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators %s | FileCheck %s --enable-var-scope

mo.graph @cast_dyn(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<[?, ?, ?], si32> {
  %1 = rmo.cast(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<[?, ?, ?], si32>
  mo.output %1 : !mo.tensor<[?, ?, ?], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @cast_dyn<[[SH1:[^ ]*]]: !mosh.ape -> [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[[D2]], [[D3]], [[D4]]], si32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.cast([[V0]]) : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH5]], si32>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ9:[^ ]*]]: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [[SH5]], [[[D6:[^ ]*]], [[D7:[^ ]*]], [[D8:[^ ]*]]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH10:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ9]])>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D6]], [[D7]], [[D8]]>([[V1]]) : !mo.tensor<[[SH5]], si32> -> !mo.tensor<[[[D6]], [[D7]], [[D8]]], si32>
// CHECK-NEXT: mo.output<[[D6]], [[D7]], [[D8]]> [[V2]] : !mo.tensor<[[[D6]], [[D7]], [[D8]]], si32>

// -----

mo.graph @cast_params(%arg0: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?], si32> {
  %1 = rmo.cast(%arg0) : (!mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?], si32>
  mo.output %1 : !mo.tensor<[?, ?, ?], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @cast_params<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]] -> [[D4:[^ ]*]], [[D5:[^ ]*]], [[D6:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]]], si32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.cast([[V0]]) : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], si32>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ13:[^ ]*]]: i1 = <and(eq([[D10:[^ ]*]], [[D7]]), eq([[D11:[^ ]*]], [[D8]]), eq([[D12:[^ ]*]], [[D9]]))>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH14:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ13]])>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D10]], [[D11]], [[D12]]>([[V1]]) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], si32> -> !mo.tensor<[[[D10]], [[D11]], [[D12]]], si32>
// CHECK-NEXT: mo.output<[[D10]], [[D11]], [[D12]]> [[V2]] : !mo.tensor<[[[D10]], [[D11]], [[D12]]], si32>
// -----

mo.graph @cast_static(%arg0: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[?, ?, ?], si32> {
  %1 = rmo.cast(%arg0) : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[?, ?, ?], si32>
  mo.output %1 : !mo.tensor<[?, ?, ?], si32>
}

// CHECK-LABEL: mo.graph
// CHECK-SAME: @cast_static<() -> [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], si32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.cast([[VARG0]]) : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], si32>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ7:[^ ]*]]: i1 = <and(eq([[D4:[^ ]*]], 1), eq([[D5:[^ ]*]], 2), eq([[D6:[^ ]*]], 3))>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH8:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ7]])>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D4]], [[D5]], [[D6]]>([[V0]]) : !mo.tensor<[1, 2, 3], si32> -> !mo.tensor<[[[D4]], [[D5]], [[D6]]], si32>
// CHECK-NEXT: mo.output<[[D4]], [[D5]], [[D6]]> [[V1]] : !mo.tensor<[[[D4]], [[D5]], [[D6]]], si32>
