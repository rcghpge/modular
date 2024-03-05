// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators %s | FileCheck %s

mo.graph @broadcast_to_dyn_dyn(%shape0: !mo.tensor<?, si64>, %shape1: !mo.tensor<?, si64>) -> !mo.tensor<[?], si64> {
    %1 = rmo.broadcast_to(%shape0, %shape1) : (!mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_dyn_dyn<[[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[[D3]]], si64> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH5]], si64>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ9:[^ ]*]]: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [[SH5]], [[[RMO_D7:[^ ]*]]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH10:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ9]])>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[RMO_D7]]>([[V1]]) : !mo.tensor<[[SH5]], si64> -> !mo.tensor<[[[RMO_D7]]], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.broadcast_to<() -> [[RMO_SH8:[^ ]*]]: !mosh.ape>([[V0]], [[V2]]) : (!mo.tensor<[[SH4]], si64>, !mo.tensor<[[[RMO_D7]]], si64>) -> !mo.tensor<[[RMO_SH8]], si64>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ11:[^ ]*]]: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [[RMO_SH8]], [[[D6:[^ ]*]]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH12:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ11]])>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.rebind<() -> [[D6]]>([[V3]]) : !mo.tensor<[[RMO_SH8]], si64> -> !mo.tensor<[[[D6]]], si64>
// CHECK-NEXT: mo.output<[[D6]]> [[V4]] : !mo.tensor<[[[D6]]], si64>

// -----

mo.graph @broadcast_to_dyn_params(%shape0: !mo.tensor<?, si64>, %shape1: !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64> {
    %1 = rmo.broadcast_to(%shape0, %shape1) : (!mo.tensor<?, si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_dyn_params<[[SH1:[^ ]*]]: !mosh.ape, [[D2:[^ ]*]] -> [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[[[D3]]], si64> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]]], si64> -> !mo.tensor<[[[D5]]], si64>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ9:[^ ]*]]: i1 = <eq([[D5]], [[RMO_D7:[^ ]*]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH10:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ9]])>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[RMO_D7]]>([[V1]]) : !mo.tensor<[[[D5]]], si64> -> !mo.tensor<[[[RMO_D7]]], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.broadcast_to<() -> [[RMO_SH8:[^ ]*]]: !mosh.ape>([[V0]], [[V2]]) : (!mo.tensor<[[SH4]], si64>, !mo.tensor<[[[RMO_D7]]], si64>) -> !mo.tensor<[[RMO_SH8]], si64>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ11:[^ ]*]]: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [[RMO_SH8]], [[[D6:[^ ]*]]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH12:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ11]])>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.rebind<() -> [[D6]]>([[V3]]) : !mo.tensor<[[RMO_SH8]], si64> -> !mo.tensor<[[[D6]]], si64>
// CHECK-NEXT: mo.output<[[D6]]> [[V4]] : !mo.tensor<[[[D6]]], si64>

// -----

mo.graph @broadcast_to_dyn_static(%shape0: !mo.tensor<?, si64>, %shape1: !mo.tensor<[5], si64>) -> !mo.tensor<[?, ?, ?, ?, ?], si64> {
    %1 = rmo.broadcast_to(%shape0, %shape1) : (!mo.tensor<?, si64>, !mo.tensor<[5], si64>) -> !mo.tensor<[?, ?, ?, ?, ?], si64>
    mo.output %1: !mo.tensor<[?, ?, ?, ?, ?], si64>
}

// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_dyn_static<[[SH1:[^ ]*]]: !mosh.ape -> [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]], [[D6:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[5], si64>) -> !mo.tensor<[[[D2]], [[D3]], [[D4]], [[D5]], [[D6]]], si64> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH7:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH7]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.broadcast_to<() -> [[RMO_D13:[^ ]*]], [[RMO_D14:[^ ]*]], [[RMO_D15:[^ ]*]], [[RMO_D16:[^ ]*]], [[RMO_D17:[^ ]*]]>([[V0]], [[VARG1]]) : (!mo.tensor<[[SH7]], si64>, !mo.tensor<[5], si64>) -> !mo.tensor<[[[RMO_D13]], [[RMO_D14]], [[RMO_D15]], [[RMO_D16]], [[RMO_D17]]], si64>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ18:[^ ]*]]: i1 = <and(eq([[D10:[^ ]*]], [[RMO_D15]]), eq([[D11:[^ ]*]], [[RMO_D16]]), eq([[D12:[^ ]*]], [[RMO_D17]]), eq([[D8:[^ ]*]], [[RMO_D13]]), eq([[D9:[^ ]*]], [[RMO_D14]]))>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH19:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ18]])>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D8]], [[D9]], [[D10]], [[D11]], [[D12]]>([[V1]]) : !mo.tensor<[[[RMO_D13]], [[RMO_D14]], [[RMO_D15]], [[RMO_D16]], [[RMO_D17]]], si64> -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]], [[D12]]], si64>
// CHECK-NEXT: mo.output<[[D8]], [[D9]], [[D10]], [[D11]], [[D12]]> [[V2]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]], [[D12]]], si64>
