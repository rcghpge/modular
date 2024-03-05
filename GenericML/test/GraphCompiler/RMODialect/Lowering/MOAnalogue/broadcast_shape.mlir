// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators %s | FileCheck %s

mo.graph @shape_of_dyn_dyn(%shape0: !mo.tensor<?, si64>, %shape1: !mo.tensor<?, si64>) -> !mo.tensor<[?], si64> {
    %1 = rmo.broadcast_shape(%shape0, %shape1) : (!mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @shape_of_dyn_dyn<[[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[[D3]]], si64> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH5]], si64>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ10:[^ ]*]]: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [[SH4]], [RMO_[[D7:[^ ]*]]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH11:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ10]])>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> RMO_[[D7]]>([[V0]]) : !mo.tensor<[[SH4]], si64> -> !mo.tensor<[RMO_[[D7]]], si64>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ12:[^ ]*]]: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [[SH5]], [RMO_[[D8:[^ ]*]]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH13:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ12]])>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> RMO_[[D8]]>([[V1]]) : !mo.tensor<[[SH5]], si64> -> !mo.tensor<[RMO_[[D8]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.broadcast_shape<() -> RMO_[[D9:[^ ]*]]>([[V2]], [[V3]]) : (!mo.tensor<[RMO_[[D7]]], si64>, !mo.tensor<[RMO_[[D8]]], si64>) -> !mo.tensor<[RMO_[[D9]]], si64>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ14:[^ ]*]]: i1 = <eq([[D6:[^ ]*]], RMO_[[D9]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH15:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ14]])>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.rebind<() -> [[D6]]>([[V4]]) : !mo.tensor<[RMO_[[D9]]], si64> -> !mo.tensor<[[[D6]]], si64>
// CHECK-NEXT: mo.output<[[D6]]> [[V5]] : !mo.tensor<[[[D6]]], si64>

// -----

mo.graph @shape_of_dyn_static(%shape0: !mo.tensor<?, si64>, %shape1: !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64> {
    %1 = rmo.broadcast_shape(%shape0, %shape1) : (!mo.tensor<?, si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @shape_of_dyn_static<[[SH1:[^ ]*]]: !mosh.ape, [[D2:[^ ]*]] -> [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[[[D3]]], si64> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]]], si64> -> !mo.tensor<[[[D5]]], si64>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ10:[^ ]*]]: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [[SH4]], [RMO_[[D7:[^ ]*]]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH11:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ10]])>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> RMO_[[D7]]>([[V0]]) : !mo.tensor<[[SH4]], si64> -> !mo.tensor<[RMO_[[D7]]], si64>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ12:[^ ]*]]: i1 = <eq([[D5]], RMO_[[D8:[^ ]*]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH13:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ12]])>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> RMO_[[D8]]>([[V1]]) : !mo.tensor<[[[D5]]], si64> -> !mo.tensor<[RMO_[[D8]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.broadcast_shape<() -> RMO_[[D9:[^ ]*]]>([[V2]], [[V3]]) : (!mo.tensor<[RMO_[[D7]]], si64>, !mo.tensor<[RMO_[[D8]]], si64>) -> !mo.tensor<[RMO_[[D9]]], si64>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ14:[^ ]*]]: i1 = <eq([[D6:[^ ]*]], RMO_[[D9]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH15:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ14]])>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.rebind<() -> [[D6]]>([[V4]]) : !mo.tensor<[RMO_[[D9]]], si64> -> !mo.tensor<[[[D6]]], si64>
// CHECK-NEXT: mo.output<[[D6]]> [[V5]] : !mo.tensor<[[[D6]]], si64>

// -----

mo.graph @shape_of_static_static(%shape0: !mo.tensor<[?], si64>, %shape1: !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64> {
    %1 = rmo.broadcast_shape(%shape0, %shape1) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @shape_of_static_static<[[D1:[^ ]*]], [[D2:[^ ]*]] -> [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[[[D3]]], si64> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]]], si64> -> !mo.tensor<[[[D4]]], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]]], si64> -> !mo.tensor<[[[D5]]], si64>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ10:[^ ]*]]: i1 = <eq([[D4]], RMO_[[D7:[^ ]*]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH11:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ10]])>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> RMO_[[D7]]>([[V0]]) : !mo.tensor<[[[D4]]], si64> -> !mo.tensor<[RMO_[[D7]]], si64>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ12:[^ ]*]]: i1 = <eq([[D5]], RMO_[[D8:[^ ]*]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH13:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ12]])>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> RMO_[[D8]]>([[V1]]) : !mo.tensor<[[[D5]]], si64> -> !mo.tensor<[RMO_[[D8]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.broadcast_shape<() -> RMO_[[D9:[^ ]*]]>([[V2]], [[V3]]) : (!mo.tensor<[RMO_[[D7]]], si64>, !mo.tensor<[RMO_[[D8]]], si64>) -> !mo.tensor<[RMO_[[D9]]], si64>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ14:[^ ]*]]: i1 = <eq([[D6:[^ ]*]], RMO_[[D9]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH15:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ14]])>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.rebind<() -> [[D6]]>([[V4]]) : !mo.tensor<[RMO_[[D9]]], si64> -> !mo.tensor<[[[D6]]], si64>
// CHECK-NEXT: mo.output<[[D6]]> [[V5]] : !mo.tensor<[[[D6]]], si64>

// -----

mo.graph @shape_of_statici_statici(%shape0: !mo.tensor<[3], si64>, %shape1: !mo.tensor<[2], si64>) -> !mo.tensor<[?], si64> {
    %1 = rmo.broadcast_shape(%shape0, %shape1) : (!mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}

// CHECK-LABEL: mo.graph
// CHECK-SAME: @shape_of_statici_statici<() -> [[D1:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[3], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[[[D1]]], si64> no_inline {
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ6:[^ ]*]]: i1 = <eq(RMO_[[D3:[^ ]*]], 3)>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH7:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ6]])>
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> RMO_[[D3]]>([[VARG0]]) : !mo.tensor<[3], si64> -> !mo.tensor<[RMO_[[D3]]], si64>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ8:[^ ]*]]: i1 = <eq(RMO_[[D4:[^ ]*]], 2)>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH9:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ8]])>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> RMO_[[D4]]>([[VARG1]]) : !mo.tensor<[2], si64> -> !mo.tensor<[RMO_[[D4]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.broadcast_shape<() -> RMO_[[D5:[^ ]*]]>([[V0]], [[V1]]) : (!mo.tensor<[RMO_[[D3]]], si64>, !mo.tensor<[RMO_[[D4]]], si64>) -> !mo.tensor<[RMO_[[D5]]], si64>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_EQ10:[^ ]*]]: i1 = <eq([[D2:[^ ]*]], RMO_[[D5]])>
// CHECK-NEXT: kgen.param.declare [[RMO_SH_CH11:[^ ]*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[LegalizeRMOOps] introduced rebind fails." : !kgen.string}, [[RMO_SH_EQ10]])>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D2]]>([[V2]]) : !mo.tensor<[RMO_[[D5]]], si64> -> !mo.tensor<[[[D2]]], si64>
// CHECK-NEXT: mo.output<[[D2]]> [[V3]] : !mo.tensor<[[[D2]]], si64>
