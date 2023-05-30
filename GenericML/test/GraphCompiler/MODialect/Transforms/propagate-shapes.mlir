// RUN: mo-opt %s --propagate-shapes --canonicalize | FileCheck %s

// CHECK: mo.graph @output_param_decl_removed<[[SH0:.*]]: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
mo.graph @output_param_decl_removed<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, f32> {
  kgen.param.declare Sh2: !mosh.ape = <Sh0>
  %0 = mo.rebind(%arg0) : !mo.tensor<Sh0, f32> -> !mo.tensor<Sh2, f32>
  // CHECK-NEXT: %[[VAL0:.*]] = mo.gelu(%[[ARG0]]) : !mo.tensor<[[SH0]], f32>
  %1 = mo.gelu(%0) : !mo.tensor<Sh2, f32>
  // CHECK-NEXT: mo.output %[[VAL0]] : !mo.tensor<[[SH0]], f32>
  mo.output<:!mosh.ape Sh2> %1 : !mo.tensor<Sh2, f32>
}

// CHECK: mo.graph @output_param_decl_kept<[[SH0:.*]]: !mosh.ape, [[SH1:.*]]: !mosh.ape -> [[SH2:.*]]: !mosh.ape>(
mo.graph @output_param_decl_kept<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(
    // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>, %[[ARG1:.*]]: !mo.tensor<[[SH1]], f32>
    %arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>
// CHECK-SAME: ) -> !mo.tensor<[[SH2]], f32> {
) -> !mo.tensor<Sh2, f32> {
  // CHECK-NEXT: %[[VAL0:.*]] = mo.batch_matmul<() -> [[SH3:.*]]: !mosh.ape>(%[[ARG0]], %[[ARG1]]) : (
  // CHECK-SAME:   !mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH3]], f32>
  %0 = mo.batch_matmul<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (
    !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh3, f32>
  // CHECK-NEXT: mo.output<:!mosh.ape [[SH3]]> %[[VAL0]] : !mo.tensor<[[SH3]], f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}

// CHECK: @batch_matmul_without_certain_checks<[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(
mo.graph @batch_matmul_without_certain_checks<D0, D1, D2, D3 -> D4, D5, D6>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>,
  %arg0: !mo.tensor<[D0, D1, D2], f32>,
  // CHECK-SAME: %[[ARG1:.*]]: !mo.tensor<[[[D0]], [[D2]], [[D3]]], f32>
  %arg1: !mo.tensor<[D0, D2, D3], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D3]]], f32> {
) -> !mo.tensor<[D4, D5, D6], f32> {
  kgen.param.declare Sh0: !mosh.ape = <[D0, D1, D2]>
  %0 = mo.rebind(%arg0) : !mo.tensor<[D0, D1, D2], f32> -> !mo.tensor<Sh0, f32>
  kgen.param.declare Sh1: !mosh.ape = <[D0, D2, D3]>
  %1 = mo.rebind(%arg1) : !mo.tensor<[D0, D2, D3], f32> -> !mo.tensor<Sh1, f32>
  kgen.param.declare P0 = <2>
  kgen.param.declare P1 = <apply(:(!mosh.ape) -> index "mosh.get_rank", Sh0)>
  kgen.param.declare P2 = <apply(:(!mosh.ape) -> index "mosh.get_rank", Sh1)>
  kgen.param.declare P3: i1 = <apply(:(index, index) -> i1 "index.cmp"{pred = #index<cmp_predicate eq>}, P1, P2)>
  kgen.param.declare CH0: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "ranks of left and right inputs are not equal" : !kgen.string}, P3)>
  kgen.param.declare P4: i1 = <apply(:(index, index) -> i1 "index.cmp"{pred = #index<cmp_predicate sgt>}, P2, P0)>
  kgen.param.declare CH1: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "input ranks are not greater than 2" : !kgen.string}, P4)>
  kgen.param.declare P8: !mosh.ape = <apply(:(!mosh.ape, index, index) -> !mosh.ape "mosh.slice", Sh0, 0, -2)>
  kgen.param.declare P11 = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", Sh0, -2)>
  kgen.param.declare P12 = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", Sh1, -1)>
  kgen.param.declare P13: !mosh.ape = <[P11, P12]>
  kgen.param.declare P14: !mosh.ape = <apply(:(!mosh.ape, !mosh.ape) -> !mosh.ape "mosh.concat", P8, P13)>
  kgen.param.declare Sh2: !mosh.ape = <P14>
  // CHECK-NEXT: %[[OUT:.*]] = mo.batch_matmul(%arg0, %arg1) : (
  // CHECK-SAME:   !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>,
  // CHECK-SAME:   !mo.tensor<[[[D0]], [[D2]], [[D3]]], f32>
  // CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D3]]], f32>
  %2 = mo.batch_matmul(%0, %1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
  kgen.param.declare D7 = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", Sh2, 0)>
  kgen.param.declare D8 = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", Sh2, 1)>
  kgen.param.declare D9 = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", Sh2, 2)>
  %3 = mo.rebind(%2) : !mo.tensor<Sh2, f32> -> !mo.tensor<[D7, D8, D9], f32>
  // CHECK-NEXT: mo.output %[[OUT]] : !mo.tensor<[[[D0]], [[D1]], [[D3]]], f32>
  mo.output<D7, D8, D9> %3 : !mo.tensor<[D7, D8, D9], f32>
}

// CHECK: mo.graph @unfolded_apply_shouldnt_be_inlined<[[SH0:.*]]: !mosh.ape, [[SH1:.*]]: !mosh.ape -> [[SH2:.*]]: !mosh.ape>(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>,
// CHECK-SAME: %[[ARG1:.*]]: !mo.tensor<[[SH1]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[SH2]], f32> {
mo.graph @unfolded_apply_shouldnt_be_inlined<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(
  %arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>
) -> !mo.tensor<Sh2, f32> {
  // CHECK-NEXT: kgen.param.declare [[P1:.*]] = <apply(:(!mosh.ape) -> index "mosh.get_rank", [[SH0]])>
  kgen.param.declare P1 = <apply(:(!mosh.ape) -> index "mosh.get_rank", Sh0)>
  // CHECK-NEXT: kgen.param.declare [[P2:.*]] = <apply(:(!mosh.ape) -> index "mosh.get_rank", [[SH1]])>
  kgen.param.declare P2 = <apply(:(!mosh.ape) -> index "mosh.get_rank", Sh1)>
  // CHECK-NEXT: kgen.param.declare [[P3:.*]]: i1 = <apply(:(index, index) -> i1 "index.cmp"{pred = #index<cmp_predicate eq>}, [[P1]], [[P2]])>
  kgen.param.declare P3: i1 = <apply(:(index, index) -> i1 "index.cmp"{pred = #index<cmp_predicate eq>}, P1, P2)>
  // CHECK-NEXT: kgen.param.declare [[CH0:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "" : !kgen.string}, [[P3]])>
  kgen.param.declare CH0: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "" : !kgen.string}, P3)>

  // CHECK-NEXT: kgen.param.declare [[P4:.*]]: !mosh.ape = <apply(:(!mosh.ape, index, index) -> !mosh.ape "mosh.slice", [[SH0]], 0, -2)>
  kgen.param.declare P4: !mosh.ape = <apply(:(!mosh.ape, index, index) -> !mosh.ape "mosh.slice", Sh0, 0, -2)>
  // CHECK-NEXT: %[[OUT:.*]] = mo.rebind(%[[ARG0]]) : !mo.tensor<[[SH0]], f32> -> !mo.tensor<[[P4]], f32>
  %1 = mo.rebind(%arg0) : !mo.tensor<Sh0, f32> -> !mo.tensor<P4, f32>

  // CHECK-NEXT: mo.output<:!mosh.ape [[P4]]> %[[OUT]] : !mo.tensor<[[P4]], f32>
  mo.output<:!mosh.ape P4> %1 : !mo.tensor<P4, f32>
}

// CHECK: mo.graph @graph_with_param_decls_in_ops<[[D0:.*]] -> [[D1:.*]], [[D2:.*]]>(
mo.graph @graph_with_param_decls_in_ops<D0 -> D1, D2>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[D0]], 20, ?], f32>,
  %arg0: !mo.tensor<[D0, 20, ?], f32>,
  // CHECK-SAME: %[[SHAPE:.*]]: !mo.tensor<[2], si32>
  %shape: !mo.tensor<[2], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D1]], [[D2]]], f32> {
) -> !mo.tensor<[D1, D2], f32> {
  kgen.param.declare D3 = <D0>
  kgen.param.declare D4 = <20>
  // CHECK-NEXT: %[[VAL0:.*]] = mo.rebind<() -> [[D5:.*]]>(%[[ARG0]]) :
  // CHECK-SAME: !mo.tensor<[[[D0]], 20, ?], f32> -> !mo.tensor<[[[D0]], 20, [[D5]]], f32>
  %0 = mo.rebind<() -> D5>(%arg0) : !mo.tensor<[D0, 20, ?], f32> -> !mo.tensor<[D3, D4, D5], f32>
  // CHECK-NEXT: %[[VAL1:.*]] = mo.reshape<() -> [[D6:.*]], [[D7:.*]]>(%[[VAL0]], %[[SHAPE]]) : (
  // CHECK-SAME:   !mo.tensor<[[[D0]], 20, [[D5]]], f32>, !mo.tensor<[2], si32>
  // CHECK-SAME: ) -> !mo.tensor<[[[D6]], [[D7]]], f32>
  %1 = mo.reshape<() -> D6, D7>(%0, %shape) : (
    !mo.tensor<[D3, D4, D5], f32>, !mo.tensor<[2], si32>
  ) -> !mo.tensor<[D6, D7], f32>
  kgen.param.declare D8 = <D5>
  kgen.param.declare D9 = <D7>
  // CHECK-NEXT: %[[VAL2:.*]] = mo.rebind(%[[VAL1]]) :
  // CHECK-SAME: !mo.tensor<[[[D6]], [[D7]]], f32> -> !mo.tensor<[[[D5]], [[D7]]], f32>
  %2 = mo.rebind(%1) : !mo.tensor<[D6, D7], f32> -> !mo.tensor<[D8, D9], f32>
  // CHECK-NEXT: mo.output<[[D5]], [[D7]]> %[[VAL2]] : !mo.tensor<[[[D5]], [[D7]]], f32>
  mo.output<D8, D9> %2 : !mo.tensor<[D8, D9], f32>
}

// CHECK: mo.graph @nested_div_mul_simplification<[[D0:.*]], [[D1:.*]]>(
mo.graph @nested_div_mul_simplification<D0, D1 -> D2>(
  // CHECK-SAME: %[[ARG:.*]]: !mo.tensor<[[[D0]], [[D1]], 20], f32>
  %arg: !mo.tensor<[D0, D1, 20], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], 4, 5], f32> {
) -> !mo.tensor<[D0, D2, 4, 5], f32> {
  // CHECK-NEXT: kgen.param.declare [[NUM:.*]] = <mul_nuw([[D0]], [[D1]], 20)>
  // CHECK-NEXT: kgen.param.declare [[DENOM:.*]] = <mul_nuw([[D0]], 20)>
  kgen.param.declare NUM = <mul_nuw(D0, D1, 20)>
  kgen.param.declare DENOM = <mul_nuw(D0, 4, 5)>
  kgen.param.declare DIV = <div(NUM, DENOM)>
  // CHECK-NEXT: %[[RES:.*]] = mo.static.reshape(%[[ARG]])
  // CHECK-SAME: : !mo.tensor<[[[D0]], [[D1]], 20], f32> -> !mo.tensor<[[[D0]], [[D1]], 4, 5], f32>
  %res = mo.static.reshape(%arg) : !mo.tensor<[D0, D1, 20], f32> -> !mo.tensor<[D0, DIV, 4, 5], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D0]], [[D1]], 4, 5], f32>
  mo.output<DIV> %res : !mo.tensor<[D0, DIV, 4, 5], f32>
}

// CHECK: mo.graph @param_expr_cse<[[D0:.*]], [[D1:.*]] -> [[D2:.*]], [[D4:.*]]>(
mo.graph @param_expr_cse<D0, D1 -> D2, D3, D4>(
  // CHECK-SAME: %[[ARG:.*]]: !mo.tensor<[[[D0]], [[D1]], 20], f32>
  %arg: !mo.tensor<[D0, D1, 20], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D2]], 42, [[D4]]], f32> {
) -> !mo.tensor<[D2, D3, D4], f32> {
  // CHECK-NEXT: kgen.param.declare [[D5:.*]] = <mul_nuw([[D0]], 20)>
  // CHECK-NEXT: kgen.param.declare [[D6:.*]] = <mul_nuw([[D0]], 20)>
  // CHECK-NEXT: kgen.param.declare [[D7:.*]] = <add([[D0]], 20)>
  kgen.param.declare D5 = <mul_nuw(D0, 20)>
  kgen.param.declare D6 = <mul_nuw(D0, 20)>
  kgen.param.declare D7 = <add(D0, 20)>
  // CHECK-NEXT: %[[RESHAPED:.*]] = mo.static.reshape(%[[ARG]])
  // CHECK-SAME: : !mo.tensor<[[[D0]], [[D1]], 20], f32> -> !mo.tensor<[[[D5]], [[D5]], [[D7]]], f32>
  %reshaped = mo.static.reshape(%arg) : !mo.tensor<[D0, D1, 20], f32> -> !mo.tensor<[D5, D6, D7], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.static.broadcast_to(%[[RESHAPED]])
  // CHECK-SAME: : !mo.tensor<[[[D5]], [[D5]], [[D7]]], f32> -> !mo.tensor<[[[D5]], 42, [[D5]]], f32>
  %res = mo.static.broadcast_to(%reshaped) : !mo.tensor<[D5, D6, D7], f32> -> !mo.tensor<[D6, 42, D5], f32>
  // CHECK-NEXT: mo.output<[[D5]], [[D5]]> %[[RES]] : !mo.tensor<[[[D5]], 42, [[D5]]], f32>
  mo.output<D6, 42, D5> %res : !mo.tensor<[D6, 42, D5], f32>
}

// CHECK: mo.graph @param_expr_apply_no_cse<[[SH0:.*]]: !mosh.ape -> [[D2:.*]], [[D3:.*]]>(
mo.graph @param_expr_apply_no_cse<Sh0: !mosh.ape -> D2, D3>(
  // CHECK-SAME: %[[ARG:.*]]: !mo.tensor<[[SH0]], f32>
  %arg: !mo.tensor<Sh0, f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D2]], [[D3]]], f32> {
) -> !mo.tensor<[D2, D3], f32> {
  // CHECK-NEXT: kgen.param.declare [[D5:.*]] = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", [[SH0]], 1)>
  // CHECK-NEXT: kgen.param.declare [[D6:.*]] = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", [[SH0]], 1)>
  kgen.param.declare D5 = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", Sh0, 1)>
  kgen.param.declare D6 = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", Sh0, 1)>
  // CHECK-NEXT: %[[RES:.*]] = mo.static.reshape(%[[ARG]])
  // CHECK-SAME: : !mo.tensor<[[SH0]], f32> -> !mo.tensor<[[[D5]], [[D6]]], f32>
  %res = mo.static.reshape(%arg) : !mo.tensor<Sh0, f32> -> !mo.tensor<[D5, D6], f32>
  // CHECK-NEXT: mo.output<[[D5]], [[D6]]> %[[RES]] : !mo.tensor<[[[D5]], [[D6]]], f32>
  mo.output<D5, D6> %res : !mo.tensor<[D5, D6], f32>
}
