// RUN: mo-opt %s --propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @output_param_decl_removed
// CHECK-SAME: <[[SH0:.*]]: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> no_inline {
mo.graph @output_param_decl_removed<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, f32> {
  kgen.param.declare Sh2: !mosh.ape = <Sh0>
  %0 = mo.rebind(%arg0) : !mo.tensor<Sh0, f32> -> !mo.tensor<Sh2, f32>
  // CHECK-NEXT: %[[VAL0:.*]] = mo.gelu(%[[ARG0]]) : !mo.tensor<[[SH0]], f32>
  %1 = mo.gelu(%0) : !mo.tensor<Sh2, f32>
  // CHECK-NEXT: mo.output %[[VAL0]] : !mo.tensor<[[SH0]], f32>
  mo.output<:!mosh.ape Sh2> %1 : !mo.tensor<Sh2, f32>
}

// CHECK-LABEL: mo.graph @output_param_decl_kept
// CHECK-SAME: <[[SH0:.*]]: !mosh.ape, [[SH1:.*]]: !mosh.ape -> [[SH2:.*]]: !mosh.ape>(
mo.graph @output_param_decl_kept<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(
    // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>, %[[ARG1:.*]]: !mo.tensor<[[SH1]], f32>
    %arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>
// CHECK-SAME: ) -> !mo.tensor<[[SH2]], f32> no_inline {
) -> !mo.tensor<Sh2, f32> {
  // CHECK-NEXT: %[[VAL0:.*]] = mo.batch_matmul<() -> [[SH3:.*]]: !mosh.ape>(%[[ARG0]], %[[ARG1]]) : (
  // CHECK-SAME:   !mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH3]], f32>
  %0 = mo.batch_matmul<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (
    !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh3, f32>
  // CHECK-NEXT: mo.output<:!mosh.ape [[SH3]]> %[[VAL0]] : !mo.tensor<[[SH3]], f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}

// CHECK-LABEL: mo.graph @batch_matmul_without_certain_checks
// CHECK-SAME: <[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(
mo.graph @batch_matmul_without_certain_checks<D0, D1, D2, D3 -> D4, D5, D6>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>,
  %arg0: !mo.tensor<[D0, D1, D2], f32>,
  // CHECK-SAME: %[[ARG1:.*]]: !mo.tensor<[[[D0]], [[D2]], [[D3]]], f32>
  %arg1: !mo.tensor<[D0, D2, D3], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D3]]], f32> no_inline {
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

// CHECK-LABEL: mo.graph @unfolded_apply_shouldnt_be_inlined
// CHECK-SAME: <[[SH0:.*]]: !mosh.ape, [[SH1:.*]]: !mosh.ape -> [[SH2:.*]]: !mosh.ape>(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>,
// CHECK-SAME: %[[ARG1:.*]]: !mo.tensor<[[SH1]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[SH2]], f32> no_inline {
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

// CHECK-LABEL: mo.graph @graph_with_param_decls_in_ops
// CHECK-SAME: <[[D0:.*]] -> [[D1:.*]], [[D2:.*]]>(
mo.graph @graph_with_param_decls_in_ops<D0 -> D1, D2>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[D0]], 20, ?], f32>,
  %arg0: !mo.tensor<[D0, 20, ?], f32>,
  // CHECK-SAME: %[[SHAPE:.*]]: !mo.tensor<[2], si32>
  %shape: !mo.tensor<[2], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D1]], [[D2]]], f32> no_inline {
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

// CHECK-LABEL: mo.graph @nested_div_mul_simplification
// CHECK-SAME: <[[D0:.*]], [[D1:.*]]>(
mo.graph @nested_div_mul_simplification<D0, D1 -> D2>(
  // CHECK-SAME: %[[ARG:.*]]: !mo.tensor<[[[D0]], [[D1]], 20], f32>
  %arg: !mo.tensor<[D0, D1, 20], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], 4, 5], f32> no_inline {
) -> !mo.tensor<[D0, D2, 4, 5], f32> {
  kgen.param.declare NUM = <mul_nuw(D0, D1, 20)>
  kgen.param.declare DENOM = <mul_nuw(D0, 4, 5)>
  kgen.param.declare DIV = <div(NUM, DENOM)>
  // CHECK-NEXT: %[[RES:.*]] = mo.static.reshape(%[[ARG]])
  // CHECK-SAME: : !mo.tensor<[[[D0]], [[D1]], 20], f32> -> !mo.tensor<[[[D0]], [[D1]], 4, 5], f32>
  %res = mo.static.reshape(%arg) : !mo.tensor<[D0, D1, 20], f32> -> !mo.tensor<[D0, DIV, 4, 5], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D0]], [[D1]], 4, 5], f32>
  mo.output<DIV> %res : !mo.tensor<[D0, DIV, 4, 5], f32>
}


// CHECK-LABEL: mo.graph @param_expr_apply_no_cse
// CHECK-SAME: <[[SH0:.*]]: !mosh.ape -> [[D2:.*]], [[D3:.*]]>(
mo.graph @param_expr_apply_no_cse<Sh0: !mosh.ape -> D2, D3>(
  // CHECK-SAME: %[[ARG:.*]]: !mo.tensor<[[SH0]], f32>
  %arg: !mo.tensor<Sh0, f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D2]], [[D3]]], f32> no_inline {
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

// CHECK-LABEL: mo.graph @param_expr_dce
// CHECK-SAME:   <[[D0:.*]]>(%[[INPUT:.*]]: !mo.tensor<[[[D0]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]]], f32> no_inline {
mo.graph @param_expr_dce<D0 -> D1>(
  %input: !mo.tensor<[D0], f32>
) -> !mo.tensor<[D1], f32> {

  // These are trivial and DCEed away
  kgen.param.declare D2 = <add(D0, 4)>
  kgen.param.declare D3 = <sub(D2, 4)>
  %res = mo.static.reshape(%input) : !mo.tensor<[D0], f32> -> !mo.tensor<[D3], f32>

  // CHECK-NEXT: mo.output %[[INPUT]] : !mo.tensor<[[[D0]]], f32>
  mo.output<D3> %res : !mo.tensor<[D3], f32>
}

// CHECK-LABEL: mo.graph @fold_nested_max
// CHECK-SAME: <[[D0:.*]], [[BS:.*]] -> [[OUT:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]]], f32>,
// CHECK-SAME:   %[[DUMMY:.*]]: !mo.tensor<[[[BS]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[OUT]], 768], f32> no_inline {
mo.graph @fold_nested_max<D0, bs -> out>(
  %input: !mo.tensor<[D0], f32>,
  %dummy: !mo.tensor<[bs], f32> // this is used solely to help define `bs`
) -> !mo.tensor<[out, 768], f32> {

  kgen.param.declare D1 = <max(bs, 1)>
  kgen.param.declare D2 = <max(D1, bs)>

  // CHECK-NEXT: %[[INPUT_RS:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[[[D0]]], f32> -> !mo.tensor<[max([[BS]], 1), 768], f32>
  %input_rs = mo.static.reshape(%input) : !mo.tensor<[D0], f32> -> !mo.tensor<[D1, 768], f32>

  // Simplified and removed.
  %res = mo.static.broadcast_to(%input_rs) : !mo.tensor<[D1, 768], f32> -> !mo.tensor<[D2, 768], f32>

  // CHECK-NEXT: mo.output<max([[BS]], 1)> %[[INPUT_RS]] : !mo.tensor<[max([[BS]], 1), 768], f32>
  mo.output<D2> %res : !mo.tensor<[D2, 768], f32>
}

// CHECK-LABEL: mo.graph @non_mo_tensor_input
mo.graph @non_mo_tensor_input<D0 -> D1>(
  // CHECK-SAME: <[[D0:.*]]>(%[[INPUT:.*]]: !mo.tensor<[[[D0]]], f32>, %[[SHAPE:.*]]: !mosh.ape
  %input: !mo.tensor<[D0], f32>, %shape: !mosh.ape
// CHECK-SAME: ) -> (!mo.tensor<[[[D0]]], f32>, !mosh.ape) no_inline {
) -> (!mo.tensor<[D1], f32>, !mosh.ape) {

  // CHECK-NEXT: mo.output %[[INPUT]], %[[SHAPE]] : !mo.tensor<[[[D0]]], f32>, !mosh.ape
  mo.output<D0> %input, %shape : !mo.tensor<[D0], f32>, !mosh.ape
}

// CHECK-LABEL: mo.graph @internal_dim_param_as_output_binding
// CHECK-SAME:  <[[D0:.*]] -> [[RES:.*]]: !mosh.ape>
// CHECK-SAME:    %[[INPUT:.*]]: !mo.tensor<[[[D0]]], f32>
// CHECK-SAME:  ) -> !mo.tensor<[[RES]], si64> no_inline {
mo.graph @internal_dim_param_as_output_binding<D0 -> res: !mosh.ape>(
  %input: !mo.tensor<[D0], f32>
) -> !mo.tensor<res, si64> {

  // CHECK-NEXT: %[[RES:.*]] = mo.arg_nonzero<() -> [[D2:.*]]>(%[[INPUT]]) : (!mo.tensor<[[[D0]]], f32>) -> !mo.tensor<[[[D2]], 1], si64>
  %res = mo.arg_nonzero<() -> D2>(%input) : (!mo.tensor<[D0], f32>) -> !mo.tensor<[D2, 1], si64>

  // CHECK-NEXT: mo.output<:!mosh.ape [[[D2]], 1]> %[[RES]] : !mo.tensor<[[[D2]], 1], si64>
  mo.output<:!mosh.ape [D2, 1]> %res : !mo.tensor<[D2, 1], si64>
}

// CHECK-LABEL: mo.graph @out_of_order_instructions
// CHECK-SAME:  <[[D0:.*]] -> [[OUT:.*]]>(
// CHECK-SAME:   %[[SCALAR:.*]]: !mo.tensor<[], f32>,
// CHECK-SAME:   %[[DUMMY:.*]]: !mo.tensor<[[[D0]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[OUT]]], f32> no_inline {
mo.graph @out_of_order_instructions<D0 -> out>(
  %scalar: !mo.tensor<[], f32>,
  %dummy: !mo.tensor<[D0], f32> // this is used solely to help define `D0`
) -> !mo.tensor<[out], f32> {

  kgen.param.declare D3 = <div(D2, D0)>
  kgen.param.declare D2 = <mul(D1, D1)>

  // CHECK-NEXT: %[[RES:.*]] = mo.static.broadcast_to(%[[SCALAR]]) : !mo.tensor<[], f32> -> !mo.tensor<[div(add(mul([[D0]], [[D0]]), mul([[D0]], 2), 1), [[D0]])], f32>
  %res = mo.static.broadcast_to(%scalar) : !mo.tensor<[], f32> -> !mo.tensor<[D3], f32>

  kgen.param.declare D1 = <add(D0, 1)>

  // CHECK-NEXT: mo.output<div(add(mul([[D0]], [[D0]]), mul([[D0]], 2), 1), [[D0]])> %[[RES]] : !mo.tensor<[div(add(mul([[D0]], [[D0]]), mul([[D0]], 2), 1), [[D0]])], f32>
  mo.output<D3> %res : !mo.tensor<[D3], f32>
}

// CHECK-LABEL: mo.graph @nested_mul
mo.graph @nested_mul<D0>() {

  // These KGEN expressions are all folded away

  // NOTE that this gets canonicalized into `mul(D0, 2)`
  kgen.param.declare P0 = <add(D0, D0)>
  kgen.param.declare P1 = <mod(P0, D0)>
  kgen.param.declare P2: i1 = <eq(P1, 0)>
  kgen.param.declare P3: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "error" : !kgen.string}, P2)>

  // CHECK-NEXT: mo.output
  mo.output
}

// CHECK-LABEL: mo.graph @range_based_folding
// CHECK-SAME:  <[[D0:.*]] -> [[O3:.*]]>(
// CHECK-SAME:    %[[IN:.*]]: !mo.tensor<[], f32>
// CHECK-SAME:  ) -> !mo.tensor<[[[D0]], 0, 0, [[O3]]], f32> no_inline {
mo.graph @range_based_folding<D0 -> O0, O1, O2, O3>(
  %in: !mo.tensor<[], f32>
) -> !mo.tensor<[O0, O1, O2, O3], f32> {

  kgen.param.declare P0 = <max(D0, 0)> // D0
  kgen.param.declare P1 = <max(mul_nuw(D0, -1), 0)> // 0
  kgen.param.declare P2 = <min(D0, 0)> // 0
  kgen.param.declare P3 = <min(mul_nuw(D0, -1), 0)> // mul_nuw(D0, -1)

  // CHECK-NEXT: %[[OUT:.*]] = mo.static.broadcast_to(%[[IN]])
  // CHECK-SAME: : !mo.tensor<[], f32> -> !mo.tensor<[[[D0]], 0, 0, mul_nuw([[D0]], -1)], f32>
  %out = mo.static.broadcast_to(%in) : !mo.tensor<[], f32> -> !mo.tensor<[P0, P1, P2, P3], f32>

  // CHECK-NEXT: mo.output<mul_nuw([[D0]], -1)> %[[OUT]] : !mo.tensor<[[[D0]], 0, 0, mul_nuw([[D0]], -1)], f32>
  mo.output<P0, P1, P2, P3> %out : !mo.tensor<[P0, P1, P2, P3], f32>
}


// This optimization simplifies statements of the form `<max(add(A, B), B)>`
// where we know `A, B >= 0`. We know `A, B >= 0` because they appear in a shape.
// As a result `A + B >= B` and `<max(add(A, B), B)> = add(A, B)`

// CHECK-LABEL: mo.graph @propagate_max_add_optimization
// CHECK-SAME: <
// CHECK-SAME: [[A:[a-zA-Z0-9_]*]], [[B:[a-zA-Z0-9_]*]] ->
// CHECK-SAME: [[D1:[a-zA-Z0-9_]*]], [[D2:[a-zA-Z0-9_]*]], [[D3:[a-zA-Z0-9_]*]], [[D4:[a-zA-Z0-9_]*]]
// CHECK-SAME: >
// CHECK-SAME: (
// CHECK-SAME: [[IN:%[a-zA-Z0-9_]*]]: !mo.tensor<[[[A]], [[B]]], f32>
// CHECK-SAME: ) ->
// CHECK-SAME: (!mo.tensor<[[[D1]], 1], f32>, !mo.tensor<[[[D2]], 1], f32>,
// CHECK-SAME: !mo.tensor<[[[D3]], 1], f32>, !mo.tensor<[[[D4]], 1], f32>)
mo.graph @propagate_max_add_optimization<A, B -> D1, D2, D3, D4>
  (%arg0: !mo.tensor<[A, B], f32>) ->
  (!mo.tensor<[D1, 1], f32>, !mo.tensor<[D2, 1], f32>, !mo.tensor<[D3, 1], f32>, !mo.tensor<[D4, 1], f32>)
{
  kgen.param.declare expr1 = <max(add(A, B), B)>
  kgen.param.declare expr2 = <max(add(A, B), A)>
  kgen.param.declare expr3 = <max(A, add(A, B))>
  kgen.param.declare expr4 = <max(B, add(A, B))>

  // NOTE: no A and B exist which fufill this rebind safely, but we just have it
  // to easily test the simplification of this MAX property.
  // CHECK-NEXT: [[V0:%.*]] = mo.rebind([[IN]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[add([[A]], [[B]]), 1], f32>
  // CHECK-NEXT: [[V1:%.*]] = mo.rebind([[IN]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[add([[A]], [[B]]), 1], f32>
  // CHECK-NEXT: [[V2:%.*]] = mo.rebind([[IN]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[add([[A]], [[B]]), 1], f32>
  // CHECK-NEXT: [[V3:%.*]] = mo.rebind([[IN]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[add([[A]], [[B]]), 1], f32>
  %a = mo.rebind(%arg0): !mo.tensor<[A, B], f32> -> !mo.tensor<[expr1, 1], f32>
  %b = mo.rebind(%arg0): !mo.tensor<[A, B], f32> -> !mo.tensor<[expr2, 1], f32>
  %c = mo.rebind(%arg0): !mo.tensor<[A, B], f32> -> !mo.tensor<[expr3, 1], f32>
  %d = mo.rebind(%arg0): !mo.tensor<[A, B], f32> -> !mo.tensor<[expr4, 1], f32>

  // CHECK-NEXT: mo.output
  // CHECK-SAME: [[V0]], [[V1]], [[V2]], [[V3]]
  mo.output<expr1, expr2, expr3, expr4> %a, %b, %c, %d : !mo.tensor<[expr1, 1], f32>, !mo.tensor<[expr2, 1], f32>, !mo.tensor<[expr3, 1], f32>, !mo.tensor<[expr4, 1], f32>
}

// CHECK: mo.graph @expand_apply_mosh_broadcast<[[D0:.*]], [[D1:.*]] -> O0, O1>([[VARG0:%.*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[O0, O1], f32>
mo.graph @expand_apply_mosh_broadcast<D0, D1 -> O0, O1>(
  %in: !mo.tensor<[D0, D1], f32>
) -> !mo.tensor<[O0, O1], f32> {

  // CHECK-DAG: kgen.param.declare dims_assert5: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions are not broadcastable" : !kgen.string}, or(eq([[D0]], [[D1]]), eq([[D0]], 1), eq([[D1]], 1)))>
  kgen.param.declare NEW_SHAPE: !mosh.ape = <apply(:(!mosh.ape, !mosh.ape) -> !mosh.ape "mosh.broadcast", [D0], [D1])>
  kgen.param.declare R1 = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", NEW_SHAPE, 0)>

  %out = mo.static.broadcast_to(%in): !mo.tensor<[D0, D1], f32> -> !mo.tensor<[R1, R1], f32>

  // CHECK-NEXT: [[V0:%.*]] = mo.static.broadcast_to([[VARG0]]) : !mo.tensor<[[[D0]], [[D1]]], f32> -> !mo.tensor<[cond(eq([[D0]], 1), [[D1]], [[D0]]), cond(eq([[D0]], 1), [[D1]], [[D0]])], f32>
  // CHECK-NEXT: mo.output<cond(eq([[D0]], 1), [[D1]], [[D0]]), cond(eq([[D0]], 1), [[D1]], [[D0]])> [[V0]] : !mo.tensor<[cond(eq([[D0]], 1), [[D1]], [[D0]]), cond(eq([[D0]], 1), [[D1]], [[D0]])], f32>
  mo.output<R1, R1> %out : !mo.tensor<[R1, R1], f32>
}

// CHECK-LABEL: mo.graph @update_param_declare_rhs
// CHECK-SAME:  <[[D1:.*]]>() no_inline {
mo.graph @update_param_declare_rhs<D1>() {
  // In cases like this, where we do some folding in the RHS of a
  // `kgen.param.decalre`, we still want to update the RHS to the simplified
  // version in the IR (not just in the evaluator during the pass).

  // CHECK-NEXT: kgen.param.declare {{.*}}: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "test" : !kgen.string}, eq([[D1]], 1))>
  // CHECK-NEXT: mo.output
  kgen.param.declare CH: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "test" : !kgen.string}, eq(cond(eq(D1, -1), 1, D1), 1))>
  mo.output
}
