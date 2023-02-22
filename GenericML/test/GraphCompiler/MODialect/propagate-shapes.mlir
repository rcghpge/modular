// RUN: mo-opt %s --propagate-shapes --canonicalize | FileCheck %s

// CHECK: mo.graph @output_param_decl_removed<[[SH0:.*]]: !mo.shape>(%[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
mo.graph @output_param_decl_removed<Sh0: !mo.shape -> Sh1: !mo.shape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, f32> {
  kgen.param.declare Sh2: !mo.shape = <Sh0>
  %0 = mo.rebind(%arg0) : !mo.tensor<Sh0, f32> -> !mo.tensor<Sh2, f32>
  // CHECK-NEXT: %[[VAL0:.*]] = mo.gelu(%[[ARG0]]) : !mo.tensor<[[SH0]], f32>
  %1 = mo.gelu(%0) : !mo.tensor<Sh2, f32>
  // CHECK-NEXT: mo.output %[[VAL0]] : !mo.tensor<[[SH0]], f32>
  mo.output<:!mo.shape Sh2> %1 : !mo.tensor<Sh2, f32>
}

// CHECK: mo.graph @output_param_decl_kept<[[SH0:.*]]: !mo.shape, [[SH1:.*]]: !mo.shape -> [[SH2:.*]]: !mo.shape>(
mo.graph @output_param_decl_kept<Sh0: !mo.shape, Sh1: !mo.shape -> Sh2: !mo.shape>(
    // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>, %[[ARG1:.*]]: !mo.tensor<[[SH1]], f32>
    %arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>
// CHECK-SAME: ) -> !mo.tensor<[[SH2]], f32> {
) -> !mo.tensor<Sh2, f32> {
  // CHECK-NEXT: %[[VAL0:.*]] = mo.batch_matmul<() -> [[SH3:.*]]: !mo.shape>(%[[ARG0]], %[[ARG1]]) : (
  // CHECK-SAME:   !mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH3]], f32>
  %0 = mo.batch_matmul<() -> Sh3: !mo.shape>(%arg0, %arg1) : (
    !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh3, f32>
  // CHECK-NEXT: mo.output<:!mo.shape [[SH3]]> %[[VAL0]] : !mo.tensor<[[SH3]], f32>
  mo.output<:!mo.shape Sh3> %0 : !mo.tensor<Sh3, f32>
}

// CHECK: @batch_matmul_without_certain_checks<[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(
mo.graph @batch_matmul_without_certain_checks<D0, D1, D2, D3>(
// CHECK-SAME: %arg0: !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>,
  %arg0: !mo.tensor<[D0, D1, D2], f32>,
// CHECK-SAME: %arg1: !mo.tensor<[[[D0]], [[D2]], [[D3]]], f32>
  %arg1: !mo.tensor<[D0, D2, D3], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32> {
) -> !mo.tensor<[D0, D1, D2], f32> {
  kgen.param.declare Sh0: !mo.shape = <[D0, D1, D2]>
  %0 = mo.rebind(%arg0) : !mo.tensor<[D0, D1, D2], f32> -> !mo.tensor<Sh0, f32>
  kgen.param.declare Sh1: !mo.shape = <[D0, D2, D3]>
  %1 = mo.rebind(%arg1) : !mo.tensor<[D0, D2, D3], f32> -> !mo.tensor<Sh1, f32>
  kgen.param.declare P0 = <2>
  kgen.param.declare P1 = <apply(:(!mo.shape) -> index "mo.shape.get_rank", Sh0)>
  kgen.param.declare P2 = <apply(:(!mo.shape) -> index "mo.shape.get_rank", Sh1)>
  kgen.param.declare P3: i1 = <apply(:(index, index) -> i1 "index.cmp"{pred = #index<cmp_predicate eq>}, P1, P2)>
  kgen.param.assert <P3>, "ranks of left and right inputs are not equal"
  kgen.param.declare P4: i1 = <apply(:(index, index) -> i1 "index.cmp"{pred = #index<cmp_predicate sgt>}, P2, P0)>
  kgen.param.assert <P4>, "input ranks are not greater than 2"
  kgen.param.declare P8: !mo.shape = <apply(:(!mo.shape) -> !mo.shape "mo.shape.slice"{end = -2 : index, start = 0 : index}, Sh0)>
  kgen.param.declare P11 = <apply(:(!mo.shape) -> index "mo.shape.get_dim"{dim = -2 : index}, Sh0)>
  kgen.param.declare P12 = <apply(:(!mo.shape) -> index "mo.shape.get_dim"{dim = -1 : index}, Sh1)>
  kgen.param.declare P13: !mo.shape = <[P11, P12]>
  kgen.param.declare P14: !mo.shape = <apply(:(!mo.shape, !mo.shape) -> !mo.shape "mo.shape.concat", P8, P13)>
  kgen.param.declare Sh2: !mo.shape = <P14>
  // CHECK-NEXT: %0 = mo.batch_matmul(%arg0, %arg1) : (
  // CHECK-SAME:   !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>,
  // CHECK-SAME:   !mo.tensor<[[[D0]], [[D2]], [[D3]]], f32>
  // CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D3]]], f32>
  %2 = mo.batch_matmul(%0, %1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
  // CHECK-NEXT: mo.output %arg0 : !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>
  mo.output %arg0 : !mo.tensor<[D0, D1, D2], f32>
}

// CHECK: mo.graph @unfolded_apply_shouldnt_be_inlined<[[SH0:.*]]: !mo.shape, [[SH1:.*]]: !mo.shape -> [[SH2:.*]]: !mo.shape>(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>,
// CHECK-SAME: %[[ARG1:.*]]: !mo.tensor<[[SH1]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[SH2]], f32> {
mo.graph @unfolded_apply_shouldnt_be_inlined<Sh0: !mo.shape, Sh1: !mo.shape -> Sh2: !mo.shape>(
  %arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>
) -> !mo.tensor<Sh2, f32> {
  // CHECK-NEXT: kgen.param.declare [[P1:.*]] = <apply(:(!mo.shape) -> index "mo.shape.get_rank", [[SH0]])>
  kgen.param.declare P1 = <apply(:(!mo.shape) -> index "mo.shape.get_rank", Sh0)>
  // CHECK-NEXT: kgen.param.declare [[P2:.*]] = <apply(:(!mo.shape) -> index "mo.shape.get_rank", [[SH1]])>
  kgen.param.declare P2 = <apply(:(!mo.shape) -> index "mo.shape.get_rank", Sh1)>
  // CHECK-NEXT: kgen.param.declare [[P3:.*]]: i1 = <apply(:(index, index) -> i1 "index.cmp"{pred = #index<cmp_predicate eq>}, [[P1]], [[P2]])>
  kgen.param.declare P3: i1 = <apply(:(index, index) -> i1 "index.cmp"{pred = #index<cmp_predicate eq>}, P1, P2)>
  // CHECK-NEXT: kgen.param.assert <[[P3:.*]]>
  kgen.param.assert <P3>, ""

  // CHECK-NEXT: kgen.param.declare [[P4:.*]]: !mo.shape = <apply(:(!mo.shape) -> !mo.shape "mo.shape.slice"{end = -2 : index, start = 0 : index}, [[SH0]])>
  kgen.param.declare P4: !mo.shape = <apply(:(!mo.shape) -> !mo.shape "mo.shape.slice"{end = -2 : index, start = 0 : index}, Sh0)>
  // CHECK-NEXT: %[[OUT:.*]] = mo.rebind(%[[ARG0]]) : !mo.tensor<[[SH0]], f32> -> !mo.tensor<[[P4]], f32>
  %1 = mo.rebind(%arg0) : !mo.tensor<Sh0, f32> -> !mo.tensor<P4, f32>

  // CHECK-NEXT: mo.output<:!mo.shape [[P4]]> %[[OUT]] : !mo.tensor<[[P4]], f32>
  mo.output<:!mo.shape P4> %1 : !mo.tensor<P4, f32>
}
