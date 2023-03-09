// RUN: mo-opt %s --resolve-unknown-parameters --materialize-shape-funcs --inline-shape-funcs --propagate-shapes --canonicalize | FileCheck %s

// COM: For now shape functions remain in the module even after they are
// COM: inlined, but this will be removed eventually.
// CHECK: kgen.func @gather_axis_0_fw
// CHECK: kgen.return
// CHECK: }
// CHECK: kgen.func @"gather_axis_-1_fw"
// CHECK: kgen.return
// CHECK: }
// CHECK: kgen.func @reduction_axis_0_fw
// CHECK: kgen.return
// CHECK: }
// CHECK: kgen.func @reduction_axis_1_fw
// CHECK: kgen.return
// CHECK: }
// CHECK: kgen.func @"reduction_axis_-1_fw"
// CHECK: kgen.return
// CHECK: }
// CHECK: kgen.func @batch_matmul_fw
// CHECK: kgen.return
// CHECK: }
// CHECK: kgen.func @batch_matmul_side
// CHECK: kgen.return
// CHECK: }

// CHECK: mo.graph @static_ranks<[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>, %[[ARG1:.*]]: !mo.tensor<[[[D2]], [[D3]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]]], f32> {
mo.graph @static_ranks(%arg0: !mo.tensor<[?, ?], f32>, %arg1: !mo.tensor<[?, ?], f32>) -> !mo.tensor<?, f32> {
  // CHECK: %[[ARG1REB:.*]] = mo.rebind(%[[ARG1]]) : !mo.tensor<[[[D2]], [[D3]]], f32> -> !mo.tensor<[[[D0]], [[D1]]], f32>
  %arg0c = mo.rebind(%arg0) : !mo.tensor<[?, ?], f32> -> !mo.tensor<?, f32>
  %arg1c = mo.rebind(%arg1) : !mo.tensor<[?, ?], f32> -> !mo.tensor<?, f32>
  // CHECK: %[[VAL0:.*]] = mo.add(%[[ARG0]], %[[ARG1REB]]) : !mo.tensor<[[[D0]], [[D1]]], f32>
  %0 = mo.add(%arg0c, %arg1c) : !mo.tensor<?, f32>
  // CHECK: %[[VAL1:.*]] = mo.gelu(%[[VAL0]]) : !mo.tensor<[[[D0]], [[D1]]], f32>
  %1 = mo.gelu(%0) : !mo.tensor<?, f32>
  // CHECK: %[[VAL2:.*]] = mo.sub(%[[VAL1]], %[[VAL0]]) : !mo.tensor<[[[D0]], [[D1]]], f32>
  %2 = mo.sub(%1, %0) : !mo.tensor<?, f32>
  // CHECK: %[[VAL3:.*]] = mo.relu(%[[VAL2]]) : !mo.tensor<[[[D0]], [[D1]]], f32>
  %3 = mo.relu(%2) : !mo.tensor<?, f32>
  // CHECK: mo.output %[[VAL3]] : !mo.tensor<[[[D0]], [[D1]]], f32>
  mo.output %3 : !mo.tensor<?, f32>
}

// CHECK: mo.graph @batch_matmul_static_rank<[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]], [[D5:.*]]>(
mo.graph @batch_matmul_static_rank(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>,
  %arg0: !mo.tensor<[?, ?, ?], f32>,
  // CHECK-SAME: %[[ARG1:.*]]: !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
  %arg1: !mo.tensor<[?, ?, ?], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D5]]], f32> {
) -> !mo.tensor<[?, ?, ?], f32> {
  // CHECK: kgen.param.declare [[P12:.*]]: i1 = <apply(:(index, index) -> i1 "index.cmp"{pred = #index<cmp_predicate eq>}, [[D2]], [[D4]])>
  // CHECK: kgen.param.assert <[[P12]]>, "inner dimensions do not match"
  // CHECK: kgen.param.declare [[P15:.*]]: i1 = <apply(:(!mo.shape, !mo.shape) -> i1 "mo.shape.eq", [[[D0]]], [[[D3]]])>
  // CHECK: kgen.param.assert <[[P15]]>, "batch sizes of left and right inputs are not equal"
  // CHECK: %[[V0:.*]] = mo.rebind(%[[ARG1]]) : !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32> -> !mo.tensor<[[[D0]], [[D2]], [[D5]]], f32>
  // CHECK: %[[V1:.*]] = mo.batch_matmul(%[[ARG0]], %[[V0]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, !mo.tensor<[[[D0]], [[D2]], [[D5]]], f32>) -> !mo.tensor<[[[D0]], [[D1]], [[D5]]], f32>
  %out = mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?], f32>
  // CHECK: mo.output %[[V1]] : !mo.tensor<[[[D0]], [[D1]], [[D5]]], f32>
  mo.output %out : !mo.tensor<[?, ?, ?], f32>
}

// CHECK: mo.graph @materialize_gather_axis_0_si32_e2e<[[D0:.*]], [[D1:.*]]>(
mo.graph @materialize_gather_axis_0_si32_e2e(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], 2], f32>,
  %input: !mo.tensor<[?, 2], f32>,
// CHECK-SAME: %[[INDICES:.*]]: !mo.tensor<[2, [[D1]]], si32>
  %indices: !mo.tensor<[2, ?], si32>
// CHECK-SAME: ) -> !mo.tensor<[2, [[D1]], 2], f32> {
) -> !mo.tensor<[?, ?, ?], f32> {
  // CHECK-NEXT: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %axis = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>
  // CHECK-NEXT: %[[RES:.*]] = mo.gather(%[[INPUT]], %[[INDICES]], %[[AXIS]]) :
  // CHECK-SAME: (!mo.tensor<[[[D0]], 2], f32>, !mo.tensor<[2, [[D1]]], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[2, [[D1]], 2], f32>
  %result = mo.gather(%input, %indices, %axis) : (!mo.tensor<[?, 2], f32>, !mo.tensor<[2, ?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, [[D1]], 2], f32>
  mo.output %result : !mo.tensor<[?, ?, ?], f32>
}

// CHECK: mo.graph @materialize_gather_axis_neg_1_si32_e2e<[[D0:.*]], [[D1:.*]]>(
mo.graph @materialize_gather_axis_neg_1_si32_e2e(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], 2], f32>,
  %input: !mo.tensor<[?, 2], f32>,
// CHECK-SAME: %[[INDICES:.*]]: !mo.tensor<[2, [[D1]]], si32>
  %indices: !mo.tensor<[2, ?], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], 2, [[D1]]], f32> {
) -> !mo.tensor<[?, ?, ?], f32> {
  // CHECK-NEXT: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<-1> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %axis = mo.constant {value = #M.dense_array<-1> : tensor<1xsi32>} : !mo.tensor<[], si32>
  // CHECK-NEXT: %[[RES:.*]] = mo.gather(%[[INPUT]], %[[INDICES]], %[[AXIS]]) :
  // CHECK-SAME: (!mo.tensor<[[[D0]], 2], f32>, !mo.tensor<[2, [[D1]]], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[D0]], 2, [[D1]]], f32>
  %result = mo.gather(%input, %indices, %axis) : (!mo.tensor<[?, 2], f32>, !mo.tensor<[2, ?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D0]], 2, [[D1]]], f32>
  mo.output %result : !mo.tensor<[?, ?, ?], f32>
}

// CHECK: mo.graph @materialize_mean_axis_0_si32_e2e<[[D0:.*]]>(
mo.graph @materialize_mean_axis_0_si32_e2e(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], 2], f32>
  %input: !mo.tensor<[?, 2], f32>
// CHECK-SAME: ) -> !mo.tensor<[1, 2], f32> {
) -> !mo.tensor<?, f32> {
  // CHECK-NEXT: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %axis = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>
  // CHECK-NEXT: %[[RES:.*]] = mo.mean(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[[[D0]], 2], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[1, 2], f32>
  %result = mo.mean(%input, %axis) : (!mo.tensor<[?, 2], f32>, !mo.tensor<[], si32>) -> !mo.tensor<?, f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[1, 2], f32>
  mo.output %result : !mo.tensor<?, f32>
}

// CHECK: mo.graph @materialize_prod_axis_1_si64_e2e<[[D0:.*]]>(
mo.graph @materialize_prod_axis_1_si64_e2e(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], 2], f32>
  %input: !mo.tensor<[?, 2], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], 1], f32> {
) -> !mo.tensor<?, f32> {
  // CHECK-NEXT: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.reduce.mul(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[[[D0]], 2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D0]], 1], f32>
  %result = mo.reduce.mul(%input, %axis) : (!mo.tensor<[?, 2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D0]], 1], f32>
  mo.output %result : !mo.tensor<?, f32>
}

// CHECK: mo.graph @materialize_sum_axis_neg_1_si64_e2e<[[D0:.*]]>(
mo.graph @materialize_sum_axis_neg_1_si64_e2e(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], 2], f32>
  %input: !mo.tensor<[?, 2], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], 1], f32> {
) -> !mo.tensor<?, f32> {
  // CHECK-NEXT: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.reduce.add(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[[[D0]], 2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D0]], 1], f32>
  %result = mo.reduce.add(%input, %axis) : (!mo.tensor<[?, 2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D0]], 1], f32>
  mo.output %result : !mo.tensor<?, f32>
}

// CHECK: mo.graph @unresolved_rebinds<[[SH0:.*]]: !mo.shape, [[SH1:.*]]: !mo.shape -> [[SH2:.*]]: !mo.shape>(
mo.graph @unresolved_rebinds(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>, %[[ARG1:.*]]: !mo.tensor<[[SH1]], f32>
  %arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>
// CHECK-SAME: ) -> !mo.tensor<[[SH2]], f32> {
) -> !mo.tensor<?, f32> {
  // CHECK: %[[ARG1REB:.*]] = mo.rebind(%[[ARG1]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH0]], f32>
  // CHECK: %[[VAL0:.*]] = mo.add(%[[ARG0]], %[[ARG1REB]]) : !mo.tensor<[[SH0]], f32>
  %0 = mo.add(%arg0, %arg1) : !mo.tensor<?, f32>
  // CHECK: %[[VAL1:.*]] = mo.gelu(%[[VAL0]]) : !mo.tensor<[[SH0]], f32>
  %1 = mo.gelu(%0) : !mo.tensor<?, f32>

  // CHECK-NOT: kgen.param.declare {{.*}}: !mo.shape = <apply(:(!mo.shape, !mo.shape) -> !mo.shape @batch_matmul_fw,

  // CHECK-DAG: kgen.param.declare [[R1:.*]] = <apply(:(!mo.shape) -> index "mo.shape.get_rank", [[SH0]])>
  // CHECK-DAG: kgen.param.declare [[R2:.*]] = <apply(:(!mo.shape) -> index "mo.shape.get_rank", [[SH1]])>
  // CHECK-DAG: kgen.param.declare [[COND1:.*]]: i1 = <apply(:(index, index) -> i1 "index.cmp"{pred = #index<cmp_predicate eq>}, [[R1]], [[R2]])>
  // CHECK-DAG: kgen.param.assert <[[COND1]]>, "ranks of left and right inputs are not equal"
  // CHECK-DAG: kgen.param.declare [[COND2:.*]]: i1 = <apply(:(index, index) -> i1 "index.cmp"{pred = #index<cmp_predicate sgt>}, [[R2]], 2)>
  // CHECK-DAG: kgen.param.assert <[[COND2]]>, "input ranks are not greater than 2"
  // CHECK-DAG: kgen.param.declare [[K:.*]] = <apply(:(!mo.shape) -> index "mo.shape.get_dim"{dim = -1 : index}, [[SH0]])>
  // CHECK-DAG: kgen.param.declare [[KK:.*]] = <apply(:(!mo.shape) -> index "mo.shape.get_dim"{dim = -2 : index}, [[SH1]])>
  // CHECK-DAG: kgen.param.declare [[COND3:.*]]: i1 = <apply(:(index, index) -> i1 "index.cmp"{pred = #index<cmp_predicate eq>}, [[K]], [[KK]])>
  // CHECK-DAG: kgen.param.assert <[[COND3]]>, "inner dimensions do not match"
  // CHECK-DAG: kgen.param.declare [[LBATCH:.*]]: !mo.shape = <apply(:(!mo.shape) -> !mo.shape "mo.shape.slice"{end = -2 : index, start = 0 : index}, [[SH0]])>
  // CHECK-DAG: kgen.param.declare [[RBATCH:.*]]: !mo.shape = <apply(:(!mo.shape) -> !mo.shape "mo.shape.slice"{end = -2 : index, start = 0 : index}, [[SH1]])>
  // CHECK-DAG: kgen.param.declare [[COND4:.*]]: i1 = <apply(:(!mo.shape, !mo.shape) -> i1 "mo.shape.eq", [[LBATCH]], [[RBATCH]])>
  // CHECK-DAG: kgen.param.assert <[[COND4]]>, "batch sizes of left and right inputs are not equal"
  // CHECK-DAG: kgen.param.declare [[NN:.*]] = <apply(:(!mo.shape) -> index "mo.shape.get_dim"{dim = -1 : index}, [[SH1]])>
  // CHECK-DAG: kgen.param.declare [[SH8:.*]]: !mo.shape = <apply(:(!mo.shape, !mo.shape) -> !mo.shape "mo.shape.concat", [[LBATCH]], [[[K]], [[NN]]])>
  // CHECK: %[[ARG1REB:.*]] = mo.rebind(%[[ARG1]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH8]], f32>

  // CHECK-DAG: kgen.param.declare [[N:.*]] = <apply(:(!mo.shape) -> index "mo.shape.get_dim"{dim = -2 : index}, [[SH0]])>
  // CHECK-DAG: kgen.param.declare [[M:.*]] = <apply(:(!mo.shape) -> index "mo.shape.get_dim"{dim = -1 : index}, [[SH8]])>
  // CHECK-DAG: kgen.param.declare [[SH4:.*]]: !mo.shape = <apply(:(!mo.shape) -> !mo.shape "mo.shape.slice"{end = -2 : index, start = 0 : index}, [[SH0]])>
  // CHECK-DAG: kgen.param.declare [[SH5:.*]]: !mo.shape = <apply(:(!mo.shape, !mo.shape) -> !mo.shape "mo.shape.concat", [[SH4]], [[[N]], [[M]]])>

  // CHECK-NOT: kgen.param.declare {{.*}}: !mo.shape = <apply(:(!mo.shape, !mo.shape) -> !mo.shape @batch_matmul_fw,

  // CHECK: %[[VAL2:.*]] = mo.batch_matmul(%[[VAL1]], %[[ARG1REB]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH8]], f32>) -> !mo.tensor<[[SH5]], f32>
  %2 = mo.batch_matmul(%1, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  // CHECK: %[[VAL3:.*]] = mo.relu(%[[VAL2]]) : !mo.tensor<[[SH5]], f32>
  %3 = mo.relu(%2) : !mo.tensor<?, f32>
  // CHECK: %[[VAL4:.*]] = mo.gelu(%[[VAL3]]) : !mo.tensor<[[SH5]], f32>
  %4 = mo.gelu(%3) : !mo.tensor<?, f32>
  // CHECK: %[[VAL5:.*]] = mo.sub(%[[VAL4]], %[[VAL3]]) : !mo.tensor<[[SH5]], f32>
  %5 = mo.sub(%4, %3) : !mo.tensor<?, f32>
  // CHECK: mo.output<:!mo.shape [[SH5]]> %[[VAL5]] : !mo.tensor<[[SH5]], f32>
  mo.output %5 : !mo.tensor<?, f32>
}
