// RUN: mo-opt %s --shape-inference | FileCheck %s

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
// CHECK: kgen.func @broadcast_2_fw
// CHECK: kgen.return
// CHECK: }
// CHECK: kgen.func @transpose_perms_1_0_fw
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
  // CHECK: kgen.param.declare [[P12:.*]]: i1 =
  // CHECK-SAME: <eq(
  // CHECK-DAG: [[D2]]
  // CHECK-DAG: ,
  // CHECK-DAG: [[D4]]
  // CHECK-SAME: )>
  // CHECK: kgen.param.declare [[CH0:.*]]: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "inner dimensions do not match"}, [[P12]])>
  // CHECK: kgen.param.declare [[P15:.*]]: i1 = <apply(:(!mo.shape, !mo.shape) -> i1 "mo.shape.eq", [[[D0]]], [[[D3]]])>
  // CHECK: kgen.param.declare [[CH1:.*]]: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "batch sizes of left and right inputs are not equal"}, [[P15]])>
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
  // CHECK: %[[ARG1REB1:.*]] = mo.rebind(%[[ARG1]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH0]], f32>
  // CHECK: %[[VAL0:.*]] = mo.add(%[[ARG0]], %[[ARG1REB1]]) : !mo.tensor<[[SH0]], f32>
  %0 = mo.add(%arg0, %arg1) : !mo.tensor<?, f32>
  // CHECK: %[[VAL1:.*]] = mo.gelu(%[[VAL0]]) : !mo.tensor<[[SH0]], f32>
  %1 = mo.gelu(%0) : !mo.tensor<?, f32>

  // CHECK-NOT: kgen.param.declare {{.*}}: !mo.shape = <apply(:(!mo.shape, !mo.shape) -> !mo.shape @batch_matmul_fw,

  // CHECK-DAG: kgen.param.declare [[R1:.*]] = <apply(:(!mo.shape) -> index "mo.shape.get_rank", [[SH0]])>
  // CHECK-DAG: kgen.param.declare [[R2:.*]] = <apply(:(!mo.shape) -> index "mo.shape.get_rank", [[SH1]])>
  // CHECK-DAG: kgen.param.declare [[COND1:.*]]: i1 =
  // CHECK-SAME: <eq(
  // CHECK-DAG: [[R1]]
  // CHECK-DAG:,
  // CHECK-DAG: [[R2]]
  // CHECK-SAME:)>
  // CHECK-DAG: kgen.param.declare [[CH0:.*]]: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "ranks of left and right inputs are not equal"}, [[COND1]])>
  // CHECK-DAG: kgen.param.declare [[COND2:.*]]: i1 = <ge([[R2]], 3)>
  // CHECK-DAG: kgen.param.declare [[CH1:.*]]: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "input ranks are not greater than 2"}, [[COND2]])>
  // CHECK-DAG: kgen.param.declare [[K:.*]] = <apply(:(!mo.shape) -> index "mo.shape.get_dim"{dim = -1 : index}, [[SH0]])>
  // CHECK-DAG: kgen.param.declare [[KK:.*]] = <apply(:(!mo.shape) -> index "mo.shape.get_dim"{dim = -2 : index}, [[SH1]])>
  // CHECK-DAG: kgen.param.declare [[COND3:.*]]: i1 =
  // CHECK-SAME: <eq(
  // CHECK-DAG: [[K]]
  // CHECK-DAG:,
  // CHECK-DAG: [[KK]]
  // CHECK-SAME:)>
  // CHECK-DAG: kgen.param.declare [[CH2:.*]]: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "inner dimensions do not match"}, [[COND3]])>
  // CHECK-DAG: kgen.param.declare [[LBATCH:.*]]: !mo.shape = <apply(:(!mo.shape) -> !mo.shape "mo.shape.slice"{end = -2 : index, start = 0 : index}, [[SH0]])>
  // CHECK-DAG: kgen.param.declare [[RBATCH:.*]]: !mo.shape = <apply(:(!mo.shape) -> !mo.shape "mo.shape.slice"{end = -2 : index, start = 0 : index}, [[SH1]])>
  // CHECK-DAG: kgen.param.declare [[COND4:.*]]: i1 = <apply(:(!mo.shape, !mo.shape) -> i1 "mo.shape.eq", [[LBATCH]], [[RBATCH]])>
  // CHECK-DAG: kgen.param.declare [[CH3:.*]]: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "batch sizes of left and right inputs are not equal"}, [[COND4]])>
  // CHECK-DAG: kgen.param.declare [[NN:.*]] = <apply(:(!mo.shape) -> index "mo.shape.get_dim"{dim = -1 : index}, [[SH1]])>
  // CHECK-DAG: kgen.param.declare [[SH8:.*]]: !mo.shape = <apply(:(!mo.shape, !mo.shape) -> !mo.shape "mo.shape.concat", [[LBATCH]], [[[K]], [[NN]]])>
  // CHECK: %[[ARG1REB2:.*]] = mo.rebind(%[[ARG1REB1]]) : !mo.tensor<[[SH0]], f32> -> !mo.tensor<[[SH8]], f32>

  // CHECK-DAG: kgen.param.declare [[N:.*]] = <apply(:(!mo.shape) -> index "mo.shape.get_dim"{dim = -2 : index}, [[SH0]])>
  // CHECK-DAG: kgen.param.declare [[M:.*]] = <apply(:(!mo.shape) -> index "mo.shape.get_dim"{dim = -1 : index}, [[SH8]])>
  // CHECK-DAG: kgen.param.declare [[SH4:.*]]: !mo.shape = <apply(:(!mo.shape) -> !mo.shape "mo.shape.slice"{end = -2 : index, start = 0 : index}, [[SH0]])>
  // CHECK-DAG: kgen.param.declare [[SH5:.*]]: !mo.shape = <apply(:(!mo.shape, !mo.shape) -> !mo.shape "mo.shape.concat", [[SH4]], [[[N]], [[M]]])>

  // CHECK-NOT: kgen.param.declare {{.*}}: !mo.shape = <apply(:(!mo.shape, !mo.shape) -> !mo.shape @batch_matmul_fw,

  // CHECK: %[[VAL2:.*]] = mo.batch_matmul(%[[VAL1]], %[[ARG1REB2]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH8]], f32>) -> !mo.tensor<[[SH5]], f32>
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

// CHECK: mo.graph @broadcasted_binary_elwise<[[D15:.*]], [[D16:.*]], [[D17:.*]], [[D18:.*]] -> [[D20:.*]], [[D21:.*]]>(
mo.graph @broadcasted_binary_elwise(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[D15]], 12, [[D16]]], f32>,
  // CHECK-SAME: %[[ARG1:.*]]: !mo.tensor<[[[D17]], [[D18]]], f32>
  %arg0: !mo.tensor<[?, 12, ?], f32>, %arg1: !mo.tensor<[?, ?], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D15]], [[D20]], [[D21]]], f32> {
) -> (!mo.tensor<[?, ?, ?], f32>) {
  // CHECK-DAG: kgen.param.declare [[P4:.*]]: i1 = <eq([[D17]], 1)>
  // CHECK-DAG: kgen.param.declare [[P6:.*]]: i1 = <eq([[D17]], 12)>
  // CHECK-DAG: kgen.param.declare [[P7:.*]]: i1 =
  // CHECK-SAME: <or(
  // CHECK-DAG: [[P4]]
  // CHECK-DAG:,
  // CHECK-DAG: [[P6]]
  // CHECK-SAME:)>
  // CHECK-DAG: kgen.param.declare [[CH0:.*]]: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "dimensions are not broadcastable"}, [[P7]])>

  // CHECK-DAG: kgen.param.declare [[P8:.*]] = <max([[D17]], 12)>
  // CHECK-DAG: kgen.param.declare [[P11:.*]]: i1 = <eq([[D16]], 1)>
  // CHECK-DAG: kgen.param.declare [[P12:.*]]: i1 = <eq([[D18]], 1)>
  // CHECK-DAG: kgen.param.declare [[P13:.*]]: i1 =
  // CHECK-SAME: <or(
  // CHECK-DAG: [[P11]]
  // CHECK-DAG:,
  // CHECK-DAG: [[P12]]
  // CHECK-SAME:)>
  // CHECK-DAG: kgen.param.declare [[P14:.*]]: i1 =
  // CHECK-SAME: <eq(
  // CHECK-DAG: [[D16]]
  // CHECK-DAG:,
  // CHECK-DAG: [[D18]]
  // CHECK-SAME:)>
  // CHECK-DAG: kgen.param.declare [[P15:.*]]: i1 =
  // CHECK-SAME: <or(
  // CHECK-DAG: [[P13]]
  // CHECK-DAG:,
  // CHECK-DAG: [[P14]]
  // CHECK-SAME:)>
  // CHECK-DAG: kgen.param.declare [[CH1:.*]]: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "dimensions are not broadcastable"}, [[P15]])>

  // CHECK-DAG: kgen.param.declare [[P16:.*]] =
  // CHECK-SAME: <max(
  // CHECK-DAG: [[D16]]
  // CHECK-DAG:,
  // CHECK-DAG: [[D18]]
  // CHECK-SAME:)>
  %shape1 = mo.shape_of(%arg0) : (!mo.tensor<[?, 12, ?], f32>) -> !mo.tensor<[3], si64>
  %shape2 = mo.shape_of(%arg1) : (!mo.tensor<[?, ?], f32>) -> !mo.tensor<[2], si64>
  %shape = mo.broadcast_shape(%shape1, %shape2) : (!mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>

  // CHECK-DAG: %[[VAL0:.*]] = mo.static.broadcast_to(%arg0) : !mo.tensor<[[[D15]], 12, [[D16]]], f32> -> !mo.tensor<[[[D15]], [[P8]], [[P16]]], f32>
  // CHECK-DAG: %[[VAL1:.*]] = mo.static.broadcast_to(%arg1) : !mo.tensor<[[[D17]], [[D18]]], f32> -> !mo.tensor<[[[D15]], [[P8]], [[P16]]], f32>
  %0 = mo.broadcast_to(%arg0, %shape) : (!mo.tensor<[?, 12, ?], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[?, ?, ?], f32>
  %1 = mo.broadcast_to(%arg1, %shape) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[?, ?, ?], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.add(%[[VAL0]], %[[VAL1]]) : !mo.tensor<[[[D15]], [[P8]], [[P16]]], f32>
  %3 = mo.add(%0, %1) : !mo.tensor<[?, ?, ?], f32>
  // CHECK-NEXT: mo.output<[[P8]], [[P16]]> %[[RES]] : !mo.tensor<[[[D15]], [[P8]], [[P16]]], f32>
  mo.output %3 : !mo.tensor<[?, ?, ?], f32>
}

// CHECK: mo.graph @transpose_static_rank<[[D8:.*]], [[D9:.*]], [[D10:.*]], [[D11:.*]]>(
mo.graph @transpose_static_rank(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[D8]], [[D9]]], f32>,
  %input0: !mo.tensor<[?, ?], f32>,
  // CHECK-SAME: %[[ARG1:.*]]: !mo.tensor<[[[D10]], [[D11]]], f32>
  %input1: !mo.tensor<[?, ?], f32>
// CHECK-SAME: ) -> (!mo.tensor<[[[D8]], [[D9]]], f32>, !mo.tensor<[[[D11]], [[D10]]], f32>) {
) -> (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>) {
  // CHECK-DAG: %[[PERM0:.*]] = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[PERM1:.*]] = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %perm0 = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %perm1 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-NEXT: %[[OUT2:.*]] = mo.transpose(%[[ARG0]], %[[PERM0]])
  // CHECK-SAME: : (!mo.tensor<[[[D8]], [[D9]]], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[D8]], [[D9]]], f32>
  %output0 = mo.transpose(%input0, %perm0) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
  // CHECK-NEXT: %[[OUT3:.*]] = mo.transpose(%[[ARG1]], %[[PERM1]])
  // CHECK-SAME: : (!mo.tensor<[[[D10]], [[D11]]], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[D11]], [[D10]]], f32>
  %output1 = mo.transpose(%input1, %perm1) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
  // CHECK-NEXT: mo.output %[[OUT2]], %[[OUT3]] : !mo.tensor<[[[D8]], [[D9]]], f32>, !mo.tensor<[[[D11]], [[D10]]], f32>
  mo.output %output0, %output1 : !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>
// CHECK-NEXT: }
}
