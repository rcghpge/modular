// RUN: mo-opt %s --canonicalize --shape-inference | FileCheck %s

// CHECK-NOT: kgen.func

// CHECK: mo.graph @static_ranks<[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>, %[[ARG1:.*]]: !mo.tensor<[[[D2]], [[D3]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]]], f32> no_inline {
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
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D5]]], f32> no_inline {
) -> !mo.tensor<[?, ?, ?], f32> {
  // CHECK: kgen.param.declare [[P12:.*]]: i1 =
  // CHECK-SAME: <eq(
  // CHECK-DAG: [[D2]]
  // CHECK-DAG: ,
  // CHECK-DAG: [[D4]]
  // CHECK-SAME: )>
  // CHECK: kgen.param.declare [[CH0:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "inner dimensions do not match" : !kgen.string}, [[P12]])>
  // CHECK: kgen.param.declare [[P15:.*]]: i1 =
  // CHECK-SAME: <eq(
  // CHECK-DAG: [[D0]]
  // CHECK-DAG: ,
  // CHECK-DAG: [[D3]]
  // CHECK-SAME: )>
  // CHECK: kgen.param.declare [[CH1:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "batch sizes of left and right inputs are not equal" : !kgen.string}, [[P15]])>
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
// CHECK-SAME: ) -> !mo.tensor<[2, [[D1]], 2], f32> no_inline {
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
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], 2, [[D1]]], f32> no_inline {
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
// CHECK-SAME: ) -> !mo.tensor<[1, 2], f32> no_inline {
) -> !mo.tensor<[?, ?], f32> {
  // CHECK-NEXT: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %axis = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>
  // CHECK-NEXT: %[[RES:.*]] = mo.mean(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[[[D0]], 2], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[1, 2], f32>
  %result = mo.mean(%input, %axis) : (!mo.tensor<[?, 2], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[1, 2], f32>
  mo.output %result : !mo.tensor<[?, ?], f32>
}

// CHECK: mo.graph @materialize_prod_axis_1_si64_e2e<[[D0:.*]]>(
mo.graph @materialize_prod_axis_1_si64_e2e(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], 2], f32>
  %input: !mo.tensor<[?, 2], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], 1], f32> no_inline {
) -> !mo.tensor<[?, ?], f32> {
  // CHECK-NEXT: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.reduce.mul(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[[[D0]], 2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D0]], 1], f32>
  %result = mo.reduce.mul(%input, %axis) : (!mo.tensor<[?, 2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D0]], 1], f32>
  mo.output %result : !mo.tensor<[?, ?], f32>
}

// CHECK: mo.graph @materialize_sum_axis_neg_1_si64_e2e<[[D0:.*]]>(
mo.graph @materialize_sum_axis_neg_1_si64_e2e(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], 2], f32>
  %input: !mo.tensor<[?, 2], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], 1], f32> no_inline {
) -> !mo.tensor<[?, ?], f32> {
  // CHECK-NEXT: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.reduce.add(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[[[D0]], 2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D0]], 1], f32>
  %result = mo.reduce.add(%input, %axis) : (!mo.tensor<[?, 2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D0]], 1], f32>
  mo.output %result : !mo.tensor<[?, ?], f32>
}

// CHECK: mo.graph @unresolved_rebinds<[[SH0:.*]]: !mosh.ape, [[SH1:.*]]: !mosh.ape -> [[SH2:.*]]: !mosh.ape>(
mo.graph @unresolved_rebinds(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>, %[[ARG1:.*]]: !mo.tensor<[[SH1]], f32>
  %arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>
// CHECK-SAME: ) -> !mo.tensor<[[SH2]], f32> no_inline {
) -> !mo.tensor<?, f32> {
  // CHECK: %[[ARG1REB1:.*]] = mo.rebind(%[[ARG1]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH0]], f32>
  // CHECK: kgen.param.declare [[EQ:.*]]: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [[SH0]], [[SH1]])>
  // CHECK: kgen.param.declare [[CH:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "input shapes must match" : !kgen.string}, [[EQ]])>
  // CHECK: %[[VAL0:.*]] = mo.add(%[[ARG0]], %[[ARG1REB1]]) : !mo.tensor<[[SH0]], f32>
  %0 = mo.add(%arg0, %arg1) : !mo.tensor<?, f32>
  // CHECK: %[[VAL1:.*]] = mo.gelu(%[[VAL0]]) : !mo.tensor<[[SH0]], f32>
  %1 = mo.gelu(%0) : !mo.tensor<?, f32>

  // CHECK-NOT: kgen.param.declare {{.*}}: !mosh.ape = <apply(:(!mosh.ape, !mosh.ape) -> !mosh.ape @matmul_like_fw,

  // CHECK-DAG: kgen.param.declare [[R1:.*]] = <apply(:(!mosh.ape) -> index "mosh.get_rank", [[SH0]])>
  // CHECK-DAG: kgen.param.declare [[R2:.*]] = <apply(:(!mosh.ape) -> index "mosh.get_rank", [[SH1]])>
  // CHECK-DAG: kgen.param.declare [[COND1:.*]]: i1 =
  // CHECK-SAME: <eq(
  // CHECK-DAG: [[R1]]
  // CHECK-DAG:,
  // CHECK-DAG: [[R2]]
  // CHECK-SAME:)>
  // CHECK-DAG: kgen.param.declare [[CH0:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "ranks of left and right inputs are not equal" : !kgen.string}, [[COND1]])>
  // CHECK-DAG: kgen.param.declare [[COND2:.*]]: i1 = <ge([[R2]], 3)>
  // CHECK-DAG: kgen.param.declare [[CH1:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "input ranks are not greater than minimum rank for matmul-like" : !kgen.string}, [[COND2]])>
  // CHECK-DAG: kgen.param.declare [[K:.*]] = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", [[SH0]], -1)>
  // CHECK-DAG: kgen.param.declare [[KK:.*]] = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", [[SH1]], -2)>
  // CHECK-DAG: kgen.param.declare [[COND3:.*]]: i1 =
  // CHECK-SAME: <eq(
  // CHECK-DAG: [[K]]
  // CHECK-DAG:,
  // CHECK-DAG: [[KK]]
  // CHECK-SAME:)>
  // CHECK-DAG: kgen.param.declare [[CH2:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "inner dimensions do not match" : !kgen.string}, [[COND3]])>
  // CHECK-DAG: kgen.param.declare [[LBATCH:.*]]: !mosh.ape = <apply(:(!mosh.ape, index, index) -> !mosh.ape "mosh.slice", [[SH0]], 0, -2)>
  // CHECK-DAG: kgen.param.declare [[RBATCH:.*]]: !mosh.ape = <apply(:(!mosh.ape, index, index) -> !mosh.ape "mosh.slice", [[SH1]], 0, -2)>
  // CHECK-DAG: kgen.param.declare [[COND4:.*]]: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [[LBATCH]], [[RBATCH]])>
  // CHECK-DAG: kgen.param.declare [[CH3:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "batch sizes of left and right inputs are not equal" : !kgen.string}, [[COND4]])>
  // CHECK-DAG: kgen.param.declare [[NN:.*]] = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", [[SH1]], -1)>
  // CHECK-DAG: kgen.param.declare [[SH8:.*]]: !mosh.ape = <apply(:(!mosh.ape, !mosh.ape) -> !mosh.ape "mosh.concat", [[LBATCH]], [[[K]], [[NN]]])>
  // CHECK-DAG: %[[ARG1REB2:.*]] = mo.rebind(%[[ARG1]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH8]], f32>

  // CHECK-DAG: kgen.param.declare [[N:.*]] = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", [[SH0]], -2)>
  // CHECK-DAG: kgen.param.declare [[M:.*]] = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", [[SH8]], -1)>
  // CHECK-DAG: kgen.param.declare [[SH4:.*]]: !mosh.ape = <apply(:(!mosh.ape, index, index) -> !mosh.ape "mosh.slice", [[SH0]], 0, -2)>
  // CHECK-DAG: kgen.param.declare [[SH5:.*]]: !mosh.ape = <apply(:(!mosh.ape, !mosh.ape) -> !mosh.ape "mosh.concat", [[SH4]], [[[N]], [[M]]])>

  // CHECK-NOT: kgen.param.declare {{.*}}: !mosh.ape = <apply(:(!mosh.ape, !mosh.ape) -> !mosh.ape @matmul_like_fw,

  // CHECK: %[[VAL2:.*]] = mo.batch_matmul(%[[VAL1]], %[[ARG1REB2]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH8]], f32>) -> !mo.tensor<[[SH5]], f32>
  %2 = mo.batch_matmul(%1, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  // CHECK: %[[VAL3:.*]] = mo.relu(%[[VAL2]]) : !mo.tensor<[[SH5]], f32>
  %3 = mo.relu(%2) : !mo.tensor<?, f32>
  // CHECK: %[[VAL4:.*]] = mo.gelu(%[[VAL3]]) : !mo.tensor<[[SH5]], f32>
  %4 = mo.gelu(%3) : !mo.tensor<?, f32>
  // CHECK: %[[VAL5:.*]] = mo.sub(%[[VAL4]], %[[VAL3]]) : !mo.tensor<[[SH5]], f32>
  %5 = mo.sub(%4, %3) : !mo.tensor<?, f32>
  // CHECK: mo.output<:!mosh.ape [[SH5]]> %[[VAL5]] : !mo.tensor<[[SH5]], f32>
  mo.output %5 : !mo.tensor<?, f32>
}

// CHECK: mo.graph @broadcasted_binary_elwise<[[D15:.*]], [[D16:.*]], [[D17:.*]], [[D18:.*]] -> [[D20:.*]], [[D21:.*]]>(
mo.graph @broadcasted_binary_elwise(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[D15]], 12, [[D16]]], f32>,
  // CHECK-SAME: %[[ARG1:.*]]: !mo.tensor<[[[D17]], [[D18]]], f32>
  %arg0: !mo.tensor<[?, 12, ?], f32>, %arg1: !mo.tensor<[?, ?], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D15]], [[D20]], [[D21]]], f32> no_inline {
) -> (!mo.tensor<[?, ?, ?], f32>) {
  // CHECK-DAG: kgen.param.declare [[P4:.*]]: i1 = <eq([[D17]], 1)>
  // CHECK-DAG: kgen.param.declare [[P6:.*]]: i1 = <eq([[D17]], 12)>
  // CHECK-DAG: kgen.param.declare [[P7:.*]]: i1 =
  // CHECK-SAME: <or(
  // CHECK-DAG: [[P4]]
  // CHECK-DAG:,
  // CHECK-DAG: [[P6]]
  // CHECK-SAME:)>
  // CHECK-DAG: kgen.param.declare [[CH0:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions are not broadcastable" : !kgen.string}, [[P7]])>

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
  // CHECK-DAG: kgen.param.declare [[CH1:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions are not broadcastable" : !kgen.string}, [[P15]])>

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
// CHECK-SAME: ) -> (!mo.tensor<[[[D8]], [[D9]]], f32>, !mo.tensor<[[[D11]], [[D10]]], f32>) no_inline {
) -> (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>) {
  // CHECK-NEXT: %[[PERM1:.*]] = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %perm0 = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %perm1 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %output0 = mo.transpose(%input0, %perm0) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
  // CHECK-NEXT: %[[OUT1:.*]] = mo.transpose(%[[ARG1]], %[[PERM1]])
  // CHECK-SAME: : (!mo.tensor<[[[D10]], [[D11]]], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[D11]], [[D10]]], f32>
  %output1 = mo.transpose(%input1, %perm1) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
  // CHECK-NEXT: mo.output %[[ARG0]], %[[OUT1]] : !mo.tensor<[[[D8]], [[D9]]], f32>, !mo.tensor<[[[D11]], [[D10]]], f32>
  mo.output %output0, %output1 : !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>
// CHECK-NEXT: }
}

// CHECK: mo.graph @scalar_to_tensor_reshapes_before_staticization<[[D0:.*]], [[D1:.*]]>(
mo.graph @scalar_to_tensor_reshapes_before_staticization(
  // CHECK-SAME: %[[ARG:.*]]: !mo.tensor<[[[D0]], [[D1]]], si32>
  %arg: !mo.tensor<[?, ?], si32>
  // CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D0]]], si32> no_inline {
) -> !mo.tensor<[?, ?], si32> {
  // CHECK-DAG: kgen.param.declare [[P0:.*]] = <mul_nuw([[D0]], [[D1]])>
  // CHECK-DAG: kgen.param.declare [[P1:.*]] = <mul_nuw([[D0]], [[D0]])>
  // CHECK-DAG: kgen.param.declare [[P2:.*]]: i1 = <eq([[P0]], [[P1]])>
  // CHECK-DAG: kgen.param.declare [[P3:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "number of elements must be equal" : !kgen.string}, [[P2]])>
  %scalar_one = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %scalar_zero = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si32>
  %tensor_zero = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %tensor_one = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  // CHECK-NEXT: %[[RES:.*]] = mo.rebind(%[[ARG]])
  // CHECK-SAME: : !mo.tensor<[[[D0]], [[D1]]], si32> -> !mo.tensor<[[[D0]], [[D0]]], si32>
  %arg_shape = mo.shape_of(%arg) : (!mo.tensor<[?, ?], si32>) -> !mo.tensor<[2], si32>
  %arg_dim0 = mo.slice(%arg_shape, %tensor_zero, %tensor_one, %tensor_one) : (!mo.tensor<[2], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[?], si32>
  %empty_tensor = mo.range(%scalar_zero, %scalar_zero, %scalar_one) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %one_ = mo.gather(%tensor_one, %empty_tensor, %scalar_zero) : (!mo.tensor<[1], si32>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %one_one = mo.concat[%scalar_zero : !mo.tensor<[], si32>] (%one_, %one_) : (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<[?], si64>
  %arg_dim0_rs1 = mo.reshape(%arg_dim0, %one_one) : (!mo.tensor<[?], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<[], si32>
  %arg_dim0_rs2 = mo.reshape(%arg_dim0_rs1, %tensor_one) : (!mo.tensor<[], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[?], si32>
  %arg_dim0_dim0 = mo.concat[%scalar_zero : !mo.tensor<[], si32>] (%arg_dim0_rs2, %arg_dim0_rs2)
  : (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<[2], si32>
  %res = mo.reshape(%arg, %arg_dim0_dim0) : (!mo.tensor<[?, ?], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[?, ?], si32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D0]], [[D0]]], si32>
  mo.output %res : !mo.tensor<[?, ?], si32>
}

// CHECK: mo.graph @binary_op_asserts<[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(
mo.graph @binary_op_asserts(
  // CHECK-SAME: %[[ARG1:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>
  %arg1: !mo.tensor<[?, ?], f32>,
  // CHECK-SAME: %[[ARG2:.*]]: !mo.tensor<[[[D2]], [[D3]]], f32>
  %arg2: !mo.tensor<[?, ?], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]]], f32> no_inline {
) -> (!mo.tensor<[?, ?], f32>) {
  // CHECK-NEXT: %[[ADD11:.*]] = mo.add(%[[ARG1]], %[[ARG1]]) : !mo.tensor<[[[D0]], [[D1]]], f32>
  // CHECK-NEXT: %[[ARG2_REB:.*]] = mo.rebind(%[[ARG2]]) : !mo.tensor<[[[D2]], [[D3]]], f32> -> !mo.tensor<[[[D0]], [[D1]]], f32>
  // CHECK-DAG: kgen.param.declare [[EQ:.*]]: i1 =
  // CHECK-SAME: <and(
  // CHECK-DAG: eq(
  // CHECK-DAG: [[D2]]
  // CHECK-DAG:,
  // CHECK-DAG: [[D0]]
  // CHECK-DAG: ),
  // CHECK-DAG: eq(
  // CHECK-DAG: [[D3]]
  // CHECK-DAG:,
  // CHECK-DAG: [[D1]]
  // CHECK-DAG: )
  // CHECK-SAME:)>
  // CHECK-DAG: kgen.param.declare [[CH:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "input shapes must match" : !kgen.string}, [[EQ]])>
  %add11 = mo.add(%arg1, %arg1) : !mo.tensor<[?, ?], f32>
  // CHECK-NEXT: %[[ADD12:.*]] = mo.add(%[[ARG1]], %[[ARG2_REB]]) : !mo.tensor<[[[D0]], [[D1]]], f32>
  %add12 = mo.add(%arg1, %arg2) : !mo.tensor<[?, ?], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.add(%[[ADD11]], %[[ADD12]]) : !mo.tensor<[[[D0]], [[D1]]], f32>
  %res = mo.add(%add11, %add12) : !mo.tensor<[?, ?], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D0]], [[D1]]], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// CHECK: mo.graph @binary_op_no_assert<[[D0:.*]], [[D1:.*]]>(
mo.graph @binary_op_no_assert(
  // CHECK-SAME: %[[ARG1:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>
  %arg1: !mo.tensor<[?, ?], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]]], f32> no_inline {
) -> (!mo.tensor<[?, ?], f32>) {
  // CHECK-NEXT: %[[ADD11:.*]] = mo.add(%[[ARG1]], %[[ARG1]]) : !mo.tensor<[[[D0]], [[D1]]], f32>
  %add11 = mo.add(%arg1, %arg1) : !mo.tensor<[?, ?], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.add(%[[ARG1]], %[[ADD11]]) : !mo.tensor<[[[D0]], [[D1]]], f32>
  %res = mo.add(%arg1, %add11) : !mo.tensor<[?, ?], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D0]], [[D1]]], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// CHECK-LABEL: mo.graph @batch_matmul_reused_rhs
// CHECK-SAME: <[[D0:.*]], [[D1:.*]]>(
mo.graph @batch_matmul_reused_rhs<D0, D1 -> Sh0: !mosh.ape>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[3, [[D0]], 4], f32>,
  %arg0: !mo.tensor<[3, D0, 4], f32>,
  // CHECK-SAME: %[[ARG1:.*]]: !mo.tensor<[3, 4, [[D1]]], f32>
  %arg1: !mo.tensor<[3, 4, D1], f32>
// CHECK-SAME: ) -> !mo.tensor<[3, [[D0]], [[D1]]], f32> no_inline {
) -> !mo.tensor<Sh0, f32> {

  %arg0reb = mo.rebind<() -> Sh1: !mosh.ape>(%arg0) : !mo.tensor<[3, D0, 4], f32> -> !mo.tensor<Sh1, f32>
  %arg1reb = mo.rebind<() -> Sh2: !mosh.ape>(%arg1) : !mo.tensor<[3, 4, D1], f32> -> !mo.tensor<Sh2, f32>

  // CHECK-NEXT: %[[BMM1:.*]] = mo.batch_matmul(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: : (!mo.tensor<[3, [[D0]], 4], f32>, !mo.tensor<[3, 4, [[D1]]], f32>) -> !mo.tensor<[3, [[D0]], [[D1]]], f32>
  // CHECK-NEXT: kgen.param.declare [[EQ:.*]]: i1 =
  // CHECK-SAME: <eq(
  // CHECK-DAG: [[D1]]
  // CHECK-DAG:,
  // CHECK-DAG: 4
  // CHECK-SAME:)>
  // CHECK-NEXT: kgen.param.declare [[CH:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "inner dimensions do not match" : !kgen.string}, [[EQ]])>
  %bmm1 = mo.batch_matmul<() -> Sh3: !mosh.ape>(%arg0reb, %arg1reb) : (!mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, f32>

  // CHECK-NEXT: %[[ARG1REB:.*]] = mo.rebind(%[[ARG1:.*]]) : !mo.tensor<[3, 4, [[D1]]], f32> -> !mo.tensor<[3, [[D1]], [[D1]]], f32>
  // CHECK-NEXT: %[[BMM2:.*]] = mo.batch_matmul(%[[BMM1]], %[[ARG1REB]])
  // CHECK-SAME: (!mo.tensor<[3, [[D0]], [[D1]]], f32>, !mo.tensor<[3, [[D1]], [[D1]]], f32>) -> !mo.tensor<[3, [[D0]], [[D1]]], f32>
  %bmm2 = mo.batch_matmul<() -> Sh4: !mosh.ape>(%bmm1, %arg1reb) : (!mo.tensor<Sh3, f32>, !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh4, f32>

  // CHECK-NEXT: mo.output %[[BMM2]] : !mo.tensor<[3, [[D0]], [[D1]]], f32>
  mo.output<:!mosh.ape Sh4> %bmm2 : !mo.tensor<Sh4, f32>
// CHECK-NEXT: }
}

// CHECK-LABEL: mo.graph @linalg_solve_reused_rhs
// CHECK-SAME: <[[D0:.*]], [[D1:.*]]>(
mo.graph @linalg_solve_reused_rhs<D0, D1 -> Sh0: !mosh.ape>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[3, [[D0]], [[D0]]], f32>,
  %arg0: !mo.tensor<[3, D0, D0], f32>,
  // CHECK-SAME: %[[ARG1:.*]]: !mo.tensor<[3, [[D0]], [[D1]]], f32>
  %arg1: !mo.tensor<[3, D0, D1], f32>
// CHECK-SAME: ) -> !mo.tensor<[3, [[D0]], [[D1]]], f32> no_inline {
) -> !mo.tensor<Sh0, f32> {

  %arg0reb = mo.rebind<() -> Sh1: !mosh.ape>(%arg0) : !mo.tensor<[3, D0, D0], f32> -> !mo.tensor<Sh1, f32>
  %arg1reb = mo.rebind<() -> Sh2: !mosh.ape>(%arg1) : !mo.tensor<[3, D0, D1], f32> -> !mo.tensor<Sh2, f32>

  // CHECK-NEXT: %[[VAL0:.*]] = mo.linalg.solve(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: : (!mo.tensor<[3, [[D0]], [[D0]]], f32>, !mo.tensor<[3, [[D0]], [[D1]]], f32>) -> !mo.tensor<[3, [[D0]], [[D1]]], f32>
  // CHECK-NEXT: kgen.param.declare [[EQ:.*]]: i1 =
  // CHECK-SAME: <eq(
  // CHECK-DAG: [[D0]]
  // CHECK-DAG:,
  // CHECK-DAG: [[D1]]
  // CHECK-SAME:)>
  // CHECK-NEXT: kgen.param.declare [[CH:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "inner dimensions do not match" : !kgen.string}, [[EQ]])>
  %0 = mo.linalg.solve<() -> Sh3: !mosh.ape>(%arg0reb, %arg1reb) : (!mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, f32>

  // CHECK-NEXT: %[[ARG1REB:.*]] = mo.rebind(%[[ARG1:.*]]) : !mo.tensor<[3, [[D0]], [[D1]]], f32> -> !mo.tensor<[3, [[D1]], [[D1]]], f32>
  // CHECK-NEXT: %[[VAL1:.*]] = mo.linalg.solve(%[[VAL0]], %[[ARG1REB]])
  // CHECK-SAME: (!mo.tensor<[3, [[D0]], [[D1]]], f32>, !mo.tensor<[3, [[D1]], [[D1]]], f32>) -> !mo.tensor<[3, [[D0]], [[D1]]], f32>
  %1 = mo.linalg.solve<() -> Sh4: !mosh.ape>(%0, %arg1reb) : (!mo.tensor<Sh3, f32>, !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh4, f32>

  // CHECK-NEXT: mo.output %[[VAL1]] : !mo.tensor<[3, [[D0]], [[D1]]], f32>
  mo.output<:!mosh.ape Sh4> %1 : !mo.tensor<Sh4, f32>
// CHECK-NEXT: }
}

// CHECK: mo.graph @slice_neg_stop(
mo.graph @slice_neg_stop(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[10], f32>
  %input: !mo.tensor<[10], f32>
// CHECK-SAME: ) -> !mo.tensor<[3], f32> no_inline {
) -> !mo.tensor<[?], f32> {
  // CHECK-NEXT: %[[START:.*]] = mo.constant {value = #M.dense_array<3> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  // CHECK-NEXT: %[[STOP:.*]] = mo.constant {value = #M.dense_array<-1> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  // CHECK-NEXT: %[[STEP:.*]] = mo.constant {value = #M.dense_array<2> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %start = mo.constant {value = #M.dense_array<3> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %stop = mo.constant {value = #M.dense_array<-1> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %step = mo.constant {value = #M.dense_array<2> : tensor<1xsi32>} : !mo.tensor<[1], si32>

  // X[3:-1:2] --> [x3, x5, x7]
  // CHECK-NEXT: %[[RES:.*]] = mo.slice(%[[INPUT]], %[[START]], %[[STOP]], %[[STEP]]) :
  // CHECK-SAME: (!mo.tensor<[10], f32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[3], f32>
  %result = mo.slice(%input, %start, %stop, %step)
  : (!mo.tensor<[10], f32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[?], f32>

  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[3], f32>
  mo.output %result : !mo.tensor<[?], f32>
}

// CHECK: mo.graph @slice_clamped_start_stop_neg_step(
mo.graph @slice_clamped_start_stop_neg_step(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[10], f32>
  %input: !mo.tensor<[10], f32>
// CHECK-SAME: ) -> !mo.tensor<[5], f32> no_inline {
) -> !mo.tensor<[?], f32> {
  // CHECK-NEXT: %[[START:.*]] = mo.constant {value = #M.dense_array<30> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  // CHECK-NEXT: %[[STOP:.*]] = mo.constant {value = #M.dense_array<-20> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  // CHECK-NEXT: %[[STEP:.*]] = mo.constant {value = #M.dense_array<-2> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %start = mo.constant {value = #M.dense_array<30> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %stop = mo.constant {value = #M.dense_array<-20> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %step = mo.constant {value = #M.dense_array<-2> : tensor<1xsi32>} : !mo.tensor<[1], si32>

  // X[30:-20:-2] --> [x9, x7, x5, x3, x1]
  // CHECK-NEXT: %[[RES:.*]] = mo.slice(%[[INPUT]], %[[START]], %[[STOP]], %[[STEP]]) :
  // CHECK-SAME: (!mo.tensor<[10], f32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[5], f32>
  %result = mo.slice(%input, %start, %stop, %step)
  : (!mo.tensor<[10], f32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[?], f32>

  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[5], f32>
  mo.output %result : !mo.tensor<[?], f32>
}

// CHECK: mo.graph @slice_4d(
mo.graph @slice_4d(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[10, 20, 30, 40], f32>
  %input: !mo.tensor<[10, 20, 30, 40], f32>
// CHECK-SAME: ) -> !mo.tensor<[3, 3, 8, 8], f32> no_inline {
) -> !mo.tensor<[?, ?, ?, ?], f32> {
  // CHECK-NEXT: %[[START:.*]] = mo.constant {value = #M.dense_array<3, -2, -100, 200> : tensor<4xsi32>} : !mo.tensor<[4], si32>
  // CHECK-NEXT: %[[STOP:.*]] = mo.constant {value = #M.dense_array<-1, 10, 200, -100> : tensor<4xsi32>} : !mo.tensor<[4], si32>
  // CHECK-NEXT: %[[STEP:.*]] = mo.constant {value = #M.dense_array<2, -3, 4, -5> : tensor<4xsi32>} : !mo.tensor<[4], si32>
  %start = mo.constant {value = #M.dense_array<3, -2, -100, 200> : tensor<4xsi32>} : !mo.tensor<[4], si32>
  %stop = mo.constant {value = #M.dense_array<-1, 10, 200, -100> : tensor<4xsi32>} : !mo.tensor<[4], si32>
  %step = mo.constant {value = #M.dense_array<2, -3, 4, -5> : tensor<4xsi32>} : !mo.tensor<[4], si32>

  // CHECK-NEXT: %[[RES:.*]] = mo.slice(%[[INPUT]], %[[START]], %[[STOP]], %[[STEP]]) :
  // CHECK-SAME: (!mo.tensor<[10, 20, 30, 40], f32>, !mo.tensor<[4], si32>, !mo.tensor<[4], si32>, !mo.tensor<[4], si32>) -> !mo.tensor<[3, 3, 8, 8], f32>
  %result = mo.slice(%input, %start, %stop, %step)
  : (!mo.tensor<[10, 20, 30, 40], f32>, !mo.tensor<[4], si32>, !mo.tensor<[4], si32>, !mo.tensor<[4], si32>) -> !mo.tensor<[?, ?, ?, ?], f32>

  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[3, 3, 8, 8], f32>
  mo.output %result : !mo.tensor<[?, ?, ?, ?], f32>
}

// CHECK: mo.graph @scatter_nd(
mo.graph @scatter_nd(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[10, 1, 1], f32>,
  %input: !mo.tensor<[10, 1, 1], f32>,
  // CHECK-SAME: %[[UPDATES:.*]]: !mo.tensor<[5, 1, 1], f32>
  %updates: !mo.tensor<[5, 1, 1], f32>,
  // CHECK-SAME: %[[INDICES_1D:.*]]: !mo.tensor<[5], si32>
  %indices_1d: !mo.tensor<[5], si32>
// CHECK-SAME: ) -> !mo.tensor<[10, 1, 1], f32> no_inline {
) -> !mo.tensor<[?, ?, ?], f32> {

  // CHECK-NEXT: %[[INDICES:.*]] = mo.static.reshape(%[[INDICES_1D]]) : !mo.tensor<[5], si32> -> !mo.tensor<[5, 1], si32>
  %shape = mo.constant {value = #M.dense_array<5, 1> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  %indices = mo.reshape(%indices_1d, %shape) : (!mo.tensor<[5], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[?, ?], si32>

  // CHECK-NEXT: %[[RES:.*]] = mo.scatter_nd(%[[INPUT]], %[[UPDATES]], %[[INDICES]])
  // CHECK-SAME: : (!mo.tensor<[10, 1, 1], f32>, !mo.tensor<[5, 1, 1], f32>, !mo.tensor<[5, 1], si32>) -> !mo.tensor<[10, 1, 1], f32>
  %res = mo.scatter_nd(%input, %updates, %indices)
  : (!mo.tensor<[10, 1, 1], f32>, !mo.tensor<[5, 1, 1], f32>, !mo.tensor<[?, ?], si32>) -> !mo.tensor<[?, ?, ?], f32>

  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[10, 1, 1], f32>
  mo.output %res : !mo.tensor<[?, ?, ?], f32>
}

// CHECK: mo.graph @scatter_nd_add(
mo.graph @scatter_nd_add(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[10, 1, 1], f32>,
  %input: !mo.tensor<[10, 1, 1], f32>,
  // CHECK-SAME: %[[UPDATES:.*]]: !mo.tensor<[5, 1, 1], f32>
  %updates: !mo.tensor<[5, 1, 1], f32>,
  // CHECK-SAME: %[[INDICES_1D:.*]]: !mo.tensor<[5], si32>
  %indices_1d: !mo.tensor<[5], si32>
// CHECK-SAME: ) -> !mo.tensor<[10, 1, 1], f32> no_inline {
) -> !mo.tensor<[?, ?, ?], f32> {

  // CHECK-NEXT: %[[INDICES:.*]] = mo.static.reshape(%[[INDICES_1D]]) : !mo.tensor<[5], si32> -> !mo.tensor<[5, 1], si32>
  %shape = mo.constant {value = #M.dense_array<5, 1> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  %indices = mo.reshape(%indices_1d, %shape) : (!mo.tensor<[5], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[?, ?], si32>

  // CHECK-NEXT: %[[RES:.*]] = mo.scatter_nd.add(%[[INPUT]], %[[UPDATES]], %[[INDICES]])
  // CHECK-SAME: : (!mo.tensor<[10, 1, 1], f32>, !mo.tensor<[5, 1, 1], f32>, !mo.tensor<[5, 1], si32>) -> !mo.tensor<[10, 1, 1], f32>
  %res = mo.scatter_nd.add(%input, %updates, %indices)
  : (!mo.tensor<[10, 1, 1], f32>, !mo.tensor<[5, 1, 1], f32>, !mo.tensor<[?, ?], si32>) -> !mo.tensor<[?, ?, ?], f32>

  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[10, 1, 1], f32>
  mo.output %res : !mo.tensor<[?, ?, ?], f32>
}


// CHECK: mo.graph @conv2d_static_input_dims(
mo.graph @conv2d_static_input_dims(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[10, 5, 5, 16], f32>
  %input: !mo.tensor<[10, 5, 5, 16], f32>,
  // CHECK-SAME: %[[FILTER:.*]]: !mo.tensor<[3, 3, 16, 32], f32>
  %filter: !mo.tensor<[3, 3, 16, 32], f32>
// CHECK-SAME: ) -> !mo.tensor<[10, 2, 2, 32], f32> no_inline {
) -> !mo.tensor<[?, ?, ?, ?], f32> {

  // CHECK-DAG: %[[ST:.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[DI:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[PA:.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %st = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>

  // CHECK-NEXT: %[[RES:.*]] = mo.conv(%[[INPUT]], %[[FILTER]]) [strides = %[[ST]], dilations = %[[DI]], paddings = %[[PA]]] : (
  // CHECK-SAME:   !mo.tensor<[10, 5, 5, 16], f32>,
  // CHECK-SAME:   !mo.tensor<[3, 3, 16, 32], f32>,
  // CHECK-SAME:   !mo.tensor<[2], si64>,
  // CHECK-SAME:   !mo.tensor<[2], si64>,
  // CHECK-SAME:   !mo.tensor<[4], si64>
  // CHECK-SAME: ) -> !mo.tensor<[10, 2, 2, 32], f32>
  %0 = mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa] : (
    !mo.tensor<[10, 5, 5, 16], f32>,
    !mo.tensor<[3, 3, 16, 32], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>

  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[10, 2, 2, 32], f32>
  mo.output %0 : !mo.tensor<[?, ?, ?, ?], f32>
}

// CHECK: mo.graph @conv2d_dyn_input_dims<[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(
mo.graph @conv2d_dyn_input_dims(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], 20, 30, [[D1]]], f32>
  %input: !mo.tensor<[?, 20, 30, ?], f32>,
  // CHECK-SAME: %[[FILTER:.*]]: !mo.tensor<[3, 3, [[D2]], [[D3]]], f32>
  %filter: !mo.tensor<[3, 3, ?, ?], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], 7, 12, [[D3]]], f32> no_inline {
) -> !mo.tensor<[?, ?, ?, ?], f32> {

  // CHECK-DAG: %[[ST:.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[DI:.*]] = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[PA:.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %st = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>

  // CHECK-DAG: kgen.param.declare [[EQ_BOOL:.*]]: i1 = <eq([[D1]], [[D2]])>
  // CHECK-DAG: kgen.param.declare [[CH:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "Input and filter channels must match" : !kgen.string}, [[EQ_BOOL]])>

  // CHECK-NEXT: %[[RES:.*]] = mo.conv(%[[INPUT]], %[[FILTER]]) [strides = %[[ST]], dilations = %[[DI]], paddings = %[[PA]]] : (
  // CHECK-SAME:   !mo.tensor<[[[D0]], 20, 30, [[D1]]], f32>,
  // CHECK-SAME:   !mo.tensor<[3, 3, [[D2]], [[D3]]], f32>,
  // CHECK-SAME:   !mo.tensor<[2], si64>,
  // CHECK-SAME:   !mo.tensor<[2], si64>,
  // CHECK-SAME:   !mo.tensor<[4], si64>
  // CHECK-SAME: ) -> !mo.tensor<[[[D0]], 7, 12, [[D3]]], f32>
  %0 = mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa] : (
    !mo.tensor<[?, 20, 30, ?], f32>,
    !mo.tensor<[3, 3, ?, ?], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>

  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D0]], 7, 12, [[D3]]], f32>
  mo.output %0 : !mo.tensor<[?, ?, ?, ?], f32>
}

// CHECK: mo.graph @max_pool2d_static_input_dims(
mo.graph @max_pool2d_static_input_dims(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[10, 5, 5, 16], f32>
  %input: !mo.tensor<[10, 5, 5, 16], f32>
// CHECK-SAME: ) -> !mo.tensor<[10, 2, 2, 16], f32> no_inline {
) -> !mo.tensor<[?, ?, ?, ?], f32> {

  // CHECK-DAG: %[[FS:.*]] = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[ST:.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[DI:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[PA:.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %fs = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %st = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>

  // CHECK-NEXT: %[[RES:.*]] = mo.max_pool(%[[INPUT]]) [filter_shape = %[[FS]], strides = %[[ST]], dilations = %[[DI]], paddings = %[[PA]]] : (
  // CHECK-SAME:   !mo.tensor<[10, 5, 5, 16], f32>,
  // CHECK-SAME:   !mo.tensor<[2], si64>,
  // CHECK-SAME:   !mo.tensor<[2], si64>,
  // CHECK-SAME:   !mo.tensor<[2], si64>
  // CHECK-SAME: ) -> !mo.tensor<[10, 2, 2, 16], f32>
  %0 = mo.max_pool(%input) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (
    !mo.tensor<[10, 5, 5, 16], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>

  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[10, 2, 2, 16], f32>
  mo.output %0 : !mo.tensor<[?, ?, ?, ?], f32>
}

// CHECK: mo.graph @avg_pool2d_dyn_input_dims<[[D0:.*]], [[D1:.*]]>(
mo.graph @avg_pool2d_dyn_input_dims(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], 20, 30, [[D1]]], f32>
  %input: !mo.tensor<[?, 20, 30, ?], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], 9, 14, [[D1]]], f32> no_inline {
) -> !mo.tensor<[?, ?, ?, ?], f32> {

  // CHECK-DAG: %[[FS:.*]] = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[ST:.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[DI:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[PA:.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %fs = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %st = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>

  // CHECK-NEXT: %[[RES:.*]] = mo.avg_pool(%[[INPUT]]) [filter_shape = %[[FS]], strides = %[[ST]], dilations = %[[DI]], paddings = %[[PA]]] : (
  // CHECK-SAME:   !mo.tensor<[[[D0]], 20, 30, [[D1]]], f32>,
  // CHECK-SAME:   !mo.tensor<[2], si64>,
  // CHECK-SAME:   !mo.tensor<[2], si64>,
  // CHECK-SAME:   !mo.tensor<[2], si64>
  // CHECK-SAME: ) -> !mo.tensor<[[[D0]], 9, 14, [[D1]]], f32>
  %0 = mo.avg_pool(%input) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (
    !mo.tensor<[?, 20, 30, ?], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>

  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D0]], 9, 14, [[D1]]], f32>
  mo.output %0 : !mo.tensor<[?, ?, ?, ?], f32>
}

// CHECK: mo.graph @arg_max<[[D1:.*]], [[D2:.*]]>(
// CHECK-SAME: %[[input:.*]]: !mo.tensor<[[[D1]], [[D2]]], si64>
// CHECK-SAME: ) -> !mo.tensor<[[[D1]], 1], si64> no_inline {
mo.graph @arg_max(%input: !mo.tensor<[?, ?], si64>) -> !mo.tensor<[?, ?], si64> {

  // CHECK-NEXT: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>

  // CHECK-NEXT: %[[RES:.*]] = mo.arg_max(%[[INPUT]], %[[AXIS]]) :
  // CHECK-SAME: (!mo.tensor<[[[D1]], [[D2]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], 1], si64>
  %res = mo.arg_max(%input, %axis) :
    (!mo.tensor<[?, ?], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?], si64>

  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D1]], 1], si64>
  mo.output %res : !mo.tensor<[?, ?], si64>
}

// CHECK:      mo.graph @concat<[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]], [[D5:.*]]>(
// CHECK-SAME:   %[[INPUT1:.*]]: !mo.tensor<[[[D0]], 1, [[D1]]], f32>,
// CHECK-SAME:   %[[INPUT2:.*]]: !mo.tensor<[[[D2]], 2, [[D3]]], f32>,
// CHECK-SAME:   %[[INPUT3:.*]]: !mo.tensor<[[[D4]], 3, [[D5]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], 6, [[D1]]], f32> no_inline {
mo.graph @concat(
  %input1: !mo.tensor<[?, 1, ?], f32>,
  %input2: !mo.tensor<[?, 2, ?], f32>,
  %input3: !mo.tensor<[?, 3, ?], f32>
) -> !mo.tensor<[?, ?, ?], f32> {

  // CHECK-NEXT: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>

  // CHECK-NEXT: kgen.param.declare [[B0:.*]]: i1 = <eq([[D0]], [[D2]])>
  // CHECK-NEXT: kgen.param.declare [[C0:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions before axis must be equal" : !kgen.string}, [[B0]])>
  // CHECK-NEXT: kgen.param.declare [[B1:.*]]: i1 = <eq([[D1]], [[D3]])>
  // CHECK-NEXT: kgen.param.declare [[C1:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions after axis must be equal" : !kgen.string}, [[B1]])>
  // CHECK-NEXT: kgen.param.declare [[B2:.*]]: i1 = <eq([[D0]], [[D4]])>
  // CHECK-NEXT: kgen.param.declare [[C2:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions before axis must be equal" : !kgen.string}, [[B2]])>
  // CHECK-NEXT: kgen.param.declare [[B3:.*]]: i1 = <eq([[D1]], [[D5]])>
  // CHECK-NEXT: kgen.param.declare [[C3:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions after axis must be equal" : !kgen.string}, [[B3]])>

  // CHECK-NEXT: %[[RES:.*]] = mo.concat[%[[AXIS:.*]] : !mo.tensor<[], si64>] (%[[INPUT1]], %[[INPUT2]], %[[INPUT3]]) :
  // CHECK-SAME:   (!mo.tensor<[[[D0]], 1, [[D1]]], f32>, !mo.tensor<[[[D2]], 2, [[D3]]], f32>, !mo.tensor<[[[D4]], 3, [[D5]]], f32>)
  // CHECK-SAME:   -> !mo.tensor<[[[D0]], 6, [[D1]]], f32>
  %res = mo.concat[%axis : !mo.tensor<[], si64>](%input1, %input2, %input3) :
    (!mo.tensor<[?, 1, ?], f32>, !mo.tensor<[?, 2, ?], f32>, !mo.tensor<[?, 3, ?], f32>)
    -> !mo.tensor<[?, ?, ?], f32>

    // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D0]], 6, [[D1]]], f32>
  mo.output %res : !mo.tensor<[?, ?, ?], f32>
}

// CHECK:      mo.graph @concat_neg_axis<[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]], [[D5:.*]]>(
// CHECK-SAME:   %[[INPUT1:.*]]: !mo.tensor<[[[D0]], 1, [[D1]]], f32>,
// CHECK-SAME:   %[[INPUT2:.*]]: !mo.tensor<[[[D2]], 2, [[D3]]], f32>,
// CHECK-SAME:   %[[INPUT3:.*]]: !mo.tensor<[[[D4]], 3, [[D5]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], 6, [[D1]]], f32> no_inline {
mo.graph @concat_neg_axis(
  %input1: !mo.tensor<[?, 1, ?], f32>,
  %input2: !mo.tensor<[?, 2, ?], f32>,
  %input3: !mo.tensor<[?, 3, ?], f32>
) -> !mo.tensor<[?, ?, ?], f32> {

  // CHECK-NEXT: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<-2> : tensor<si64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<-2> : tensor<si64>} : !mo.tensor<[], si64>

  // CHECK-NEXT: kgen.param.declare [[B0:.*]]: i1 = <eq([[D0]], [[D2]])>
  // CHECK-NEXT: kgen.param.declare [[C0:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions before axis must be equal" : !kgen.string}, [[B0]])>
  // CHECK-NEXT: kgen.param.declare [[B1:.*]]: i1 = <eq([[D1]], [[D3]])>
  // CHECK-NEXT: kgen.param.declare [[C1:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions after axis must be equal" : !kgen.string}, [[B1]])>
  // CHECK-NEXT: kgen.param.declare [[B2:.*]]: i1 = <eq([[D0]], [[D4]])>
  // CHECK-NEXT: kgen.param.declare [[C2:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions before axis must be equal" : !kgen.string}, [[B2]])>
  // CHECK-NEXT: kgen.param.declare [[B3:.*]]: i1 = <eq([[D1]], [[D5]])>
  // CHECK-NEXT: kgen.param.declare [[C3:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions after axis must be equal" : !kgen.string}, [[B3]])>

  // CHECK-NEXT: %[[RES:.*]] = mo.concat[%[[AXIS:.*]] : !mo.tensor<[], si64>] (%[[INPUT1]], %[[INPUT2]], %[[INPUT3]]) :
  // CHECK-SAME:   (!mo.tensor<[[[D0]], 1, [[D1]]], f32>, !mo.tensor<[[[D2]], 2, [[D3]]], f32>, !mo.tensor<[[[D4]], 3, [[D5]]], f32>)
  // CHECK-SAME:   -> !mo.tensor<[[[D0]], 6, [[D1]]], f32>
  %res = mo.concat[%axis : !mo.tensor<[], si64>](%input1, %input2, %input3) :
    (!mo.tensor<[?, 1, ?], f32>, !mo.tensor<[?, 2, ?], f32>, !mo.tensor<[?, 3, ?], f32>)
    -> !mo.tensor<[?, ?, ?], f32>

    // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D0]], 6, [[D1]]], f32>
  mo.output %res : !mo.tensor<[?, ?, ?], f32>
}

// CHECK:      mo.graph @pad(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[2, 3], f32>,
// CHECK-SAME:   %[[CONSTANT:.*]]: !mo.tensor<[], f32>
// CHECK-SAME: ) -> !mo.tensor<[32, 73], f32> no_inline {
mo.graph @pad(
  %input: !mo.tensor<[2, 3], f32>,
  %constant: !mo.tensor<[], f32>
) -> !mo.tensor<[?, ?], f32> {

  // CHECK-NEXT: %[[PADDINGS:.*]] = mo.constant {value = #M.dense_array<10, 20, 30, 40> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>
  %paddings = mo.constant {value = #M.dense_array<10, 20, 30, 40> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>

  // CHECK-NEXT: %[[RES:.*]] = mo.pad(%[[INPUT]], %[[PADDINGS]], %[[CONSTANT]]) : (
  // CHECK-SAME:   !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 2], si32>, !mo.tensor<[], f32>
  // CHECK-SAME: ) -> !mo.tensor<[32, 73], f32>
  %res = mo.pad(%input, %paddings, %constant) : (
    !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 2], si32>, !mo.tensor<[], f32>
  ) -> !mo.tensor<[?, ?], f32>

  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[32, 73], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// CHECK: mo.graph @tile(%[[INPUT:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[20, 60], f32> no_inline {
mo.graph @tile(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[?, ?], f32> {

  // CHECK-NEXT: %[[REPEATS:.*]] = mo.constant {value = #M.dense_array<10, 20> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  %repeats = mo.constant {value = #M.dense_array<10, 20> : tensor<2xsi32>} : !mo.tensor<[2], si32>

  // CHECK-NEXT: %[[RES:.*]] = mo.tile(%[[INPUT]], %[[REPEATS]]) : (
  // CHECK-SAME:   !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si32>
  // CHECK-SAME: ) -> !mo.tensor<[20, 60], f32>
  %res = mo.tile(%input, %repeats) : (
    !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si32>
  ) -> !mo.tensor<[?, ?], f32>

  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[20, 60], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// CHECK:      mo.graph @partially_parameterized_signature<[[IN:.*]], [[D0:.*]], [[D1:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[IN]], [[D0]], [[D1]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[IN]], 1, [[D1]]], f32> no_inline {
mo.graph @partially_parameterized_signature<in>(
  %input: !mo.tensor<[in, ?, ?], f32>
) -> !mo.tensor<[?, ?, ?], f32> {

  // CHECK-NEXT: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>

  // CHECK-NEXT: %[[RES:.*]]  = mo.reduce.mul(%[[INPUT]], %[[AXIS]])
  // CHECK-SAME:   : (!mo.tensor<[[[IN]], [[D0]], [[D1]]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[IN]], 1, [[D1]]], f32>
  %res  = mo.reduce.mul(%input, %axis)
    : (!mo.tensor<[in, ?, ?], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, ?], f32>

  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[IN]], 1, [[D1]]], f32>
  mo.output %res : !mo.tensor<[?, ?, ?], f32>
}

// CHECK:      mo.graph @binary_logical_op<[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(
// CHECK-SAME:   %[[LHS:.*]]: !mo.tensor<[[[D0]], [[D1]]], bool>,
// CHECK-SAME:   %[[RHS:.*]]: !mo.tensor<[[[D2]], [[D3]]], bool>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]]], bool>
mo.graph @binary_logical_op(
  %lhs: !mo.tensor<[?, ?], bool>,
  %rhs: !mo.tensor<[?, ?], bool>
) -> !mo.tensor<[?, ?], bool> {

  // CHECK-NEXT: %[[RHS_REB:.*]] = mo.rebind(%[[RHS]]) : !mo.tensor<[[[D2]], [[D3]]], bool> -> !mo.tensor<[[[D0]], [[D1]]], bool>

  // CHECK-NEXT: kgen.param.declare [[EQ0:.*]]: i1 = <and(eq([[D3]], [[D1]]), eq([[D0]], [[D2]]))>
  // CHECK-NEXT: kgen.param.declare [[CH0:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "input shapes must match" : !kgen.string}, [[EQ0]])>
  // CHECK-NEXT: %[[AND:.*]] = mo.and(%[[LHS]], %[[RHS_REB]]) : !mo.tensor<[[[D0]], [[D1]]], bool>
  %and = mo.and(%lhs, %rhs) : !mo.tensor<[?, ?], bool>

  // CHECK-NEXT: kgen.param.declare [[EQ1:.*]]: i1 = <and(eq([[D3]], [[D1]]), eq([[D0]], [[D2]]))>
  // CHECK-NEXT: kgen.param.declare [[CH1:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "input shapes must match" : !kgen.string}, [[EQ1]])>
  // CHECK-NEXT: %[[XOR:.*]] = mo.xor(%[[AND]], %[[RHS_REB]]) : !mo.tensor<[[[D0]], [[D1]]], bool>
  %xor = mo.xor(%and, %rhs) : !mo.tensor<[?, ?], bool>

  // CHECK-NEXT: mo.output %[[XOR]] : !mo.tensor<[[[D0]], [[D1]]], bool>
  mo.output %xor : !mo.tensor<[?, ?], bool>
}

// CHECK-LABEL: mo.graph @top_k
// CHECK-NEXT:   %[[IN:.*]] = mo.constant
// CHECK-NEXT:   %[[K:.*]] = mo.constant
// CHECK-NEXT:   %[[AXIS:.*]] = mo.constant
// CHECK-NEXT:   %[[VALUES:.*]], %[[INDICES:.*]] = mo.top_k(%[[IN]], %[[K]], %[[AXIS]]) : (!mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>)
// CHECK-NEXT:   mo.output %[[VALUES]], %[[INDICES]] : !mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>
// CHECK-NEXT: }
mo.graph @top_k(%input : !mo.tensor<[5, 5], si64>) -> (!mo.tensor<[?, ?], si64>, !mo.tensor<[?, ?], si64>) {
  %in = "mo.constant"() {value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24> : tensor<5x5xsi64>} : () -> !mo.tensor<[5, 5], si64>
  %K = "mo.constant"() {value = #M.dense_array<3> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %axis = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %values, %indices = mo.top_k(%in, %K, %axis) : (
    !mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> (!mo.tensor<[?, ?], si64>, !mo.tensor<[?, ?], si64>)
  mo.output %values, %indices : !mo.tensor<[?, ?], si64>, !mo.tensor<[?, ?], si64>
}

// CHECK-LABEL: mo.graph @bottom_k
// CHECK-NEXT:   %[[IN:.*]] = mo.constant
// CHECK-NEXT:   %[[K:.*]] = mo.constant
// CHECK-NEXT:   %[[AXIS:.*]] = mo.constant
// CHECK-NEXT:   %[[VALUES:.*]], %[[INDICES:.*]] = mo.bottom_k(%[[IN]], %[[K]], %[[AXIS]]) : (!mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>)
// CHECK-NEXT:   mo.output %[[VALUES]], %[[INDICES]] : !mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>
// CHECK-NEXT: }
mo.graph @bottom_k(%input : !mo.tensor<[5, 5], si64>) -> (!mo.tensor<[?, ?], si64>, !mo.tensor<[?, ?], si64>) {
  %in = "mo.constant"() {value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24> : tensor<5x5xsi64>} : () -> !mo.tensor<[5, 5], si64>
  %K = "mo.constant"() {value = #M.dense_array<3> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %axis = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %values, %indices = mo.bottom_k(%in, %K, %axis) : (
    !mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> (!mo.tensor<[?, ?], si64>, !mo.tensor<[?, ?], si64>)
  mo.output %values, %indices : !mo.tensor<[?, ?], si64>, !mo.tensor<[?, ?], si64>
}
