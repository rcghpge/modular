// RUN: mo-opt %s --symbolic-optimize -cse | FileCheck %s

// Standard case, run reduction on axis not affected by the bcast
mo.graph @reduce_sinking_success<A, B, P1, P2, P3>(
  %arg0: !mo.tensor<[A, B], f32>
) -> (!mo.tensor<[P1, P2, P3, 1, B], f32>, !mo.tensor<[P1, P2, P3, A, 1], f32>) {
  %axis3 = mo.constant {value = #M.dense_array<3> : tensor<si64>} : !mo.tensor<[], si64>
  %axis4 = mo.constant {value = #M.dense_array<4> : tensor<si64>} : !mo.tensor<[], si64>

  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, P2, P3, A, B], f32>
  %reduce_axis1 = mo.reduce.add(%broadcasted_arg0, %axis3) : (!mo.tensor<[P1, P2, P3, A, B], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[P1, P2, P3, 1, B], f32>
  %reduce_axis2 = mo.mean(%broadcasted_arg0, %axis4) : (!mo.tensor<[P1, P2, P3, A, B], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[P1, P2, P3, A, 1], f32>

  mo.output %reduce_axis1, %reduce_axis2: !mo.tensor<[P1, P2, P3, 1, B], f32>, !mo.tensor<[P1, P2, P3, A, 1], f32>
}
// CHECK: mo.graph @reduce_sinking_success<[[A:[A-Z0-9]*]], [[B:[A-Z0-9]*]], [[P1:[A-Z0-9]*]], [[P2:[A-Z0-9]*]], [[P3:[A-Z0-9]*]]>
// CHECK-SAME: ([[INPUT:%.*]]: !mo.tensor<[[[A]], [[B]]], f32>) -> (!mo.tensor<[[[P1]], [[P2]], [[P3]], 1, [[B]]], f32>, !mo.tensor<[[[P1]], [[P2]], [[P3]], [[A]], 1], f32>) no_inline {
// CHECK-DAG: [[V0:%.*]] = mo.constant {value = #M.dense_array<0> : !M.array<1xsi64>} : !mo.tensor<[], si64>
// CHECK-DAG: [[V1:%.*]] = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V2:%.*]] = mo.reduce.add([[INPUT]], [[V0]]) : (!mo.tensor<[[[A]], [[B]]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[1, [[B]]], f32>
// CHECK-NEXT: [[V3:%.*]] = mo.static.broadcast_to([[V2]]) : !mo.tensor<[1, [[B]]], f32> -> !mo.tensor<[[[P1]], [[P2]], [[P3]], 1, [[B]]], f32>
// CHECK-NEXT: [[V4:%.*]] = mo.mean([[INPUT]], [[V1]]) : (!mo.tensor<[[[A]], [[B]]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[A]], 1], f32>
// CHECK-NEXT: [[V5:%.*]] = mo.static.broadcast_to([[V4]]) : !mo.tensor<[[[A]], 1], f32> -> !mo.tensor<[[[P1]], [[P2]], [[P3]], [[A]], 1], f32>
// CHECK-NEXT: mo.output [[V3]], [[V5]] : !mo.tensor<[[[P1]], [[P2]], [[P3]], 1, [[B]]], f32>, !mo.tensor<[[[P1]], [[P2]], [[P3]], [[A]], 1], f32>


// Right now cannot do further canonicalizations on bcasted axes at this time
mo.graph @reduce_sinking_fail_on_new_bcasted_dim<A, B, P1, P2, P3>(
  %arg0: !mo.tensor<[A, B], f32>
) -> (!mo.tensor<[1, P2, P3, A, B], f32>, !mo.tensor<[P1, 1, P3, A, B], f32>) {
  %axis0 = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %axis1 = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>

  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, P2, P3, A, B], f32>
  %reduce_axis1 = mo.reduce.max(%broadcasted_arg0, %axis0) : (!mo.tensor<[P1, P2, P3, A, B], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[1, P2, P3, A, B], f32>
  %reduce_axis2 = mo.mean(%broadcasted_arg0, %axis1) : (!mo.tensor<[P1, P2, P3, A, B], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[P1, 1, P3, A, B], f32>

  mo.output %reduce_axis1, %reduce_axis2: !mo.tensor<[1, P2, P3, A, B], f32>, !mo.tensor<[P1, 1, P3, A, B], f32>
}
// CHECK: mo.graph @reduce_sinking_fail_on_new_bcasted_dim<[[A:[A-Z0-9]*]], [[B:[A-Z0-9]*]], [[P1:[A-Z0-9]*]], [[P2:[A-Z0-9]*]], [[P3:[A-Z0-9]*]]>
// CHECK-SAME: ([[INPUT:%.*]]: !mo.tensor<[[[A]], [[B]]], f32>) -> (!mo.tensor<[1, [[P2]], [[P3]], [[A]], [[B]]], f32>, !mo.tensor<[[[P1]], 1, [[P3]], [[A]], [[B]]], f32>) no_inline {
// CHECK-DAG: [[V0:%.*]] = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-DAG: [[V1:%.*]] = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V2:%.*]] = mo.static.broadcast_to([[INPUT]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[[[P1]], [[P2]], [[P3]], [[A]], [[B]]], f32>
// CHECK-NEXT: [[V3:%.*]] = mo.reduce.max([[V2]], [[V0]]) : (!mo.tensor<[[[P1]], [[P2]], [[P3]], [[A]], [[B]]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[1, [[P2]], [[P3]], [[A]], [[B]]], f32>
// CHECK-NEXT: [[V4:%.*]] = mo.mean([[V2]], [[V1]]) : (!mo.tensor<[[[P1]], [[P2]], [[P3]], [[A]], [[B]]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[P1]], 1, [[P3]], [[A]], [[B]]], f32>
// CHECK-NEXT: mo.output [[V3]], [[V4]] : !mo.tensor<[1, [[P2]], [[P3]], [[A]], [[B]]], f32>, !mo.tensor<[[[P1]], 1, [[P3]], [[A]], [[B]]], f32>

// We cannot do further sinking on 1-broadcasted axes at this time
mo.graph @reduce_sinking_fail_on_1_bcasted_dim<A, B, P1>(
  %arg0: !mo.tensor<[A, 1, B], f32>
) -> !mo.tensor<[A, 1, B], f32> {
  %axis1 = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, 1, B], f32> -> !mo.tensor<[A, P1, B], f32>
  %reduce_axis1 = mo.reduce.min(%broadcasted_arg0, %axis1) : (!mo.tensor<[A, P1, B], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[A, 1, B], f32>

  mo.output %reduce_axis1: !mo.tensor<[A, 1, B], f32>
}
// CHECK: mo.graph @reduce_sinking_fail_on_1_bcasted_dim<[[A:[A-Z0-9]*]], [[B:[A-Z0-9]*]], [[P1:[A-Z0-9]*]]>
// CHECK-SAME: ([[INPUT:%.*]]: !mo.tensor<[[[A]], 1, [[B]]], f32>) -> !mo.tensor<[[[A]], 1, [[B]]], f32> no_inline {
// CHECK-NEXT: [[V0:%.*]] = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:%.*]] = mo.static.broadcast_to([[INPUT]]) : !mo.tensor<[[[A]], 1, [[B]]], f32> -> !mo.tensor<[[[A]], [[P1]], [[B]]], f32>
// CHECK-NEXT: [[V2:%.*]] = mo.reduce.min([[V1]], [[V0]]) : (!mo.tensor<[[[A]], [[P1]], [[B]]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[A]], 1, [[B]]], f32>
// CHECK-NEXT: mo.output [[V2]] : !mo.tensor<[[[A]], 1, [[B]]], f32>

// Need to know fixed rank
mo.graph @reduce_sinking_fail_on_unknown_rank<Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape>(
  %arg0: !mo.tensor<Sh1, f32>
) -> !mo.tensor<Sh3, f32> {
  %axis_neg_one = mo.constant {value = #M.dense_array<-1> : tensor<si64>} : !mo.tensor<[], si64>
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<Sh1, f32> -> !mo.tensor<Sh2, f32>
  %reduce_axis1 = mo.reduce.min(%broadcasted_arg0, %axis_neg_one) : (!mo.tensor<Sh2, f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh3, f32>

  mo.output %reduce_axis1: !mo.tensor<Sh3, f32>
}
// CHECK: mo.graph @reduce_sinking_fail_on_unknown_rank
// CHECK-SAME: [[SH1:[A-Za-z0-9]*]]: !mosh.ape, [[SH2:[A-Za-z0-9]*]]: !mosh.ape, [[SH3:[A-Za-z0-9]*]]: !mosh.ape
// CHECK-SAME: ([[INPUT:%.*]]: !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH3]], f32> no_inline {
// CHECK-NEXT: [[V0:%.*]] = mo.constant {value = #M.dense_array<-1> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:%.*]] = mo.static.broadcast_to([[INPUT]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: [[V2:%.*]] = mo.reduce.min([[V1]], [[V0]]) : (!mo.tensor<[[SH2]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: mo.output [[V2]] : !mo.tensor<[[SH3]], f32>

//===----------------------------------------------------------------------===//
// Coverage on all reduction operators
//===----------------------------------------------------------------------===//

mo.graph @reduce_sinking_success_mean<A, B, P1>(
  %arg0: !mo.tensor<[A, B], f32>
) -> !mo.tensor<[P1, A, 1], f32> {
  %axis2 = mo.constant {value = #M.dense_array<2> : tensor<si64>} : !mo.tensor<[], si64>
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, A, B], f32>
  %reduce_axis1 = mo.mean(%broadcasted_arg0, %axis2) : (!mo.tensor<[P1, A, B], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[P1, A, 1], f32>
  mo.output %reduce_axis1: !mo.tensor<[P1, A, 1], f32>
}
// CHECK: mo.graph @reduce_sinking_success_mean<[[A:[A-Z]*]], [[B:[A-Z]*]]
// CHECK: mo.mean
// CHECK-SAME : (!mo.tensor<[[[A]], [[B]]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[A]], 1], f32>

mo.graph @reduce_sinking_success_reduce_mul<A, B, P1>(
  %arg0: !mo.tensor<[A, B], f32>
) -> !mo.tensor<[P1, A, 1], f32> {
  %axis2 = mo.constant {value = #M.dense_array<2> : tensor<si64>} : !mo.tensor<[], si64>
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, A, B], f32>
  %reduce_axis1 = mo.reduce.mul(%broadcasted_arg0, %axis2) : (!mo.tensor<[P1, A, B], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[P1, A, 1], f32>
  mo.output %reduce_axis1: !mo.tensor<[P1, A, 1], f32>
}
// CHECK: mo.graph @reduce_sinking_success_reduce_mul<[[A:[A-Z]*]], [[B:[A-Z]*]]
// CHECK: mo.reduce.mul
// CHECK-SAME : (!mo.tensor<[[[A]], [[B]]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[A]], 1], f32>

mo.graph @reduce_sinking_success_reduce_add<A, B, P1>(
  %arg0: !mo.tensor<[A, B], f32>
) -> !mo.tensor<[P1, A, 1], f32> {
  %axis2 = mo.constant {value = #M.dense_array<2> : tensor<si64>} : !mo.tensor<[], si64>
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, A, B], f32>
  %reduce_axis1 = mo.reduce.add(%broadcasted_arg0, %axis2) : (!mo.tensor<[P1, A, B], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[P1, A, 1], f32>
  mo.output %reduce_axis1: !mo.tensor<[P1, A, 1], f32>
}
// CHECK: mo.graph @reduce_sinking_success_reduce_add<[[A:[A-Z]*]], [[B:[A-Z]*]]
// CHECK: mo.reduce.add
// CHECK-SAME : (!mo.tensor<[[[A]], [[B]]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[A]], 1], f32>

mo.graph @reduce_sinking_success_reduce_min<A, B, P1>(
  %arg0: !mo.tensor<[A, B], f32>
) -> !mo.tensor<[P1, A, 1], f32> {
  %axis2 = mo.constant {value = #M.dense_array<2> : tensor<si64>} : !mo.tensor<[], si64>
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, A, B], f32>
  %reduce_axis1 = mo.reduce.min(%broadcasted_arg0, %axis2) : (!mo.tensor<[P1, A, B], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[P1, A, 1], f32>
  mo.output %reduce_axis1: !mo.tensor<[P1, A, 1], f32>
}
// CHECK: mo.graph @reduce_sinking_success_reduce_min<[[A:[A-Z]*]], [[B:[A-Z]*]]
// CHECK: mo.reduce.min
// CHECK-SAME : (!mo.tensor<[[[A]], [[B]]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[A]], 1], f32>

mo.graph @reduce_sinking_success_reduce_max<A, B, P1>(
  %arg0: !mo.tensor<[A, B], f32>
) -> !mo.tensor<[P1, A, 1], f32> {
  %axis2 = mo.constant {value = #M.dense_array<2> : tensor<si64>} : !mo.tensor<[], si64>
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, A, B], f32>
  %reduce_axis1 = mo.reduce.max(%broadcasted_arg0, %axis2) : (!mo.tensor<[P1, A, B], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[P1, A, 1], f32>
  mo.output %reduce_axis1: !mo.tensor<[P1, A, 1], f32>
}
// CHECK: mo.graph @reduce_sinking_success_reduce_max<[[A:[A-Z]*]], [[B:[A-Z]*]]
// CHECK: mo.reduce.max
// CHECK-SAME : (!mo.tensor<[[[A]], [[B]]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[A]], 1], f32>

mo.graph @reduce_sinking_success_arg_max<A, B, P1>(
  %arg0: !mo.tensor<[A, B], f32>
) -> !mo.tensor<[P1, A, 1], si64> {
  %axis2 = mo.constant {value = #M.dense_array<2> : tensor<si64>} : !mo.tensor<[], si64>
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, A, B], f32>
  %reduce_axis1 = mo.arg_max(%broadcasted_arg0, %axis2) : (!mo.tensor<[P1, A, B], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[P1, A, 1], si64>
  mo.output %reduce_axis1: !mo.tensor<[P1, A, 1], si64>
}
// CHECK: mo.graph @reduce_sinking_success_arg_max<[[A:[A-Z]*]], [[B:[A-Z]*]]
// CHECK: mo.arg_max
// CHECK-SAME : (!mo.tensor<[[[A]], [[B]]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[A]], 1], si64>

mo.graph @reduce_sinking_success_arg_min<A, B, P1>(
  %arg0: !mo.tensor<[A, B], f32>
) -> !mo.tensor<[P1, A, 1], si64> {
  %axis2 = mo.constant {value = #M.dense_array<2> : tensor<si64>} : !mo.tensor<[], si64>
  %broadcasted_arg0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[P1, A, B], f32>
  %reduce_axis1 = mo.arg_min(%broadcasted_arg0, %axis2) : (!mo.tensor<[P1, A, B], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[P1, A, 1], si64>
  mo.output %reduce_axis1: !mo.tensor<[P1, A, 1], si64>
}
// CHECK: mo.graph @reduce_sinking_success_arg_min<[[A:[A-Z]*]], [[B:[A-Z]*]]
// CHECK: mo.arg_min
// CHECK-SAME : (!mo.tensor<[[[A]], [[B]]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[A]], 1], si64>
