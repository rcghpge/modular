// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK: kgen.generator @concat_fold<[[SH1:.*]]: !mo.shape, [[D1:.*]], [[D2:.*]]>(
kgen.generator @concat_fold<Sh1: !mo.shape, D1, D2>(
// CHECK-SAME: ) -> (!mo.shape, !mo.shape, !mo.shape) {
) -> (!mo.shape, !mo.shape, !mo.shape) {
  // CHECK-DAG: %[[V1:.*]] = kgen.param.constant: !mo.shape = <[[SH1]]>
  // CHECK-DAG: %[[V2:.*]] = kgen.param.constant: !mo.shape = <[[[D1]], 42]>
  // CHECK-DAG: %[[V23:.*]] = kgen.param.constant: !mo.shape = <[[[D1]], 42, 3, [[D2]]]>
  %v1 = kgen.param.constant: !mo.shape = <Sh1>
  %v2 = kgen.param.constant: !mo.shape = <[D1, 42]>
  %v3 = kgen.param.constant: !mo.shape = <[3, D2]>
  // CHECK-DAG: %[[V11:.*]] = mo.shape.concat(%[[V1]], %[[V1]])
  // CHECK-DAG: %[[V12:.*]] = mo.shape.concat(%[[V1]], %[[V2]])
  %v11 = mo.shape.concat(%v1, %v1)
  %v12 = mo.shape.concat(%v1, %v2)
  %v23 = mo.shape.concat(%v2, %v3)
  // CHECK-NEXT: kgen.return %[[V11]], %[[V12]], %[[V23]]
  kgen.return %v11, %v12, %v23 : !mo.shape, !mo.shape, !mo.shape
}

// CHECK: kgen.func @concat_empty_fold() -> !mo.shape {
kgen.func @concat_empty_fold() -> !mo.shape {
  // CHECK-DAG: %[[V1:.*]] = kgen.param.constant: !mo.shape = <[]>
  %c1 = mo.shape.concat()
  // CHECK: kgen.return %[[V1]]
  kgen.return %c1 : !mo.shape
}

// CHECK: kgen.generator @concat_unary_fold<[[SH1:.*]]: !mo.shape>() -> (!mo.shape, !mo.shape)
kgen.generator @concat_unary_fold<Sh1: !mo.shape>() -> (!mo.shape, !mo.shape) {
  // CHECK-DAG: %[[V1:.*]] = kgen.param.constant: !mo.shape = <[[SH1]]>
  // CHECK-DAG: %[[V2:.*]] = kgen.param.constant: !mo.shape = <[[[D1]], 42]>
  %v1 = kgen.param.constant: !mo.shape = <Sh1>
  %v2 = kgen.param.constant: !mo.shape = <[D1, 42]>
  %c1 = mo.shape.concat(%v1)
  %c2 = mo.shape.concat(%v2)
  // CHECK: kgen.return %[[V1]], %[[V2]]
  kgen.return %c1, %c2 : !mo.shape, !mo.shape
}

// CHECK: kgen.generator @concat_variadic_fold<[[SH1:.*]]: !mo.shape, [[D1:.*]], [[D2:.*]]>(
kgen.generator @concat_variadic_fold<Sh1: !mo.shape, D1, D2>(
// CHECK-SAME: ) -> (!mo.shape, !mo.shape) {
) -> (!mo.shape, !mo.shape) {
  // CHECK-DAG: %[[V1:.*]] = kgen.param.constant: !mo.shape = <[[SH1]]>
  // CHECK-DAG: %[[V2:.*]] = kgen.param.constant: !mo.shape = <[[[D1]], 42]>
  // CHECK-DAG: %[[V232:.*]] = kgen.param.constant: !mo.shape = <[[[D1]], 42, 3, [[D2]], [[D1]], 42]>
  %v1 = kgen.param.constant: !mo.shape = <Sh1>
  %v2 = kgen.param.constant: !mo.shape = <[D1, 42]>
  %v3 = kgen.param.constant: !mo.shape = <[3, D2]>
  // CHECK-DAG: %[[V121:.*]] = mo.shape.concat(%[[V1]], %[[V2]], %[[V1]])
  %v121 = mo.shape.concat(%v1, %v2, %v1)
  %v232 = mo.shape.concat(%v2, %v3, %v2)
  // CHECK-NEXT: kgen.return %[[V121]], %[[V232]]
  kgen.return %v121, %v232 : !mo.shape, !mo.shape
}

// TODO: implement tests for shapes that do not fold
//       - becasue of unknown shape arg
//       - becasue of not broadcastable shapes
//       - becasue it's not possible to decide without asserts if broadcastable
// CHECK: kgen.generator @broadcast_fold_static() -> (!mo.shape, !mo.shape) {
kgen.generator @broadcast_fold_static() -> (!mo.shape, !mo.shape) {
  // CHECK-NEXT: %[[RES:.*]] = kgen.param.constant: !mo.shape = <[5, 10, 42]>
  %s0 = kgen.param.constant: !mo.shape = <[5, 10, 1]>
  %s1 = kgen.param.constant: !mo.shape = <[10, 42]>
  %s3 = mo.shape.broadcast(%s0, %s1)
  %s4 = mo.shape.broadcast(%s1, %s0)
  // CHECK-NEXT: kgen.return %[[RES]], %[[RES]] : !mo.shape, !mo.shape
  kgen.return %s3, %s4 : !mo.shape, !mo.shape
}

// CHECK: kgen.generator @broadcast_fold_dyn<[[N:.*]], [[K:.*]]>() -> (!mo.shape, !mo.shape) {
kgen.generator @broadcast_fold_dyn<N, K>() -> (!mo.shape, !mo.shape) {
  // CHECK: %[[RES:.*]] = kgen.param.constant: !mo.shape = <[[[N]], 10, 42]>
  %s0 = kgen.param.constant: !mo.shape = <[N, 10, 1]>
  %s1 = kgen.param.constant: !mo.shape = <[10, 42]>
  %s2 = kgen.param.constant: !mo.shape = <[1, 10, 42]>
  %s3 = mo.shape.broadcast(%s0, %s1)
  %s4 = mo.shape.broadcast(%s0, %s2)
  // CHECK-NEXT: kgen.return %[[RES]], %[[RES]] : !mo.shape, !mo.shape
  kgen.return %s3, %s4 : !mo.shape, !mo.shape
}

// CHECK: kgen.generator @broadcast_not_fold<[[N:.*]], [[K:.*]], [[SH:.*]]: !mo.shape>()
kgen.generator @broadcast_not_fold<N, K, Sh: !mo.shape>()
    // CHECK-SAME: -> (!mo.shape, !mo.shape, !mo.shape, !mo.shape) {
    -> (!mo.shape, !mo.shape, !mo.shape, !mo.shape) {
  // CHECK-DAG: %[[VAL0:.*]] = kgen.param.constant: !mo.shape = <[[[N]], 10, 42]>
  // CHECK-DAG: %[[VAL1:.*]] = kgen.param.constant: !mo.shape = <[[[N]], 10, 5]>
  // CHECK-DAG: %[[VAL2:.*]] = kgen.param.constant: !mo.shape = <[[[K]], 10, 42]>
  // CHECK-DAG: %[[VAL3:.*]] = kgen.param.constant: !mo.shape = <[1, 10, [[K]]]>
  // CHECK-DAG: %[[VAL4:.*]] = kgen.param.constant: !mo.shape = <[[SH]]>
  // CHECK-DAG: %[[VAL5:.*]] = mo.shape.broadcast(%[[VAL0]], %[[VAL1]])
  // CHECK-DAG: %[[VAL6:.*]] = mo.shape.broadcast(%[[VAL0]], %[[VAL2]])
  // CHECK-DAG: %[[VAL7:.*]] = mo.shape.broadcast(%[[VAL0]], %[[VAL3]])
  // CHECK-DAG: %[[VAL8:.*]] = mo.shape.broadcast(%[[VAL0]], %[[VAL4]])
  %s0 = kgen.param.constant: !mo.shape = <[N, 10, 42]>
  %s1 = kgen.param.constant: !mo.shape = <[N, 10, 5]>
  %s2 = kgen.param.constant: !mo.shape = <[K, 10, 42]>
  %s3 = kgen.param.constant: !mo.shape = <[1, 10, K]>
  %s4 = kgen.param.constant: !mo.shape = <Sh>
  %s5 = mo.shape.broadcast(%s0, %s1)
  %s6 = mo.shape.broadcast(%s0, %s2)
  %s7 = mo.shape.broadcast(%s0, %s3)
  %s8 = mo.shape.broadcast(%s0, %s4)
  // CHECK-NEXT: kgen.return %[[VAL5]], %[[VAL6]], %[[VAL7]], %[[VAL8]] : !mo.shape, !mo.shape, !mo.shape, !mo.shape
  kgen.return %s5, %s6, %s7, %s8 : !mo.shape, !mo.shape, !mo.shape, !mo.shape
}

// CHECK: kgen.generator @equal_fold<[[SH1:.*]]: !mo.shape, [[D1:.*]], [[D2:.*]]>()
kgen.generator @equal_fold<Sh1: !mo.shape, D1, D2>() -> (i1, i1, i1, i1, i1) {
  // CHECK-DAG: %[[V1:.*]] = kgen.param.constant: !mo.shape = <[[SH1]]>
  // CHECK-DAG: %[[V2:.*]] = kgen.param.constant: !mo.shape = <[[[D1]], 42]>
  // CHECK-DAG: %[[V3:.*]] = kgen.param.constant: !mo.shape = <[3, [[D2]]]>
  // CHECK-DAG: %[[TRUE:.*]] = index.bool.constant true
  // CHECK-DAG: %[[B12:.*]] = mo.shape.eq(%[[V1]], %[[V2]])
  // CHECK-DAG: %[[B23:.*]] = mo.shape.eq(%[[V2]], %[[V3]])
  // CHECK-DAG: %[[FALSE:.*]] = index.bool.constant false
  // CHECK: kgen.return %[[TRUE]], %[[B12]], %[[B23]], %[[FALSE]], %[[FALSE]]
  %v1 = kgen.param.constant: !mo.shape = <Sh1>
  %v2 = kgen.param.constant: !mo.shape = <[D1, 42]>
  %v3 = kgen.param.constant: !mo.shape = <[3, D2]>
  %v4 = kgen.param.constant: !mo.shape = <[4, D2]>
  %v5 = kgen.param.constant: !mo.shape = <[D1, D2, 3, 4]>
  %b11 = mo.shape.eq(%v1, %v1)
  %b12 = mo.shape.eq(%v1, %v2)
  %b23 = mo.shape.eq(%v2, %v3)
  %b34 = mo.shape.eq(%v3, %v4)
  %b25 = mo.shape.eq(%v2, %v5)
  kgen.return %b11, %b12, %b23, %b34, %b25 : i1, i1, i1, i1, i1
}

// CHECK: mo.graph @from_tensor_fold() -> !mo.shape {
mo.graph @from_tensor_fold() -> !mo.shape {
  // CHECK: %[[RES:.*]] = kgen.param.constant: !mo.shape = <[2, 3]>
  %sh1 = mo.constant {value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %0 = mo.shape.from_tensor(%sh1) : !mo.tensor<[2], si64>
  // CHECK: mo.output %[[RES]] : !mo.shape
  mo.output %0 : !mo.shape
}

// CHECK: kgen.generator @get_dim_fold<[[SH1:.*]]: !mo.shape, [[D1:.*]]>()
kgen.generator @get_dim_fold<Sh1: !mo.shape, D1>() -> (index, index, index, index) {
  // CHECK-DAG: %[[V1:.*]] = kgen.param.constant: !mo.shape = <[[SH1]]>
  // CHECK-DAG: %[[D10:.*]] = mo.shape.get_dim(%[[V1]])[0]
  // CHECK-DAG: %[[D20:.*]] = kgen.param.constant = <[[D1]]>
  // CHECK-DAG: %[[D21:.*]] = index.constant 42
  // CHECK-DAG: %[[D22:.*]] = index.constant 23
  // CHECK: kgen.return %[[D10]], %[[D20]], %[[D21]], %[[D22]]
  %v1 = kgen.param.constant: !mo.shape = <Sh1>
  %v2 = kgen.param.constant: !mo.shape = <[D1, 42, 23]>
  %d10 = mo.shape.get_dim(%v1)[0]
  %d20 = mo.shape.get_dim(%v2)[0]
  %d21 = mo.shape.get_dim(%v2)[1]
  %d22 = mo.shape.get_dim(%v2)[-1]
  kgen.return %d10, %d20, %d21, %d22 : index, index, index, index
}

// CHECK: kgen.generator @get_rank_fold<[[SH1:.*]]: !mo.shape, [[D1:.*]]>()
kgen.generator @get_rank_fold<Sh1: !mo.shape, D1>() -> (index, index, index) {
  // CHECK-DAG: %[[V1:.*]] = kgen.param.constant: !mo.shape = <[[SH1]]>
  // CHECK-DAG: %[[R1:.*]] = mo.shape.get_rank(%[[V1]])
  // CHECK-DAG: %[[R2:.*]] = index.constant 2
  // CHECK-DAG: %[[R3:.*]] = index.constant 0
  // CHECK: kgen.return %[[R1]], %[[R2]]
  %v1 = kgen.param.constant: !mo.shape = <Sh1>
  %v2 = kgen.param.constant: !mo.shape = <[D1, 42]>
  %v3 = kgen.param.constant: !mo.shape = <[]>
  %r1 = mo.shape.get_rank(%v1)
  %r2 = mo.shape.get_rank(%v2)
  %r3 = mo.shape.get_rank(%v3)
  kgen.return %r1, %r2, %r3 : index, index, index
}

// CHECK: kgen.generator @new_fold<[[D1:.*]], [[D2:.*]]>()
kgen.generator @new_fold<D1, D2>() -> !mo.shape {
  // CHECK: %[[S1:.*]] = kgen.param.constant: !mo.shape = <[[[D1]], 42, [[D2]]]>
  // CHECK: kgen.return %[[S1]]
  %d1 = kgen.param.constant: index = <D1>
  %d2 = index.constant 42
  %d3 = kgen.param.constant: index = <D2>
  %s1 = mo.shape.new(%d1, %d2, %d3)
  kgen.return %s1 : !mo.shape
}

// CHECK: kgen.generator @slice_fold<[[SH1:.*]]: !mo.shape, [[D1:.*]], [[D2:.*]]>(
kgen.generator @slice_fold<Sh1: !mo.shape, D1, D2>(
// CHECK: ) -> (!mo.shape, !mo.shape, !mo.shape, !mo.shape) {
) -> (!mo.shape, !mo.shape, !mo.shape, !mo.shape) {
  // CHECK-DAG: %[[V1:.*]] = kgen.param.constant: !mo.shape = <[[SH1]]>
  // CHECK-DAG: %[[V2:.*]] = kgen.param.constant: !mo.shape = <[[[D1]]]>
  // CHECK-DAG: %[[V3:.*]] = kgen.param.constant: !mo.shape = <[[[D2]], [[D1]]]>
  // CHECK-DAG: %[[V4:.*]] = kgen.param.constant: !mo.shape = <[42, [[D2]]]>
  // CHECK-DAG: %[[S1:.*]] = mo.shape.slice(%[[V1]])[0, -1]
  // CHECK: kgen.return %[[S1]], %[[V2]], %[[V3]], %[[V4]]
  %v1 = kgen.param.constant: !mo.shape = <Sh1>
  %v2 = kgen.param.constant: !mo.shape = <[D1, 42, D2]>
  %v3 = kgen.param.constant: !mo.shape = <[3, D2, D1]>
  %s1 = mo.shape.slice(%v1)[0, -1]
  %s2 = mo.shape.slice(%v2)[0, -2]
  %s3 = mo.shape.slice(%v3)[1]
  %s4 = mo.shape.slice(%v2)[-2]
  kgen.return %s1, %s2, %s3, %s4 : !mo.shape, !mo.shape, !mo.shape, !mo.shape
}

// CHECK: mo.graph @to_tensor_fold<[[D0:.*]], [[SH0:.*]]: !mo.shape>() -> (
mo.graph @to_tensor_fold<D0, Sh0: !mo.shape>() -> (
  // CHECK-SAME: !mo.tensor<[?], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  !mo.tensor<[?], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
) {
  // CHECK-DAG: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<17, 42> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[VAL1:.*]] = kgen.param.constant: !mo.shape = <[[SH0]]>
  // CHECK-DAG: %[[VAL2:.*]] = kgen.param.constant: !mo.shape = <[[[D0]], 42]>
  // CHECK-DAG: %[[VAL3:.*]] = mo.shape.to_tensor(%[[VAL1]]) -> !mo.tensor<[?], si64>
  // CHECK-DAG: %[[VAL4:.*]] = mo.shape.to_tensor(%[[VAL2]]) -> !mo.tensor<[2], si64>
  %sh0 = kgen.param.constant: !mo.shape = <Sh0>
  %sh1 = kgen.param.constant: !mo.shape = <[D0, 42]>
  %sh2 = kgen.param.constant: !mo.shape = <[17, 42]>
  %0 = mo.shape.to_tensor(%sh0) -> !mo.tensor<[?], si64>
  %1 = mo.shape.to_tensor(%sh1) -> !mo.tensor<[2], si64>
  %2 = mo.shape.to_tensor(%sh2) -> !mo.tensor<[2], si64>
  // CHECK-NEXT: mo.output %[[VAL3]], %[[VAL4]], %[[VAL0]] : !mo.tensor<[?], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  mo.output %0, %1, %2 : !mo.tensor<[?], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
}

// CHECK: mo.graph @to_tensor_fold_unknown<[[K:.*]]>() -> !mo.tensor<[?], si32> {
mo.graph @to_tensor_fold_unknown<K>() -> !mo.tensor<[?], si32> {
  // CHECK: %[[VAL0:.*]] = kgen.param.constant: !mo.shape = <[[[K]]]>
  %0 = kgen.param.constant: !mo.shape = <[K]>
  // CHECK: %[[VAL1:.*]] = mo.shape.to_tensor(%[[VAL0]]) -> !mo.tensor<[1], si32>
  %1 = mo.shape.to_tensor(%0) -> !mo.tensor<[?], si32>
  // CHECK: %[[VAL2:.*]] = mo.rebind(%[[VAL1]]) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  // CHECK: mo.output %[[VAL2]] : !mo.tensor<[?], si32>
  mo.output %1 : !mo.tensor<[?], si32>
}
