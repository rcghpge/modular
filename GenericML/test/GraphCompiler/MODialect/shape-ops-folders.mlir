// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK: kgen.generator @concat_fold<[[SH1:.*]]: !mo.shape, [[D1:.*]], [[D2:.*]]>(
kgen.generator @concat_fold<Sh1: !mo.shape, D1, D2>(
// CHECK: ) -> (!mo.shape, !mo.shape, !mo.shape) {
) -> (!mo.shape, !mo.shape, !mo.shape) {
  // CHECK-DAG: %[[V1:.*]] = kgen.param.constant: !mo.shape = <[[SH1]]>
  // CHECK-DAG: %[[V2:.*]] = kgen.param.constant: !mo.shape = <[[[D1]], 42]>
  // CHECK-DAG: %[[V23:.*]] = kgen.param.constant: !mo.shape = <[[[D1]], 42, 3, [[D2]]]>
  // CHECK-DAG: %[[V11:.*]] = mo.shape.concat(%[[V1]], %[[V1]])
  // CHECK-DAG: %[[V12:.*]] = mo.shape.concat(%[[V1]], %[[V2]])
  // CHECK: kgen.return %[[V11]], %[[V12]], %[[V23]]
  %v1 = kgen.param.constant: !mo.shape = <Sh1>
  %v2 = kgen.param.constant: !mo.shape = <[D1, 42]>
  %v3 = kgen.param.constant: !mo.shape = <[3, D2]>
  %v11 = mo.shape.concat(%v1, %v1)
  %v12 = mo.shape.concat(%v1, %v2)
  %v23 = mo.shape.concat(%v2, %v3)
  kgen.return %v11, %v12, %v23 : !mo.shape, !mo.shape, !mo.shape
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
