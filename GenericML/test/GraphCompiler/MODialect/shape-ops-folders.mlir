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
