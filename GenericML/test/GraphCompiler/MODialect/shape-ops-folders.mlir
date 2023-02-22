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
