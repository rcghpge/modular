// RUN: mo-opt %s --canonicalize | FileCheck %s

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// FromTensorOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @from_tensor_fold() -> !mosh.ape {
mo.graph @from_tensor_fold() -> !mosh.ape {
  // CHECK: %[[RES:.*]] = kgen.param.constant: !mosh.ape = <[2, 3]>
  %sh1 = mo.constant {value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %0 = mo.shape.from_tensor(%sh1) : !mo.tensor<[2], si64>
  // CHECK: mo.output %[[RES]] : !mosh.ape
  mo.output %0 : !mosh.ape
}

//===----------------------------------------------------------------------===//
// ToTensorOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @to_tensor_fold<[[D0:.*]], [[SH0:.*]]: !mosh.ape -> [[D1:.*]]>() -> (
mo.graph @to_tensor_fold<D0, Sh0: !mosh.ape -> D1>() -> (
  // CHECK-SAME: !mo.tensor<[?], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[1], si32>, !mo.tensor<[[[D1]]], si32>
  !mo.tensor<[?], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[1], si32>, !mo.tensor<[D1], si32>
) {
  // CHECK-DAG: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<17, 42> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[VAL1:.*]] = kgen.param.constant: !mosh.ape = <[[SH0]]>
  // CHECK-DAG: %[[VAL2:.*]] = kgen.param.constant: !mosh.ape = <[[[D0]], 42]>
  // CHECK-DAG: %[[VAL3:.*]] = mo.shape.to_tensor(%[[VAL1]]) -> !mo.tensor<[?], si64>
  // CHECK-DAG: %[[VAL4:.*]] = mo.shape.to_tensor(%[[VAL2]]) -> !mo.tensor<[2], si64>
  // CHECK-DAG: %[[VAL5:.*]] = mo.constant {value = #M.dense_array<23> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  // CHECK-DAG: %[[VAL6:.*]] = mo.rebind(%[[VAL5]]) : !mo.tensor<[1], si32> -> !mo.tensor<[[[D0]]], si32>
  %sh0 = kgen.param.constant: !mosh.ape = <Sh0>
  %sh1 = kgen.param.constant: !mosh.ape = <[D0, 42]>
  %sh2 = kgen.param.constant: !mosh.ape = <[17, 42]>
  %sh3 = kgen.param.constant: !mosh.ape = <[23]>
  %0 = mo.shape.to_tensor(%sh0) -> !mo.tensor<[?], si64>
  %1 = mo.shape.to_tensor(%sh1) -> !mo.tensor<[2], si64>
  %2 = mo.shape.to_tensor(%sh2) -> !mo.tensor<[2], si64>
  %3 = mo.shape.to_tensor(%sh3) -> !mo.tensor<[1], si32>
  %4 = mo.shape.to_tensor(%sh3) -> !mo.tensor<[D0], si32>
  // CHECK-NEXT: mo.output<[[D0]]> %[[VAL3]], %[[VAL4]], %[[VAL0]], %[[VAL5]], %[[VAL6]] : !mo.tensor<[?], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[1], si32>, !mo.tensor<[D0], si32>
  mo.output<D0> %0, %1, %2, %3, %4 : !mo.tensor<[?], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[1], si32>, !mo.tensor<[D0], si32>
}

// CHECK: mo.graph @to_tensor_fold_unknown<[[K:.*]]>() -> !mo.tensor<[?], si32> {
mo.graph @to_tensor_fold_unknown<K>() -> !mo.tensor<[?], si32> {
  // CHECK: %[[VAL0:.*]] = kgen.param.constant: !mosh.ape = <[[[K]]]>
  %0 = kgen.param.constant: !mosh.ape = <[K]>
  // CHECK: %[[VAL1:.*]] = mo.shape.to_tensor(%[[VAL0]]) -> !mo.tensor<[1], si32>
  %1 = mo.shape.to_tensor(%0) -> !mo.tensor<[?], si32>
  // CHECK: %[[VAL2:.*]] = mo.rebind(%[[VAL1]]) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  // CHECK: mo.output %[[VAL2]] : !mo.tensor<[?], si32>
  mo.output %1 : !mo.tensor<[?], si32>
}
