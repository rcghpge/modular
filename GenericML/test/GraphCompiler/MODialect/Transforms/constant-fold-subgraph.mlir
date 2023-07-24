// RUN: mo-opt %s --constant-fold-subgraph-test | FileCheck %s

// CHECK-LABEL: all_const
// CHECK-NEXT: %[[S1:.*]] = mo.constant {value = #M.dense_array<0, 1>
// CHECK-NEXT: %[[S2:.*]] = mo.constant {value = #M.dense_array<6, 3>
// CHECK-NEXT: %[[S3:.*]] = mo.mul
// CHECK-NEXT: %[[S4:.*]] = mo.sub
// CHECK: subgraph %[[S1]] %[[S2]] %[[S3]] %[[S4]]
// CHECK-NOT: module
mo.graph @all_const() -> (!mo.tensor<[2], si32>) {
  %s1 = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>
  %s2 = mo.constant {value = #M.dense_array<6, 3> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>

  %s3 = mo.mul(%s1, %s2) : !mo.tensor<[2], si32>
  %s4 = mo.sub(%s3, %s1) : !mo.tensor<[2], si32>
  mo.output %s4 : !mo.tensor<[2], si32>
}

// CHECK: multiple_subgraphs(%[[R1:.*]]:
// CHECK-NEXT: %[[R2:.*]] = mo.constant {value = #M.dense_array<0, 1>
// CHECK-NEXT: %[[R3:.*]] = mo.constant {value = #M.dense_array<6, 3>
// CHECK-NEXT: %[[R4:.*]] = mo.mul
// CHECK-NEXT: %[[R5:.*]] = mo.sub
// CHECK-NEXT: %[[R6:.*]] = mo.sub
// CHECK-NEXT: %[[R7:.*]] = mo.sub
// CHECK-NEXT: %[[R8:.*]] = mo.sub
// CHECK-DAG: subgraph %[[R2]] %[[R4]]
// CHECK-DAG: subgraph %[[R3]] %[[R5]]
// CHECK-NOT: module
mo.graph @multiple_subgraphs(%r1 : !mo.tensor<[2], si32>) -> (!mo.tensor<[2], si32>) {
  %r2 = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>
  %r3 = mo.constant {value = #M.dense_array<6, 3> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>

  %r4 = mo.mul(%r2, %r2) : !mo.tensor<[2], si32>
  %r5 = mo.sub(%r3, %r3) : !mo.tensor<[2], si32>
  %r6 = mo.sub(%r1, %r5) : !mo.tensor<[2], si32>
  %r7 = mo.sub(%r1, %r4) : !mo.tensor<[2], si32>
  %r8 = mo.sub(%r6, %r7) : !mo.tensor<[2], si32>
  mo.output %r8 : !mo.tensor<[2], si32>
}

// CHECK: if_not_handled(%[[E1:.*]]:
// CHECK-NEXT: %[[E2:.*]] = mo.constant {value = #M.dense_array<0, 1>
// CHECK-NEXT: %[[E3:.*]] = mo.constant {value = #M.dense_array<1>
// CHECK-NEXT: %[[E4:.*]] = mo.if
// CHECK-NEXT: %[[E5:.*]] = mo.mul
// CHECK-DAG: %[[E6:.*]] = mo.add
// CHECK-DAG: subgraph %[[E2]] %[[E5]] %[[E6]]
mo.graph @if_not_handled(%e1 : !mo.tensor<[2], si32>) -> (!mo.tensor<[2], si32>) {
  %e2 = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>
  %e3 = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} :
				!mo.tensor<[], bool>

  %e4 = mo.if(%e3 : !mo.tensor<[], bool>) -> !mo.tensor<[2], si32> {
    %e5 = mo.mul(%e2, %e2) : !mo.tensor<[2], si32>
    mo.yield %e5 : !mo.tensor<[2], si32>
  } else {
    %e6 = mo.add(%e2, %e2) : !mo.tensor<[2], si32>
    mo.yield %e6 : !mo.tensor<[2], si32>
  }
  mo.output %e4 : !mo.tensor<[2], si32>
}

// CHECK-LABEL: nested_region
// CHECK: %[[G3:.*]] = mo.constant {value = #M.dense_array<0, 1>
// CHECK-NEXT: %[[G4:.*]] = mo.mul
// CHECK: subgraph %[[G3]] %[[G4]]
mo.graph @nested_region(%g1 : !mo.tensor<[], bool>) -> (!mo.tensor<[2], si32>) {
  %g2 = mo.if(%g1 : !mo.tensor<[], bool>) -> !mo.tensor<[2], si32> {
    %g3 = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>
    %g4 = mo.mul(%g3, %g3) : !mo.tensor<[2], si32>
    mo.yield %g4 : !mo.tensor<[2], si32>
  } else {
    %g3 = mo.constant {value = #M.dense_array<6, 1> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>
    mo.yield %g3 : !mo.tensor<[2], si32>
  }
  mo.output %g2 : !mo.tensor<[2], si32>
}

// make sure that we are not matching anything in the mlir of the result
// CHECK: module
