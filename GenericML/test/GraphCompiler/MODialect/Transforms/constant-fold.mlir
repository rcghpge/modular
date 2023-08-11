// RUN: mo-opt %s --constant-fold --mlir-print-ir-after=mo-to-mogg --allow-unregistered-dialect 2>&1 | FileCheck %s

// These are checks to make sure that the constant subgraphs are lowered
// instead of being connonicalized. We inspect the mlir code after mo-to-mogg
// to make sure that the kernels exist.
// CHECK: IR Dump After MOToMOGG
// CHECK-NEXT: module
// CHECK-NEXT: all_const
// CHECK-DAG: mo.mul
// CHECK-DAG: mo.sub
// CHECK-DAG: mo.add

// CHECK: multiple_subgraphs
// CHECK-DAG: mo.mul
// CHECK-DAG: mo.sub

// CHECK: if_not_handled
// CHECK: mo.mul

// CHECK: nested_region
// CHECK: mo.mul

// No kernel for large_broadcast because it is not constant folded.
// CHECK-NOT: large_broadcast

// ensure we hit the end of mo-to-mogg ir printing
// CHECK: module

// CHECK: all_const
// CHECK-NEXT: %[[S1:.*]] = mo.constant {value = #M.dense_array<0, 5>
// CHECK-NEXT: mo.output %[[S1]]
mo.graph @all_const() -> (!mo.tensor<[2], si32>) {
  %s1 = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>
  %s2 = mo.constant {value = #M.dense_array<6, 3> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>

  %s3 = mo.mul(%s1, %s2) : !mo.tensor<[2], si32>
  %s4 = mo.sub(%s3, %s1) : !mo.tensor<[2], si32>
  %s5 = mo.add(%s4, %s3) : !mo.tensor<[2], si32>
  mo.output %s5 : !mo.tensor<[2], si32>
}

// CHECK: multiple_subgraphs(%[[R1:.*]]:
mo.graph @multiple_subgraphs(%r1 : !mo.tensor<[2], si32>) -> (!mo.tensor<[2], si32>) {
  %r2 = mo.constant {value = #M.dense_array<1, -1> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>
  %r3 = mo.constant {value = #M.dense_array<6, 3> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>

  // CHECK-DAG: %[[R4:.*]] = mo.constant {value = #M.dense_array<1, 1>
  %r4 = mo.mul(%r2, %r2) : !mo.tensor<[2], si32>
  // CHECK-DAG: %[[R5:.*]] = mo.constant {value = #M.dense_array<0, 0>
  %r5 = mo.sub(%r3, %r3) : !mo.tensor<[2], si32>
  // CHECK-DAG = mo.sub(%[[R1]], %[[R5]])
  %r6 = mo.sub(%r1, %r5) : !mo.tensor<[2], si32>
  // CHECK-DAG = mo.sub(%[[R1]], %[[R4]])
  %r7 = mo.sub(%r1, %r4) : !mo.tensor<[2], si32>
  %r8 = mo.sub(%r6, %r7) : !mo.tensor<[2], si32>
  mo.output %r8 : !mo.tensor<[2], si32>
}

// TODO: if should be handled by applying connonicalization
// CHECK: if_not_handled
mo.graph @if_not_handled(%e1 : !mo.tensor<[2], si32>) -> (!mo.tensor<[2], si32>) {
  // CHECK-DAG: %[[E2:.*]] = mo.constant {value = #M.dense_array<0, 9>
  %e2 = mo.constant {value = #M.dense_array<0, 3> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>
  // CHECK-DAG: %{{.*}} = mo.constant {value = #M.dense_array<0, 6>
  // CHECK-DAG: %{{.*}} = mo.constant {value = #M.dense_array<1>
  %e3 = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} :
				!mo.tensor<[], bool>

  // CHECK-NEXT: mo.if
  // CHECK-NEXT: mo.yield %[[E2]]
  %e4 = mo.if(%e3 : !mo.tensor<[], bool>) -> !mo.tensor<[2], si32> {
    %e5 = mo.mul(%e2, %e2) : !mo.tensor<[2], si32>
    mo.yield %e5 : !mo.tensor<[2], si32>
  } else {
    %e6 = mo.add(%e2, %e2) : !mo.tensor<[2], si32>
    mo.yield %e6 : !mo.tensor<[2], si32>
  }
  mo.output %e4 : !mo.tensor<[2], si32>
}

// CHECK: nested_region
mo.graph @nested_region(%g1 : !mo.tensor<[], bool>) -> (!mo.tensor<[2], si32>) {
  // CHECK-DAG: %[[G0:.*]] = mo.constant {value = #M.dense_array<0, 1>
  // CHECK-DAG: %[[G1:.*]] = mo.constant {value = #M.dense_array<6, 1>
  // CHECK-NEXT: mo.if
  %g2 = mo.if(%g1 : !mo.tensor<[], bool>) -> !mo.tensor<[2], si32> {
    // CHECK-NEXT: mo.yield %[[G0]]
    %g3 = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>
    %g4 = mo.mul(%g3, %g3) : !mo.tensor<[2], si32>
    mo.yield %g4 : !mo.tensor<[2], si32>
  // CHECK-NEXT: else
  } else {
    // CHECK-NEXT: mo.yield %[[G1]]
    %g3 = mo.constant {value = #M.dense_array<6, 1> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>
    mo.yield %g3 : !mo.tensor<[2], si32>
  }
  mo.output %g2 : !mo.tensor<[2], si32>
}

// CHECK: large_broadcast
mo.graph @large_broadcast() -> (!mo.tensor<[1024], si8>) {
  // CHECK-NEXT: %{{.*}} = mo.constant {value = #M.dense_array<1024>
  // CHECK-NEXT: %{{.*}} = mo.constant {value = #M.dense_array<1>
  // CHECK-NEXT: mo.broadcast
  // CHECK-NEXT: mo.output
  %shape = mo.constant {value = #M.dense_array<1024> : tensor<1xsi32>} :
			!mo.tensor<[1], si32>
  %data = mo.constant {value = #M.dense_array<1> : tensor<1xsi8>} :
			!mo.tensor<[1], si8>
  %broadcast = mo.broadcast_to(%data, %shape) :
     (!mo.tensor<[1], si8>, !mo.tensor<[1], si32>) -> !mo.tensor<[1024], si8>
  mo.output %broadcast : !mo.tensor<[1024], si8>
}

// CHECK: repeated_application
mo.graph @repeated_application(%e0 : !mo.tensor<[], si32>) -> !mo.tensor<[], si32> {
  // CHECK-NEXT: %{{.*}} = mo.constant {value = #M.dense_array<4>
  // CHECK-NEXT: mo.output
  %e1 = mo.constant {value = #M.dense_array<3> : tensor<1xsi32>} :
				!mo.tensor<[], si32>
  %e2 = mo.constant {value = #M.dense_array<2> : tensor<1xsi32>} :
				!mo.tensor<[], si32>
  %e3 = mo.add(%e1, %e1) : !mo.tensor<[], si32>
  %e4 = mo.greater_equal(%e3, %e2) :(!mo.tensor<[], si32>, !mo.tensor<[], si32>)
    -> !mo.tensor<[], bool>
  %e5 = mo.select(%e4, %e3, %e0) :
    (!mo.tensor<[], bool>, !mo.tensor<[], si32>, !mo.tensor<[], si32>)
    -> !mo.tensor<[], si32>
  %e6 = mo.sub(%e5, %e2) : !mo.tensor<[], si32>
  mo.output %e6 : !mo.tensor<[], si32>
}
