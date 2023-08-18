// RUN: mo-opt %s --constant-fold --mlir-print-ir-after=mo-to-mogg --allow-unregistered-dialect 2>&1 | FileCheck %s

// These are checks to make sure that the constant subgraphs are lowered
// instead of being connonicalized. We inspect the mlir code after mo-to-mogg
// to make sure that the kernels exist.
// CHECK: IR Dump After MOToMOGG
// CHECK: module
// CHECK: all_const
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

// CHECK: shape_func
// CHECK-DAG: kgen.param.declare MMPackSh0 = <768>
// CHECK-DAG: kgen.param.declare MMPackSh1 = <768>
// CHECK-DAG: %[[A0:.*]] = mo.constant {value = dense_resource<{{.*}}> : !M.array<589824xf32>
// CHECK-DAG: %[[A1:.*]] = mo.rebind(%[[A0]])
// CHECK: mo.output %[[A1]]
mo.graph @shape_func() -> (!mo.tensor<[?, ?], f32>) {
  %1 = mo.constant {value = #M.dense_array<1.> : tensor<1xf32>} : !mo.tensor<[1], f32>
  %2 = mo.static.broadcast_to(%1):!mo.tensor<[1], f32> -> !mo.tensor<[768,768], f32>
  mo.invoke_shape_func["pack_matmul_b_shape_func"]<() -> MMPackSh0, MMPackSh1>(%2) {kgenParams = {transpose_in_0 = false}} : (!mo.tensor<[768, 768], f32>)
  %3 = mo.pack_mm_weights(%2 : !mo.tensor<[768, 768], f32>) {transposed = false} -> !mo.tensor<[MMPackSh0, MMPackSh1], f32>
  %4 = mo.rebind(%3): !mo.tensor<[MMPackSh0, MMPackSh1], f32> -> !mo.tensor<[?, ?], f32>
  mo.output %4: !mo.tensor<[?, ?], f32>
}

// CHECK: constant_params
// CHECK: %[[A0:.*]] = mo.constant {value = #M.dense_array<1, 1>
// CHECL: %[[A1:.*]] = mo.rebind(%[[A0]])
// CHECK: mo.output %[[A1]]
mo.graph @constant_params() -> (!mo.tensor<[?], si32>) {
  kgen.param.declare K = <1>
  kgen.param.declare KAdd = <add(K, K)>
  %0 = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %1 = mo.static.broadcast_to(%0) : !mo.tensor<[1], si32> -> !mo.tensor<[KAdd], si32>
  %2 = mo.rebind(%1): !mo.tensor<[KAdd], si32> -> !mo.tensor<[?], si32>
  mo.output %2 : !mo.tensor<[?], si32>
}

// CHECK: index_to_tensor
// CHECK: %[[A0:.*]] = mo.constant {value = #M.dense_array<1>
// CHECK: mo.output %[[A0]]
mo.graph @index_to_tensor() -> !mo.tensor<[], si32> {
  %0 = kgen.param.constant = <1>
  %1 = mo.index.to_tensor(%0) -> !mo.tensor<[], si32>
  mo.output %1 : !mo.tensor<[], si32>
}

// CHECK: constant_param_used_externally
// CHECK-DAG: kgen.param.declare K = <1>
// CHECK-DAG: kgen.param.declare KAdd = <2>
// CHECK: %[[A0:.*]] = mo.constant {value = #M.dense_array<1, 1> : !M.array<2xsi32>} : !mo.tensor<[KAdd], si32>
// CHECK: %[[A1:.*]] = mo.rebind(%{{.*}}) : !mo.tensor<[2], si32> -> !mo.tensor<[KAdd], si32>
// CHECK: %[[A2:.*]] = mo.add(%[[A1]], %[[A0]]) : !mo.tensor<[KAdd], si32>
// CHECK: %[[A3:.*]] = mo.rebind(%[[A2]]) : !mo.tensor<[KAdd], si32> -> !mo.tensor<[?], si32>
// CHECK: mo.output %3 : !mo.tensor<[?], si32>
mo.graph @constant_param_used_externally(%arg0 : !mo.tensor<[2], si32>) -> (!mo.tensor<[?], si32>) {
  kgen.param.declare K = <1>
  kgen.param.declare KAdd = <add(K, K)>
  %0 = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %1 = mo.static.broadcast_to(%0) : !mo.tensor<[1], si32> -> !mo.tensor<[KAdd], si32>
  %2 = mo.rebind(%arg0): !mo.tensor<[2], si32> -> !mo.tensor<[KAdd], si32>
  %3 = mo.add(%2, %1) : !mo.tensor<[KAdd], si32>
  %4 = mo.rebind(%3): !mo.tensor<[KAdd], si32> -> !mo.tensor<[?], si32>
  mo.output %4 : !mo.tensor<[?], si32>
}
