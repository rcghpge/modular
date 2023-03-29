// RUN: mo-opt %s --canonicalize | mo-opt | FileCheck %s

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// Static_BrodcastToOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @static_broadcast_to<[[N:.*]]>(%[[FROM:.*]]: !mo.tensor<[[[N]], 3], f32>) -> !mo.tensor<[1, [[N]], 3], f32> {
mo.graph @static_broadcast_to<N>(%from: !mo.tensor<[N, 3], f32>) -> !mo.tensor<[1, N, 3], f32> {
  // CHECK: [[RES:.*]] = mo.static.broadcast_to(%[[FROM]]) : !mo.tensor<[[[N]], 3], f32> -> !mo.tensor<[1, [[N]], 3], f32>
  %res = mo.static.broadcast_to(%from) : !mo.tensor<[N, 3], f32> -> !mo.tensor<[1, N, 3], f32>
  mo.output %res : !mo.tensor<[1, N, 3], f32>
}

// CHECK: mo.graph @static_broadcast_to_fold<[[N:.*]]>(%[[FROM:.*]]: !mo.tensor<[[[N]], 3], f32>) -> !mo.tensor<[[[N]], 3], f32> {
mo.graph @static_broadcast_to_fold<N>(%from: !mo.tensor<[N, 3], f32>) -> !mo.tensor<[N, 3], f32> {
  %res = mo.static.broadcast_to(%from) : !mo.tensor<[N, 3], f32> -> !mo.tensor<[N, 3], f32>
  // CHECK: mo.output %[[FROM]] : !mo.tensor<[[[N]], 3], f32>
  mo.output %res : !mo.tensor<[N, 3], f32>
}

// CHECK: mo.graph @static_broadcast_to_no_fold<[[D0:.*]], [[D1:.*]]>(
mo.graph @static_broadcast_to_no_fold<D0, D1>(
  // CHECK-SAME: %[[FROM:.*]]: !mo.tensor<[[[D0]], 3], f32>,
  %from: !mo.tensor<[D0, 3], f32>,
  // CHECK-SAME: %[[TO:.*]]: !mo.tensor<[[[D1]], 3], f32>
  %to: !mo.tensor<[D1, 3], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D1]], 3], f32> {
) -> !mo.tensor<[D1, 3], f32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.static.broadcast_to(%[[FROM]]) : !mo.tensor<[[[D0]], 3], f32> -> !mo.tensor<[[[D1]], 3], f32>
  %res = mo.static.broadcast_to(%from) : !mo.tensor<[D0, 3], f32> -> !mo.tensor<[D1, 3], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D1]], 3], f32>
  mo.output %res : !mo.tensor<[D1, 3], f32>
}

//===----------------------------------------------------------------------===//
// Static_ReshapeOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @static_reshape<[[N:.*]]>(%[[FROM:.*]]: !mo.tensor<[[[N]], 3], f32>)
mo.graph @static_reshape<N>(%from: !mo.tensor<[N, 3], f32>) -> !mo.tensor<[1, 3, N], f32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.static.reshape(%[[FROM]]) : !mo.tensor<[[[N]], 3], f32> -> !mo.tensor<[1, 3, [[N]]], f32>
  %res = mo.static.reshape(%from) : !mo.tensor<[N, 3], f32> -> !mo.tensor<[1, 3, N], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[1, 3, [[N]]], f32>
  mo.output %res : !mo.tensor<[1, 3, N], f32>
}

// CHECK: mo.graph @static_reshape_fold_same<[[N:.*]]>(%[[FROM:.*]]: !mo.tensor<[[[N]], 3], f32>)
mo.graph @static_reshape_fold_same<N>(%from: !mo.tensor<[N, 3], f32>) -> !mo.tensor<[N, 3], f32> {
  %res = mo.static.reshape(%from) : !mo.tensor<[N, 3], f32> -> !mo.tensor<[N, 3], f32>
  // CHECK-NEXT: mo.output %[[FROM]] : !mo.tensor<[N, 3], f32>
  mo.output %res : !mo.tensor<[N, 3], f32>
}

// CHECK: mo.graph @static_reshape_fold_static_out_shape_attr() -> !mo.tensor<[2, 2], si32> {
mo.graph @static_reshape_fold_static_out_shape_attr() -> !mo.tensor<[2, 2], si32> {
  // CHECK-NEXT: %[[INPUT:.*]] = mo.constant {value = #M.dense_array<1, 2, 3, 4> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>
  %from = mo.constant {value = #M.dense_array<1, 2, 3, 4> : tensor<4xsi32>} : !mo.tensor<[4], si32>
  %res = mo.static.reshape(%from) : !mo.tensor<[4], si32> -> !mo.tensor<[2, 2], si32>
  // CHECK-NEXT: mo.output %[[INPUT]] : !mo.tensor<[2, 2], si32>
  mo.output %res : !mo.tensor<[2, 2], si32>
}

// CHECK: mo.graph @static_reshape_no_fold_neg_one_dim() -> !mo.tensor<[-1], si32> {
mo.graph @static_reshape_no_fold_neg_one_dim() -> !mo.tensor<[-1], si32> {
  // CHECK-NEXT: %[[INPUT:.*]] = mo.constant {value = #M.dense_array<1, 2, 3, 4> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>
  %from = mo.constant {value = #M.dense_array<1, 2, 3, 4> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>
  // CHECK-NEXT: %[[RES:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[2, 2], si32> -> !mo.tensor<[-1], si32>
  %res = mo.static.reshape(%from) : !mo.tensor<[2, 2], si32> -> !mo.tensor<[-1], si32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[-1], si32>
  mo.output %res : !mo.tensor<[-1], si32>
}

// CHECK: mo.graph @static_reshape_fold_to_rebind_static<[[A:.*]], [[B:.*]]>(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[2, [[B]]], f32> {
mo.graph @static_reshape_fold_to_rebind_static<A, B>(%arg0: !mo.tensor<[A, B], f32>) -> !mo.tensor<[2, B], f32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.rebind(%[[ARG0]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[2, [[B]]], f32>
  %0 = mo.static.reshape(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[2, B], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, B], f32>
  mo.output %0 : !mo.tensor<[2, B], f32>
}

// CHECK: mo.graph @static_reshape_fold_to_rebind_param<[[A:.*]], [[B:.*]]>(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[[[A]], [[A]]], f32> {
mo.graph @static_reshape_fold_to_rebind_param<A, B>(%arg0: !mo.tensor<[A, B], f32>) -> !mo.tensor<[A, A], f32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.rebind(%[[ARG0]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[[[A]], [[A]]], f32>
  %0 = mo.static.reshape(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[A, A], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[A]], [[A]]], f32>
  mo.output %0 : !mo.tensor<[A, A], f32>
}

// CHECK: mo.graph @static_reshape_not_fold_to_rebind<[[A:.*]], [[B:.*]]>(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[[[B]], [[A]]], f32> {
mo.graph @static_reshape_not_fold_to_rebind<A, B>(%arg0: !mo.tensor<[A, B], f32>) -> !mo.tensor<[B, A], f32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.static.reshape(%[[ARG0]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[[[B]], [[A]]], f32>
  %0 = mo.static.reshape(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[B, A], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[B]], [[A]]], f32>
  mo.output %0 : !mo.tensor<[B, A], f32>
}
