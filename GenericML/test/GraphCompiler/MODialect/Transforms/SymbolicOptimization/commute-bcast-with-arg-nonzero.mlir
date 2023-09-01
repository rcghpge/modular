// RUN: mo-opt %s --symbolic-optimize --propagate-shapes --canonicalize -cse | FileCheck %s

// CHECK: mo.graph @static_shape<() -> [[M:.*]]>(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[[[M]], 1], si64> no_inline {
mo.graph @static_shape<() -> M>(%arg0: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[M, 1], si64> {
  // CHECK-DAG: %[[RANGE:.*]] = mo.constant {value = #M.dense_array<0, 6, 12, 18, 24> : tensor<5x1xsi64>} : !mo.tensor<[5, 1], si64>
  // CHECK-DAG: %[[PERM:.*]] = mo.constant {value = #M.dense_array<1, 0> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[FLATARG:.*]] = mo.static.reshape(%[[ARG0]]) : !mo.tensor<[2, 3], f32> -> !mo.tensor<[6], f32>
  // CHECK-DAG: %[[WHERE:.*]] = mo.arg_nonzero<() -> [[W:.*]]>(%[[FLATARG]]) : (!mo.tensor<[6], f32>) -> !mo.tensor<[[[W]], 1], si64>
  // CHECK-DAG: %[[VAL5:.*]] = mo.static.broadcast_to(%[[WHERE]]) : !mo.tensor<[[[W]], 1], si64> -> !mo.tensor<[[[W]], 5], si64>
  // CHECK-DAG: %[[VAL6:.*]] = mo.static.broadcast_to(%[[RANGE]]) : !mo.tensor<[5, 1], si64> -> !mo.tensor<[5, [[W]]], si64>
  // CHECK-DAG: %[[WHERETP:.*]] = mo.transpose(%[[VAL5]], %[[PERM]]) : (!mo.tensor<[[[W]], 5], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, [[W]]], si64>
  // CHECK-DAG: %[[RES:.*]] = mo.add(%[[WHERETP]], %[[VAL6]]) : !mo.tensor<[5, [[W]]], si64>
  // CHECK-DAG: kgen.param.declare [[K:.*]] = <mul_nuw([[W]], 5)>
  // CHECK-DAG: %[[OLDRES:.*]] = mo.static.reshape(%[[RES]]) : !mo.tensor<[5, [[W]]], si64> -> !mo.tensor<[[[K]], 1], si64>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[2, 3], f32> -> !mo.tensor<[5, 2, 3], f32>
  %1 = mo.static.reshape(%0) : !mo.tensor<[5, 2, 3], f32> -> !mo.tensor<[30], f32>
  %2 = mo.arg_nonzero<() -> K>(%1) : (!mo.tensor<[30], f32>) -> !mo.tensor<[K, 1], si64>
  // CHECK: mo.output<[[K]]> %[[OLDRES]] : !mo.tensor<[[[K]], 1], si64>
  mo.output<K> %2 : !mo.tensor<[K, 1], si64>
}

// CHECK: mo.graph @dynamic_shape<[[A:.*]], [[B:.*]] -> [[M:.*]]>(%[[ARG:.*]]: !mo.tensor<[2, [[A]]], f32>) -> !mo.tensor<[[[M]], 1], si64> no_inline {
mo.graph @dynamic_shape<A, B -> M>(%arg0: !mo.tensor<[2, A], f32>) -> !mo.tensor<[M, 1], si64> {
  // CHECK-DAG: %[[C0:.*]] = kgen.param.constant = <mul_nuw([[A]], [[A]], [[B]], 2)>
  // CHECK-DAG: %[[C1:.*]] = kgen.param.constant = <mul_nuw([[A]], 2)>
  // CHECK-DAG: %[[START:.*]] = mo.constant {value = #M.dense_array<0> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-DAG: %[[LIMIT:.*]] = mo.index.to_tensor(%[[C0]]) -> !mo.tensor<[], si64>
  // CHECK-DAG: %[[STEP:.*]] = mo.index.to_tensor(%[[C1]]) -> !mo.tensor<[], si64>
  // CHECK-DAG: kgen.param.declare [[BS:.*]] = <mul_nuw([[A]], [[B]])>
  // CHECK-DAG: %[[RANGE:.*]] = mo.range(%[[START]], %[[LIMIT]], %[[STEP]]) : (!mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[BS]]], si64>
  // CHECK-DAG: %[[PERM:.*]] = mo.constant {value = #M.dense_array<1, 0> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: kgen.param.declare [[FS:.*]] = <mul_nuw([[A]], 2)>
  // CHECK-DAG: %[[FLATARG:.*]] = mo.static.reshape(%[[ARG0]]) : !mo.tensor<[2, [[A]]], f32> -> !mo.tensor<[[[FS]]], f32>
  // CHECK-DAG: %[[WHERE:.*]] = mo.arg_nonzero<() -> [[W:.*]]>(%[[FLATARG]]) : (!mo.tensor<[[[FS]]], f32>) -> !mo.tensor<[[[W]], 1], si64>
  // CHECK-DAG: %[[RNG2:.*]] = mo.static.reshape(%[[RANGE]]) : !mo.tensor<[[[BS]]], si64> -> !mo.tensor<[[[BS]], 1], si64>
  // CHECK-DAG: %[[VAL5:.*]] = mo.static.broadcast_to(%[[WHERE]]) : !mo.tensor<[[[W]], 1], si64> -> !mo.tensor<[[[W]], [[BS]]], si64>
  // CHECK-DAG: %[[VAL6:.*]] = mo.static.broadcast_to(%[[RNG2]]) : !mo.tensor<[[[BS]], 1], si64> -> !mo.tensor<[[[BS]], [[W]]], si64>
  // CHECK-DAG: %[[WHERETP:.*]] = mo.transpose(%[[VAL5]], %[[PERM]]) : (!mo.tensor<[[[W]], [[BS]]], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[BS]], [[W]]], si64>
  // CHECK-DAG: %[[RES:.*]] = mo.add(%[[WHERETP]], %[[VAL6]]) : !mo.tensor<[[[BS]], [[W]]], si64>
  // CHECK-DAG: kgen.param.declare [[K:.*]] = <mul_nuw([[A]], [[B]], [[W]])>
  // CHECK-DAG: %[[OLDRES:.*]] = mo.static.reshape(%[[RES]]) : !mo.tensor<[[[BS]], [[W]]], si64> -> !mo.tensor<[[[K]], 1], si64>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[2, A], f32> -> !mo.tensor<[B, A, 2, A], f32>
  kgen.param.declare D = <mul_nuw(B, A, 2, A)>
  %1 = mo.static.reshape(%0) : !mo.tensor<[B, A, 2, A], f32> -> !mo.tensor<[D], f32>
  %2 = mo.arg_nonzero<() -> K>(%1) : (!mo.tensor<[D], f32>) -> !mo.tensor<[K, 1], si64>
  // CHECK: mo.output<[[K]]> %[[OLDRES]] : !mo.tensor<[[[K]], 1], si64>
  mo.output<K> %2 : !mo.tensor<[K, 1], si64>
}

// CHECK: mo.graph @non_flat_input<() -> M>(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[M, 2], si64> no_inline {
mo.graph @non_flat_input<() -> M>(%arg0: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[M, 2], si64> {
  // CHECK-NEXT: %[[BCASTARG0:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[2, 3], f32> -> !mo.tensor<[5, 2, 3], f32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = mo.static.reshape(%[[BCASTARG0]]) : !mo.tensor<[5, 2, 3], f32> -> !mo.tensor<[5, 6], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[2, 3], f32> -> !mo.tensor<[5, 2, 3], f32>
  %1 = mo.static.reshape(%0) : !mo.tensor<[5, 2, 3], f32> -> !mo.tensor<[5, 6], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.arg_nonzero<() -> [[K:.*]]>(%[[RESHAPE]]) : (!mo.tensor<[5, 6], f32>) -> !mo.tensor<[[[K]], 2], si64>
  %2 = mo.arg_nonzero<() -> K>(%1) : (!mo.tensor<[5, 6], f32>) -> !mo.tensor<[K, 2], si64>
  // CHECK-NEXT: mo.output<[[K]]> %[[RES]] : !mo.tensor<[[[K]], 2], si64>
  mo.output<K> %2 : !mo.tensor<[K, 2], si64>
}

// CHECK: mo.graph @no_reshape<[[A:.*]], [[B:.*]] -> [[M:.*]]>(%[[ARG0:.*]]: !mo.tensor<[[[A]]], f32>) -> !mo.tensor<[[[M]], 1], si64> no_inline {
mo.graph @no_reshape<A, B -> M>(%arg0: !mo.tensor<[A], f32>) -> !mo.tensor<[M, 1], si64> {
  // CHECK-NEXT: %[[BCASTARG0:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[[[A]]], f32> -> !mo.tensor<[[[B]]], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A], f32> -> !mo.tensor<[B], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.arg_nonzero<() -> [[K:.*]]>(%[[BCASTARG0]]) : (!mo.tensor<[[[B]]], f32>) -> !mo.tensor<[[[K]], 1], si64>
  %1 = mo.arg_nonzero<() -> K>(%0) : (!mo.tensor<[B], f32>) -> !mo.tensor<[K, 1], si64>
  // CHECK-NEXT: mo.output<[[K]]> %[[RES]] : !mo.tensor<[[[K]], 1], si64>
  mo.output<K> %1 : !mo.tensor<[K, 1], si64>
}

// CHECK: mo.graph @reshape_dynamic_input<[[A:.*]], [[SH0:.*]]: !mosh.ape -> [[M:.*]]>(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[[[M]], 1], si64> no_inline {
mo.graph @reshape_dynamic_input<A, Sh0: !mosh.ape -> M>(%arg0: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[M, 1], si64> {
  // CHECK-NEXT: %[[BCASTARG0:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[2, 3], f32> -> !mo.tensor<[[SH0]], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[2, 3], f32> -> !mo.tensor<Sh0, f32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = mo.static.reshape(%[[BCASTARG0]]) : !mo.tensor<[[SH0]], f32> -> !mo.tensor<[[[A]]], f32>
  %1 = mo.static.reshape(%0) : !mo.tensor<Sh0, f32> -> !mo.tensor<[A], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.arg_nonzero<() -> [[K:.*]]>(%[[RESHAPE]]) : (!mo.tensor<[[[A]]], f32>) -> !mo.tensor<[[[K]], 1], si64>
  %2 = mo.arg_nonzero<() -> K>(%1) : (!mo.tensor<[A], f32>) -> !mo.tensor<[K, 1], si64>
  // CHECK-NEXT: mo.output<[[K]]> %[[RES]] : !mo.tensor<[[[K]], 1], si64>
  mo.output<K> %2 : !mo.tensor<[K, 1], si64>
}

// CHECK: mo.graph @bcast_dynamic_input<[[A:.*]], [[B:.*]], [[SH0:.*]]: !mosh.ape -> [[M:.*]]>(
// CHECk-SAME: %[[ARG0]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[[M]], 1], si64> no_inline {
mo.graph @bcast_dynamic_input<A, B, Sh0: !mosh.ape -> M>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<[M, 1], si64> {
  // CHECK-NEXT: %[[BCASTARG0:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[[SH0]], f32> -> !mo.tensor<[[[B]], [[B]]], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<Sh0, f32> -> !mo.tensor<[B, B], f32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = mo.static.reshape(%[[BCASTARG0]]) : !mo.tensor<[[[B]], [[B]]], f32> -> !mo.tensor<[[[A]]], f32>
  %1 = mo.static.reshape(%0) : !mo.tensor<[B, B], f32> -> !mo.tensor<[A], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.arg_nonzero<() -> [[K:.*]]>(%[[RESHAPE]]) : (!mo.tensor<[[[A]]], f32>) -> !mo.tensor<[[[K]], 1], si64>
  %2 = mo.arg_nonzero<() -> K>(%1) : (!mo.tensor<[A], f32>) -> !mo.tensor<[K, 1], si64>
  // CHECK-NEXT: mo.output<[[K]]> %[[RES]] : !mo.tensor<[[[K]], 1], si64>
  mo.output<K> %2 : !mo.tensor<[K, 1], si64>
}
