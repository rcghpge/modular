// RUN: mo-opt %s --pass-pipeline='builtin.module(mo.graph(map.iter.opaque(map-loop-invariant-view-motion)))' | FileCheck %s

// LoopInvariantViewMotion hoists a map.index.broadcast chain feeding a
// map.load out of the loop body into a map.index.broadcast.stride op in the
// enclosing map.iter.opaque, rewriting the load to use the loop's own
// (untransformed) index directly -- same generic IndexTransformOpInterface
// mechanism already exercised by map.index.slice, no pass changes needed.
// Covers a rank-preserving case and a rank-expanding one (numpy semantics).

module {

// Rank-preserving: in-shape dim 0 (size 1) broadcasts to 8.

// CHECK-LABEL: mo.graph @broadcastSameRank
// CHECK:         [[RESULT:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[INPUT:%[^ :,]+]]: !mo.tensor<[1, 64], f32>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[8, 64], f32>) {
// CHECK-NEXT:      [[STRIDED:%[^ ]+]] = map.index.broadcast.stride<#map.index.broadcast.spec<[1, 64] -> [8, 64]>> ([[INPUT]])
// CHECK-NEXT:      map.iter.foreach.body<[8, 64]>
// CHECK-NEXT:        index   ([[IDX:%[^ :,]+]]: !map.index<2>) {
// CHECK-NOT:       map.index.broadcast
// CHECK-NEXT:      [[VAL:%[^ ]+]] = map.load [[STRIDED]]{{\[}}[[IDX]]] : !mo.tensor<[8, 64], f32>
// CHECK-NEXT:      map.store [[OUT]]{{\[}}[[IDX]]] = [[VAL]] : !mo.tensor<[8, 64], f32>
// CHECK-NEXT:    }
mo.graph @broadcastSameRank(%input: !mo.tensor<[1, 64], f32>)
    -> !mo.tensor<[8, 64], f32> {
  %output = map.iter.opaque
      inputs (%input : !mo.tensor<[1, 64], f32>)
      outputs (%out : !mo.tensor<[8, 64], f32>) {
    map.iter.foreach.body<[8, 64]>
        index (%idx: !map.index<2>) {
      %idx_ = map.index.broadcast<#map.index.broadcast.spec<[1, 64] -> [8, 64]>> (%idx)
          : !map.index<2> -> !map.index<2>
      %val = map.load %input[%idx_] : !mo.tensor<[1, 64], f32>
      map.store %out[%idx] = %val : !mo.tensor<[8, 64], f32>
    }
  }
  mo.output %output : !mo.tensor<[8, 64], f32>
}

// Rank-expanding: in has one fewer dim than out; the hoisted view still has
// the *output* shape/rank (like every other IndexTransformOpInterface
// hoist), even though the pre-hoist projected index had a smaller rank.

// CHECK-LABEL: mo.graph @broadcastRankExpanding
// CHECK:         [[RESULT2:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[INPUT2:%[^ :,]+]]: !mo.tensor<[64], f32>)
// CHECK-NEXT:    outputs ([[OUT2:%[^ :,]+]] : !mo.tensor<[8, 64], f32>) {
// CHECK-NEXT:      [[STRIDED2:%[^ ]+]] = map.index.broadcast.stride<#map.index.broadcast.spec<[64] -> [8, 64]>> ([[INPUT2]])
// CHECK-NEXT:      map.iter.foreach.body<[8, 64]>
// CHECK-NEXT:        index   ([[IDX2:%[^ :,]+]]: !map.index<2>) {
// CHECK-NOT:       map.index.broadcast
// CHECK-NEXT:      [[VAL2:%[^ ]+]] = map.load [[STRIDED2]]{{\[}}[[IDX2]]] : !mo.tensor<[8, 64], f32>
// CHECK-NEXT:      map.store [[OUT2]]{{\[}}[[IDX2]]] = [[VAL2]] : !mo.tensor<[8, 64], f32>
// CHECK-NEXT:    }
mo.graph @broadcastRankExpanding(%input: !mo.tensor<[64], f32>)
    -> !mo.tensor<[8, 64], f32> {
  %output = map.iter.opaque
      inputs (%input : !mo.tensor<[64], f32>)
      outputs (%out : !mo.tensor<[8, 64], f32>) {
    map.iter.foreach.body<[8, 64]>
        index (%idx: !map.index<2>) {
      %idx_ = map.index.broadcast<#map.index.broadcast.spec<[64] -> [8, 64]>> (%idx)
          : !map.index<2> -> !map.index<1>
      %val = map.load %input[%idx_] : !mo.tensor<[64], f32>
      map.store %out[%idx] = %val : !mo.tensor<[8, 64], f32>
    }
  }
  mo.output %output : !mo.tensor<[8, 64], f32>
}

}
