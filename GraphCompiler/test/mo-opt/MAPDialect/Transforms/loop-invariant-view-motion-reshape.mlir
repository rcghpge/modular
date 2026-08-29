// RUN: mo-opt %s --pass-pipeline='builtin.module(mo.graph(map.iter.opaque(map-loop-invariant-view-motion)))' | FileCheck %s

// LoopInvariantViewMotion hoists a map.index.reshape chain feeding a
// map.load out of the loop body into a map.index.reshape.stride op in the
// enclosing map.iter.opaque, rewriting the load to use the loop's own
// (untransformed) index directly -- same generic IndexTransformOpInterface
// mechanism already exercised by slice/broadcast/transpose, no pass changes
// needed. Covers a rank-reducing case and a rank-expanding one.

module {

// Rank-reducing: flatten [4, 6] into [24].

// CHECK-LABEL: mo.graph @reshapeFlatten
// CHECK:         [[RESULT:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[INPUT:%[^ :,]+]]: !mo.tensor<[4, 6], f32>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[24], f32>) {
// CHECK-NEXT:      [[STRIDED:%[^ ]+]] = map.index.reshape.stride<#map.index.reshape.spec<[4, 6] -> [24]>> ([[INPUT]])
// CHECK-NEXT:      map.iter.foreach.body<[24]>
// CHECK-NEXT:        index   ([[IDX:%[^ :,]+]]: !map.index<1>) {
// CHECK-NOT:       map.index.reshape
// CHECK-NEXT:      [[VAL:%[^ ]+]] = map.load [[STRIDED]]{{\[}}[[IDX]]] : !mo.tensor<[24], f32>
// CHECK-NEXT:      map.store [[OUT]]{{\[}}[[IDX]]] = [[VAL]] : !mo.tensor<[24], f32>
// CHECK-NEXT:    }
mo.graph @reshapeFlatten(%input: !mo.tensor<[4, 6], f32>)
    -> !mo.tensor<[24], f32> {
  %output = map.iter.opaque
      inputs (%input : !mo.tensor<[4, 6], f32>)
      outputs (%out : !mo.tensor<[24], f32>) {
    map.iter.foreach.body<[24]>
        index (%idx: !map.index<1>) {
      %idx_ = map.index.reshape<#map.index.reshape.spec<[4, 6] -> [24]>> (%idx)
          : !map.index<1> -> !map.index<2>
      %val = map.load %input[%idx_] : !mo.tensor<[4, 6], f32>
      map.store %out[%idx] = %val : !mo.tensor<[24], f32>
    }
  }
  mo.output %output : !mo.tensor<[24], f32>
}

// Rank-expanding: unflatten [24] into [4, 6].

// CHECK-LABEL: mo.graph @reshapeExpand
// CHECK:         [[RESULT2:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[INPUT2:%[^ :,]+]]: !mo.tensor<[24], f32>)
// CHECK-NEXT:    outputs ([[OUT2:%[^ :,]+]] : !mo.tensor<[4, 6], f32>) {
// CHECK-NEXT:      [[STRIDED2:%[^ ]+]] = map.index.reshape.stride<#map.index.reshape.spec<[24] -> [4, 6]>> ([[INPUT2]])
// CHECK-NEXT:      map.iter.foreach.body<[4, 6]>
// CHECK-NEXT:        index   ([[IDX2:%[^ :,]+]]: !map.index<2>) {
// CHECK-NOT:       map.index.reshape
// CHECK-NEXT:      [[VAL2:%[^ ]+]] = map.load [[STRIDED2]]{{\[}}[[IDX2]]] : !mo.tensor<[4, 6], f32>
// CHECK-NEXT:      map.store [[OUT2]]{{\[}}[[IDX2]]] = [[VAL2]] : !mo.tensor<[4, 6], f32>
// CHECK-NEXT:    }
mo.graph @reshapeExpand(%input: !mo.tensor<[24], f32>)
    -> !mo.tensor<[4, 6], f32> {
  %output = map.iter.opaque
      inputs (%input : !mo.tensor<[24], f32>)
      outputs (%out : !mo.tensor<[4, 6], f32>) {
    map.iter.foreach.body<[4, 6]>
        index (%idx: !map.index<2>) {
      %idx_ = map.index.reshape<#map.index.reshape.spec<[24] -> [4, 6]>> (%idx)
          : !map.index<2> -> !map.index<1>
      %val = map.load %input[%idx_] : !mo.tensor<[24], f32>
      map.store %out[%idx] = %val : !mo.tensor<[4, 6], f32>
    }
  }
  mo.output %output : !mo.tensor<[4, 6], f32>
}

}
