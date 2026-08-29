// RUN: mo-opt %s --pass-pipeline='builtin.module(mo.graph(map.iter.opaque(map-loop-invariant-view-motion)))' | FileCheck %s

// LoopInvariantViewMotion hoists a map.index.transpose chain feeding a
// map.load out of the loop body into a map.index.transpose.stride op in the
// enclosing map.iter.opaque, rewriting the load to use the loop's own
// (untransformed) index directly -- same generic IndexTransformOpInterface
// mechanism already exercised by map.index.slice/map.index.broadcast, no
// pass changes needed.

module {

// 2D swap: out.dim[0] = in.dim[1], out.dim[1] = in.dim[0].

// CHECK-LABEL: mo.graph @transposeSwap
// CHECK:         [[RESULT:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[INPUT:%[^ :,]+]]: !mo.tensor<[2, 3], f32>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[3, 2], f32>) {
// CHECK-NEXT:      [[STRIDED:%[^ ]+]] = map.index.transpose.stride<#map.index.transpose.spec<[2, 3] -> [3, 2], perm = [1, 0]>> ([[INPUT]])
// CHECK-NEXT:      map.iter.foreach.body<[3, 2]>
// CHECK-NEXT:        index   ([[IDX:%[^ :,]+]]: !map.index<2>) {
// CHECK-NOT:       map.index.transpose
// CHECK-NEXT:      [[VAL:%[^ ]+]] = map.load [[STRIDED]]{{\[}}[[IDX]]] : !mo.tensor<[3, 2], f32>
// CHECK-NEXT:      map.store [[OUT]]{{\[}}[[IDX]]] = [[VAL]] : !mo.tensor<[3, 2], f32>
// CHECK-NEXT:    }
mo.graph @transposeSwap(%input: !mo.tensor<[2, 3], f32>)
    -> !mo.tensor<[3, 2], f32> {
  %output = map.iter.opaque
      inputs (%input : !mo.tensor<[2, 3], f32>)
      outputs (%out : !mo.tensor<[3, 2], f32>) {
    map.iter.foreach.body<[3, 2]>
        index (%idx: !map.index<2>) {
      %idx_ = map.index.transpose<#map.index.transpose.spec<[2, 3] -> [3, 2], perm = [1, 0]>> (%idx)
          : !map.index<2> -> !map.index<2>
      %val = map.load %input[%idx_] : !mo.tensor<[2, 3], f32>
      map.store %out[%idx] = %val : !mo.tensor<[3, 2], f32>
    }
  }
  mo.output %output : !mo.tensor<[3, 2], f32>
}

// 3D permutation: out.dim[i] = in.dim[perm[i]] for perm = [2, 0, 1].

// CHECK-LABEL: mo.graph @transpose3D
// CHECK:         [[RESULT2:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[INPUT2:%[^ :,]+]]: !mo.tensor<[2, 3, 4], f32>)
// CHECK-NEXT:    outputs ([[OUT2:%[^ :,]+]] : !mo.tensor<[4, 2, 3], f32>) {
// CHECK-NEXT:      [[STRIDED2:%[^ ]+]] = map.index.transpose.stride<#map.index.transpose.spec<[2, 3, 4] -> [4, 2, 3], perm = [2, 0, 1]>> ([[INPUT2]])
// CHECK-NEXT:      map.iter.foreach.body<[4, 2, 3]>
// CHECK-NEXT:        index   ([[IDX2:%[^ :,]+]]: !map.index<3>) {
// CHECK-NOT:       map.index.transpose
// CHECK-NEXT:      [[VAL2:%[^ ]+]] = map.load [[STRIDED2]]{{\[}}[[IDX2]]] : !mo.tensor<[4, 2, 3], f32>
// CHECK-NEXT:      map.store [[OUT2]]{{\[}}[[IDX2]]] = [[VAL2]] : !mo.tensor<[4, 2, 3], f32>
// CHECK-NEXT:    }
mo.graph @transpose3D(%input: !mo.tensor<[2, 3, 4], f32>)
    -> !mo.tensor<[4, 2, 3], f32> {
  %output = map.iter.opaque
      inputs (%input : !mo.tensor<[2, 3, 4], f32>)
      outputs (%out : !mo.tensor<[4, 2, 3], f32>) {
    map.iter.foreach.body<[4, 2, 3]>
        index (%idx: !map.index<3>) {
      %idx_ = map.index.transpose<#map.index.transpose.spec<[2, 3, 4] -> [4, 2, 3], perm = [2, 0, 1]>> (%idx)
          : !map.index<3> -> !map.index<3>
      %val = map.load %input[%idx_] : !mo.tensor<[2, 3, 4], f32>
      map.store %out[%idx] = %val : !mo.tensor<[4, 2, 3], f32>
    }
  }
  mo.output %output : !mo.tensor<[4, 2, 3], f32>
}

}
