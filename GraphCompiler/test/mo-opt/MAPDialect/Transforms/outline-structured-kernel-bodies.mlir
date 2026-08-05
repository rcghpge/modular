// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --map-outline-structured-kernel-bodies | FileCheck %s

module {

// CHECK-LABEL: mo.graph @add
// CHECK:         [[RESULT:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[X:%[^ :,]+]] : !mo.tensor<[256, 256], f32>, [[Y:%[^ :,]+]] : !mo.tensor<[256, 256], f32>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[256, 256], f32>) {
// CHECK-NEXT:      map.iter.foreach.body<[256, 256]>
// CHECK-NEXT:        index   ([[IDX:%[^ :,]+]]: !map.index<2>) {
// CHECK-NEXT:      [[XV:%[^ ]+]] = map.load [[X]][[[IDX]]] : !mo.tensor<[256, 256], f32>
// CHECK-NEXT:      [[YV:%[^ ]+]] = map.load [[Y]][[[IDX]]] : !mo.tensor<[256, 256], f32>
// CHECK-NEXT:      [[RES:%[^ ]+]] = map.call.elementwise @builtin_kernels.elementwise.Add([[IDX]], [[XV]], [[YV]]) : !map.index<2>, f32, f32 -> f32
// CHECK-NEXT:      map.store [[OUT]][[[IDX]]] = [[RES]] : !mo.tensor<[256, 256], f32>
// CHECK-NEXT:    }
// CHECK-NEXT:  } {fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.add">>}
// CHECK-NEXT:  mo.output [[RESULT]] : !mo.tensor<[256, 256], f32>
mo.graph @add(%arg0: !mo.tensor<[256, 256], f32>, %arg1: !mo.tensor<[256, 256], f32>) -> !mo.tensor<[256, 256], f32> {
  %0 = map.iter.foreach<[256, 256]>
      inputs  (%arg0: !mo.tensor<[256, 256], f32>, %arg1: !mo.tensor<[256, 256], f32>)
      outputs (%out: !mo.tensor<[256, 256], f32>)
      index   (%idx: !map.index<2>)
      -> !mo.tensor<[256, 256], f32>
  {
    %xv = map.load %arg0[%idx] : !mo.tensor<[256, 256], f32>
    %yv = map.load %arg1[%idx] : !mo.tensor<[256, 256], f32>
    %outv = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %xv, %yv)
                : !map.index<2>, f32, f32 -> f32
    map.store %out[%idx] = %outv : !mo.tensor<[256, 256], f32>
  } {fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.add">>, hasReadEffect = false, hasWriteEffect = false}
  mo.output %0 : !mo.tensor<[256, 256], f32>
}

// CHECK-LABEL: mo.graph @addDynamic
// CHECK:         [[RESULT2:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[X2:%[^ :,]+]] : !mo.tensor<[M, N], f32>, [[Y2:%[^ :,]+]] : !mo.tensor<[M, N], f32>)
// CHECK-NEXT:    outputs ([[OUT2:%[^ :,]+]] : !mo.tensor<[M, N], f32>) {
// CHECK-NEXT:      map.iter.foreach.body<[M, N]>
// CHECK-NEXT:        index   ([[IDX2:%[^ :,]+]]: !map.index<2>) {
// CHECK-NEXT:      [[XV2:%[^ ]+]] = map.load [[X2]][[[IDX2]]] : !mo.tensor<[M, N], f32>
// CHECK-NEXT:      [[YV2:%[^ ]+]] = map.load [[Y2]][[[IDX2]]] : !mo.tensor<[M, N], f32>
// CHECK-NEXT:      [[RES2:%[^ ]+]] = map.call.elementwise @builtin_kernels.elementwise.Add([[IDX2]], [[XV2]], [[YV2]]) : !map.index<2>, f32, f32 -> f32
// CHECK-NEXT:      map.store [[OUT2]][[[IDX2]]] = [[RES2]] : !mo.tensor<[M, N], f32>
// CHECK-NEXT:    }
// CHECK-NEXT:  } {fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.add">>}
// CHECK-NEXT:  mo.output [[RESULT2]] : !mo.tensor<[M, N], f32>
mo.graph @addDynamic<M, N>(%arg0: !mo.tensor<[M, N], f32>, %arg1: !mo.tensor<[M, N], f32>) -> !mo.tensor<[M, N], f32> {
  %0 = map.iter.foreach<[M, N]>
      inputs  (%arg0: !mo.tensor<[M, N], f32>, %arg1: !mo.tensor<[M, N], f32>)
      outputs (%out: !mo.tensor<[M, N], f32>)
      index   (%idx: !map.index<2>)
      -> !mo.tensor<[M, N], f32>
  {
    %xv = map.load %arg0[%idx] : !mo.tensor<[M, N], f32>
    %yv = map.load %arg1[%idx] : !mo.tensor<[M, N], f32>
    %outv = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %xv, %yv)
                : !map.index<2>, f32, f32 -> f32
    map.store %out[%idx] = %outv : !mo.tensor<[M, N], f32>
  } {fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.add">>, hasReadEffect = false, hasWriteEffect = false}
  mo.output %0 : !mo.tensor<[M, N], f32>
}

// map.iter.opaque is left untouched.
// CHECK-LABEL: mo.graph @alreadyOpaque
// CHECK:         [[RESULT3:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[X3:%[^ :,]+]] : !mo.tensor<[2, 2], f32>)
// CHECK-NEXT:    outputs ([[OUT3:%[^ :,]+]] : !mo.tensor<[2, 2], f32>) {
// CHECK-NEXT:  }
// CHECK-NEXT:  mo.output [[RESULT3]] : !mo.tensor<[2, 2], f32>
mo.graph @alreadyOpaque(%arg0: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %0 = map.iter.opaque inputs (%arg0 : !mo.tensor<[2, 2], f32>) outputs (%out : !mo.tensor<[2, 2], f32>) {
  }
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

}
