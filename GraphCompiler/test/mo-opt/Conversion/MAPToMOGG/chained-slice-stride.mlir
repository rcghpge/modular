// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --pass-pipeline='builtin.module(mogg-import-packages, map-to-mogg)' | FileCheck %s

// Two map.index.slice.stride ops in the same kernel body, the second's
// $input being the first's own result: verifies mogg._tensor.create.slice
// composes when its input is itself a slice view, and that
// mogg._call.foreach only captures the innermost view and the output (the
// two values the loop body actually references).

// CHECK-LABEL: mo.graph @chainedSliceStride
// CHECK:       [[INNER:%[^ ]+]] = mogg._tensor.create.slice([[IT0:%[^ ]+]], {{.*}}) {{.*}}outShape = #mosh<ape[4, 32]>{{.*}} : (!mogg.tensor_, !mogg.tensor_, !mogg.tensor_, !mogg.tensor_) -> !mogg.tensor_
// CHECK-NEXT:  [[OUTER:%[^ ]+]] = mogg._tensor.create.slice([[INNER]], {{.*}}) {{.*}}outShape = #mosh<ape[2, 16]>{{.*}} : (!mogg.tensor_, !mogg.tensor_, !mogg.tensor_, !mogg.tensor_) -> !mogg.tensor_
// CHECK:       mogg._call.foreach @builtin_primitives.primitives._foreach shape({{%[^ ]+}}) context({{%[^ ]+}}) capturing([[OUTER]], [[OT:%[^ ]+]] : !mogg.tensor_, !mogg.tensor_) attributes {device = #M.device_ref<"cpu", 0>} {
// CHECK-NEXT:  ^bb0([[ARG:%[^ :,]+]]: !mogg.tensor_, [[OUTARG:%[^ :,]+]]: !mogg.tensor_, [[IDX:%[^ :,]+]]: !map.index<2>):
// CHECK-NEXT:    [[VAL:%[^ ]+]] = mogg._tensor.load [[ARG]]{{\[}}[[IDX]]] : !mogg.tensor_, !map.index<2> -> f32
// CHECK-NEXT:    mogg._tensor.store [[OUTARG]]{{\[}}[[IDX]]] = [[VAL]] : !mogg.tensor_, !map.index<2>, f32
mo.graph @chainedSliceStride(%input: !mo.tensor<[10, 64], f32>) -> !mo.tensor<[2, 16], f32> {
  %start0 = mo.constant {value = #M.dense_array<1, 16> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop0 = mo.constant {value = #M.dense_array<5, 48> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step0 = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %start1 = mo.constant {value = #M.dense_array<1, 8> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop1 = mo.constant {value = #M.dense_array<3, 24> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step1 = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = map.iter.opaque
    inputs (%input: !mo.tensor<[10, 64], f32>, %start0: !mo.tensor<[2], si64>, %stop0: !mo.tensor<[2], si64>, %step0: !mo.tensor<[2], si64>, %start1: !mo.tensor<[2], si64>, %stop1: !mo.tensor<[2], si64>, %step1: !mo.tensor<[2], si64>)
    outputs (%out: !mo.tensor<[2, 16], f32>) {
    %view1 = map.index.slice.stride<#map.index.slice.spec<[10, 64] -> [4, 32], starts = [1, 16], stops = [5, 48], steps = [1, 1]>> (%input, %start0, %stop0, %step0) : (!mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 32], f32>
    %view2 = map.index.slice.stride<#map.index.slice.spec<[4, 32] -> [2, 16], starts = [1, 8], stops = [3, 24], steps = [1, 1]>> (%view1, %start1, %stop1, %step1) : (!mo.tensor<[4, 32], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 16], f32>
    map.iter.foreach.body<[2, 16]>
      index (%idx: !map.index<2>)
      capturing (%view20 = %view2 : !mo.tensor<[2, 16], f32>, %out0 = %out : !mo.tensor<[2, 16], f32>) {
      %val = map.load %view20[%idx] : !mo.tensor<[2, 16], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[2, 16], f32>
    }
  }
  mo.output %res : !mo.tensor<[2, 16], f32>
}
