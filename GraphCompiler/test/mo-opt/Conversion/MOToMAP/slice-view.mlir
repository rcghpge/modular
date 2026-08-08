// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages --mo-to-map | FileCheck %s

// Verify that mo.slice lowers to a standalone (unfused) map.iter.foreach over
// the output shape, with a map.index.slice op computing the input index --
// not map.iter.opaque, and not the "view kernels are not yet supported"
// error. Each of starts/stops/steps independently shows up as a literal in
// the map.index.slice.spec when it's a compile-time constant (a mo.constant
// operand), and as `?` (ShapedType::kDynamic) when it isn't.

// All of starts/stops/steps are mo.constant: every bound is a literal.

// CHECK-LABEL: mo.graph @sliceStaticBounds
// CHECK-NOT:     map.iter.opaque
// CHECK:         [[OUT:%.*]] = map.iter.foreach<[3, 32]>
// CHECK:         inputs  ([[INPUT:%.*]]: !mo.tensor<[10, 64], f32>, [[STARTS:%.*]]: !mo.tensor<[2], si64>, [[STOPS:%.*]]: !mo.tensor<[2], si64>, [[STEPS:%.*]]: !mo.tensor<[2], si64>)
// CHECK:         outputs ([[OUTARG:%.*]]: !mo.tensor<[3, 32], f32>)
// CHECK:         index   ([[IDX:%.*]]: !map.index<2>) -> !mo.tensor<[3, 32], f32> {
// CHECK:         [[IDX_:%.*]] = map.index.slice<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>> ([[IDX]], [[STARTS]], [[STOPS]], [[STEPS]])
// CHECK:         [[VAL:%.*]] = map.load [[INPUT]]{{\[}}[[IDX_]]] : !mo.tensor<[10, 64], f32>
// CHECK:         map.store [[OUTARG]]{{\[}}[[IDX]]] = [[VAL]] : !mo.tensor<[3, 32], f32>
mo.graph @sliceStaticBounds(%input: !mo.tensor<[10, 64], f32>) -> !mo.tensor<[3, 32], f32> {
  %start = mo.constant {value = #M.dense_array<1, 32> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop = mo.constant {value = #M.dense_array<4, 64> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32], f32>
  mo.output %res : !mo.tensor<[3, 32], f32>
}

// All of starts/stops/steps come from graph arguments (not mo.constant):
// every bound is dynamic (`?`).

// CHECK-LABEL: mo.graph @sliceDynamicBounds
// CHECK-NOT:     map.iter.opaque
// CHECK:         map.iter.foreach<[3, 32]>
// CHECK:         map.index.slice<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [?, ?], stops = [?, ?], steps = [?, ?]>>
mo.graph @sliceDynamicBounds(
    %input: !mo.tensor<[10, 64], f32>,
    %start: !mo.tensor<[2], si64>,
    %stop: !mo.tensor<[2], si64>,
    %step: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 32], f32> {
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32], f32>
  mo.output %res : !mo.tensor<[3, 32], f32>
}

// starts/stops are mo.constant but step is a graph argument: only the steps
// field of the spec is dynamic, proving each of starts/stops/steps is
// tracked independently (a non-constant step doesn't force starts/stops to
// also be treated as dynamic).

// CHECK-LABEL: mo.graph @sliceNonConstantStep
// CHECK-NOT:     map.iter.opaque
// CHECK:         map.iter.foreach<[3, 32]>
// CHECK:         map.index.slice<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [?, ?]>>
mo.graph @sliceNonConstantStep(
    %input: !mo.tensor<[10, 64], f32>,
    %step: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 32], f32> {
  %start = mo.constant {value = #M.dense_array<1, 32> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop = mo.constant {value = #M.dense_array<4, 64> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32], f32>
  mo.output %res : !mo.tensor<[3, 32], f32>
}
