// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers | FileCheck %s

// A dtype-changing cast fused into a matmul epilogue (mixed precision): the
// kernel accumulates in f32, the fused cast converts to si32 on the way out.
// `AllocateAccumulatingBuffers` adds a pre-cast (f32) scratch buffer as an extra
// kernel result and points the epilogue's `$intermediateTensor` at it, so the
// accumulation goes through f32 storage while the real output stays si32.

// CHECK-LABEL: mo.graph @matmul_cast
// The kernel gains a second (scratch, f32) output alongside the si32 output.
// CHECK: map.iter.opaque
// CHECK: outputs (%{{.*}} : !mo.tensor<[2, 2], si32>, %{{.*}} : !mo.tensor<[2, 2], f32>)
// The epilogue captures the si32 output but stores through the f32 scratch
// (its `intermediate` operand), and the store itself still writes the si32 value.
// CHECK: map.iter.epilogue
// CHECK-SAME: intermediate %{{.*}} : !mo.tensor<[2, 2], f32>
mo.graph @matmul_cast(
  %a: !mo.tensor<[2, 2], f32>,
  %b: !mo.tensor<[2, 2], f32>
) -> !mo.tensor<[2, 2], si32> {
  %mm = mo.matmul(%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.cast(%mm) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], si32>
  mo.output %c : !mo.tensor<[2, 2], si32>
}
