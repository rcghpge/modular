// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-prologue --map-to-mogg | FileCheck %s

// A dynamic index reshape in a PROLOGUE gets its runtime from/to shapes the
// same way the epilogue's store-side one does: the lowering builds them, hands
// them to `mogg.tensor.create.prologue` as `shapes(...)`, and they arrive as
// block args trailing the tensor captures. `mogg.index.reshape` then reads them
// as `$dynamicShapes`.
//
// `[D, 2] -> [2, D]` re-linearizes across `D`, so this is the case that needs
// them: with `$dynamicShapes` empty, emit takes its static path and
// de-linearizes against placeholder dims with no diagnostic. Numerics are
// covered by `max/tests/integration/graph/test_reshape_prologue_fusion.py`.

// CHECK-LABEL: mo.graph @dyn_reshape_into_prologue
// CHECK: mogg.tensor.create.prologue %{{.*}} capturing(%{{.*}}, %{{.*}} : !mogg.tensor, !mogg.tensor) shapes(%{{.*}}, %{{.*}} : !mogg.shape, !mogg.shape)
// The shape captures trail the tensor captures, after the index -- and the
// reshape reads those block args rather than falling back to the static shape.
// CHECK: ^bb0(%[[IDX:.*]]: !map.index<2>, %{{.*}}: !mogg.tensor, %{{.*}}: !mogg.tensor, %[[FROMARG:.*]]: !mogg.shape, %[[TOARG:.*]]: !mogg.shape):
// CHECK:   mogg.index.reshape(%[[IDX]], %[[FROMARG]], %[[TOARG]]) <fromShape = #mosh<ape[2, D]> : !mosh.ape, toShape = #mosh<ape[D, 2]> : !mosh.ape>
// CHECK: mogg.call.opaque @{{.*}}MapLowerPartialFusedInputs {{.*}}hasPrologueFusion = true
mo.graph @dyn_reshape_into_prologue<D>(
  %x: !mo.tensor<[D, 2], f32>,
  %y: !mo.tensor<[2, D], f32>
) -> !mo.tensor<[2, D], f32> {
  %r = mo.static.reshape(%x) : !mo.tensor<[D, 2], f32> -> !mo.tensor<[2, D], f32>
  %s = mo.add(%r, %y) : !mo.tensor<[2, D], f32>
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_partial_fused_inputs"}(%s, %y) : (!mo.tensor<[2, D], f32>, !mo.tensor<[2, D], f32>) -> !mo.tensor<[2, D], f32>
  mo.output %c : !mo.tensor<[2, D], f32>
}
