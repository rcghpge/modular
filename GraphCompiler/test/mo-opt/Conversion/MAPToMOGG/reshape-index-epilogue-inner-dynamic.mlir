// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers --map-to-mogg | FileCheck %s

// A reshape with an INNER dynamic dim fused into a fused-output kernel's
// epilogue. `[D, 2] -> [2, D]` re-linearizes across the dynamic dim `D` (the
// store's unravel needs `D` for its modulo), so the store-index transform
// cannot be evaluated from compile-time shape alone. Its runtime from/to
// `!mogg.shape`s are threaded into the epilogue as trailing SHAPE CAPTURES and
// wired onto the in-body `mogg.index.reshape` as its dynamic-shape operands.

// CHECK-LABEL: mo.graph @custom_reshape_inner_dynamic
// CHECK: mogg._tensor.get_ptr
// CHECK: mogg._tensor.create.epilogue
// CHECK-SAME: shapes(%[[FROM:.*]], %[[TO:.*]])
// The epilogue body carries the two shape captures as trailing block args after
// the tensor captures; the store-index reshape reads them as dynamic shapes.
// CHECK: ^bb0(%[[V:.*]]: f32, %[[I:.*]]: !map.index<2>, %[[CAP0:.*]]: !mogg.tensor_, %[[CAP1:.*]]: !mogg.tensor_, %[[SFROM:.*]]: !mogg.shape, %[[STO:.*]]: !mogg.shape):
// CHECK: %[[SIDX:.*]] = mogg.index.reshape(%[[I]], %[[SFROM]], %[[STO]]) {fromShape = {{.*}}, toShape = {{.*}}} : (!map.index<2>, !mogg.shape, !mogg.shape) -> !map.index<2>
// CHECK: mogg._tensor.store %[[CAP0]][%[[SIDX]]]
mo.graph @custom_reshape_inner_dynamic<D>(
  %x: !mo.tensor<[D, 2], f32>,
  %z: !mo.tensor<[2, D], f32>
) -> !mo.tensor<[2, D], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_fused_output"}(%x) : (!mo.tensor<[D, 2], f32>) -> !mo.tensor<[D, 2], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[D, 2], f32> -> !mo.tensor<[2, D], f32>
  %out = mo.add(%r, %z) : !mo.tensor<[2, D], f32>
  mo.output %out : !mo.tensor<[2, D], f32>
}
