// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-prologue --map-to-mogg | FileCheck %s

// The prologue builds its shape from the dynamic dim.
// CHECK-LABEL: mo.graph @dyn_fused_producer
// CHECK: mogg.kernel
// CHECK: dims (%[[D:.*]]: !mogg.dim<batch>)
// CHECK: mogg.shape.create(%[[D]])
// CHECK-SAME: <shape = #mosh<ape[dyn, 4]> : !mosh.ape>
// CHECK-SAME: (!mogg.dim<batch>) -> !mogg.shape
// CHECK: %[[PRO:.*]] = mogg.tensor.create.prologue %{{.*}} capturing(
// CHECK-SAME: %[[C0:.*]], %[[C1:.*]] : !mogg.tensor, !mogg.tensor)
// CHECK-SAME: !mogg.shape -> !mogg.tensor
// CHECK-SAME: device = "cpu", elementType = f32,
// CHECK-SAME: io = Input, mut = false {
// CHECK: ^bb0(%[[I:.*]]: !map.index<2>, %[[A0:.*]]: !mogg.tensor,
// CHECK-SAME: %[[A1:.*]]: !mogg.tensor):
// CHECK: %[[L0:.*]] = mogg.tensor.load %[[A0]][%[[I]]]
// CHECK: %[[L1:.*]] = mogg.tensor.load %[[A1]][%[[I]]]
// CHECK: %[[ADD:.*]] = mogg.call.elementwise @{{.*}}Add(%[[L0]], %[[L1]])
// CHECK: mogg.yield %[[ADD]] : f32
// CHECK: mogg.call.opaque @{{.*}}MapLowerPartialFusedInputs
// CHECK-SAME: inputs(%[[PRO]], %{{.*}} : !mogg.tensor, !mogg.tensor)
// CHECK-NOT: map.iter.prologue
mo.graph @dyn_fused_producer(
  %a: !mo.tensor<[batch, 4], f32>,
  %b: !mo.tensor<[batch, 4], f32>,
  %c: !mo.tensor<[batch, 4], f32>
) -> !mo.tensor<[batch, 4], f32> {
  %p = mo.add(%a, %b) : !mo.tensor<[batch, 4], f32>
  %out = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_partial_fused_inputs"}(%p, %c) : (!mo.tensor<[batch, 4], f32>, !mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32>
  mo.output %out : !mo.tensor<[batch, 4], f32>
}
