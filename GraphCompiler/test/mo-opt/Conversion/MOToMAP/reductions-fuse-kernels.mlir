// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels | FileCheck %s

// Checks that kernel fusion fuses every reduction pattern we care about. It
// grows as fusion gains features: what does not fuse yet is checked as
// unfused, and moves over once it does.
//
// Today the `mul` fuses into the sum's fold. The rest stays outside: sibling
// reductions, a reduction of a reduction, a full-row consumer and a view of a
// result.

// CHECK-LABEL: mo.graph @aroundReductions
// CHECK-SAME:  (%[[X:.*]]: !mo.tensor<[8, 128], f32>)
// CHECK:       %[[S:.*]] = map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:  inputs  (%[[X]]: !mo.tensor<[8, 128], f32>)
// CHECK:       map.iter.reduce @builtin_kernels.monoids.ReduceSum
// CHECK:       map.call.elementwise @builtin_kernels.elementwise.Mul
// CHECK:       %[[M:.*]] = map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:  inputs  (%[[X]]: !mo.tensor<[8, 128], f32>)
// CHECK:       map.iter.reduce @builtin_kernels.monoids.ReduceMax
// CHECK:       map.iter.foreach<[8, 128]>
// CHECK-NEXT:  inputs  (%[[X]]: !mo.tensor<[8, 128], f32>, %[[S]]: !mo.tensor<[8, 1], f32>)
// CHECK:       map.call.elementwise @builtin_kernels.elementwise.Div
// CHECK:       map.iter.foreach<[8]>
// CHECK-NEXT:  inputs  (%[[M]]: !mo.tensor<[8, 1], f32>)
// CHECK:       map.iter.rowwise<[8, 1], axis = 0>
// CHECK-NEXT:  inputs  (%[[M]]: !mo.tensor<[8, 1], f32>)
// CHECK:       map.iter.reduce @builtin_kernels.monoids.ReduceMin
mo.graph @aroundReductions(%x: !mo.tensor<[8, 128], f32>) -> (!mo.tensor<[8, 128], f32>, !mo.tensor<[8], f32>, !mo.tensor<[1, 1], f32>, !mo.tensor<[8, 1], f32>) {
  %sq = mo.mul(%x, %x) : !mo.tensor<[8, 128], f32>
  %s = mo.custom [symbol = "mo.reduce.add.tmp", device = #M.device_ref<"cpu", 0>](%sq) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %m = mo.custom [symbol = "mo.reduce.max.tmp", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %d = mo.div(%x, %sb) : !mo.tensor<[8, 128], f32>
  %mr = mo.static.reshape(%m) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8], f32>
  %mm = mo.custom [symbol = "mo.reduce.min.tmp", device = #M.device_ref<"cpu", 0>](%m) : (!mo.tensor<[8, 1], f32>) -> !mo.tensor<[1, 1], f32>
  mo.output %d, %mr, %mm, %s : !mo.tensor<[8, 128], f32>, !mo.tensor<[8], f32>, !mo.tensor<[1, 1], f32>, !mo.tensor<[8, 1], f32>
}
