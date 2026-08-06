// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg | FileCheck %s

// CHECK-LABEL: mo.graph @add_with_index
// CHECK: mogg._kernel (%{{.*}} : !mo.tensor<[8], si32>) -> !mo.tensor<[8], si32>
// CHECK-NEXT: inputs ([[IP:%[^ :,]+]]: !mogg.pointer_)
// CHECK-NEXT: outputs ([[OP:%[^ :,]+]]: !mogg.pointer_)
// CHECK-NEXT: context ([[CTX:%[^ :,]+]]: !mogg.context) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK-NEXT:   [[ISH:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[8]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:   [[IT:%[^ ]+]] = mogg._tensor.create([[IP]], [[ISH]]) {device = #M.device_ref<"cpu", 0>, elementType = si32, io = 1 : i32, mut = false} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK-NEXT:   [[OSH:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[8]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:   [[OT:%[^ ]+]] = mogg._tensor.create([[OP]], [[OSH]]) {device = #M.device_ref<"cpu", 0>, elementType = si32, io = 0 : i32, mut = true} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK-NEXT:   [[LSH:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[8]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:   mogg._call.foreach @builtin_primitives.primitives._foreach shape([[LSH]]) context([[CTX]]) capturing([[IT]], [[OT]] : !mogg.tensor_, !mogg.tensor_) {
// CHECK-NEXT:   ^bb0([[IN:%[^ :,]+]]: !mogg.tensor_, [[OUT:%[^ :,]+]]: !mogg.tensor_, [[IDX:%[^ :,]+]]: !map.index<1>):
// CHECK-NEXT:     [[V:%[^ ]+]] = mogg._tensor.load [[IN]][[[IDX]]] : !mogg.tensor_, !map.index<1> -> si32
// CHECK-NEXT:     [[SUM:%[^ ]+]] = mogg._call.elementwise["testing_kernels.testing_kernels.ElemwiseAddWithIndex"] ([[V]], [[IDX]]) {moOpName = "elemwise_add_with_index"} : si32, !map.index<1> -> si32
// CHECK-NEXT:     mogg._tensor.store [[OUT]][[[IDX]]] = [[SUM]] : !mogg.tensor_, !map.index<1>, si32
// CHECK-NEXT:   }
// CHECK-NEXT: }
mo.graph @add_with_index(%0: !mo.tensor<[8], si32>) -> !mo.tensor<[8], si32> {
  %1 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "elemwise_add_with_index"}(%0) : (!mo.tensor<[8], si32>) -> !mo.tensor<[8], si32>
  mo.output %1 : !mo.tensor<[8], si32>
}
