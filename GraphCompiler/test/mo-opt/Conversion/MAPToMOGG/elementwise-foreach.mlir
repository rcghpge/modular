// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg | FileCheck %s

// CHECK-LABEL: mo.graph @add
// CHECK: mogg._kernel (%{{.*}} : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
// CHECK-NEXT: inputs ([[IP0:%[^ :,]+]]: !mogg.pointer_, [[IP1:%[^ :,]+]]: !mogg.pointer_)
// CHECK-NEXT: outputs ([[OP:%[^ :,]+]]: !mogg.pointer_)
// CHECK-NEXT: context ([[CTX:%[^ :,]+]]: !mogg.context) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK-NEXT:   [[ISH0:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[2, 2]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:   [[IT0:%[^ ]+]] = mogg._tensor.create([[IP0]], [[ISH0]]) {device = #M.device_ref<"cpu", 0>, elementType = f32, io = 1 : i32, mut = false} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK-NEXT:   [[ISH1:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[2, 2]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:   [[IT1:%[^ ]+]] = mogg._tensor.create([[IP1]], [[ISH1]]) {device = #M.device_ref<"cpu", 0>, elementType = f32, io = 1 : i32, mut = false} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK-NEXT:   [[OSH:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[2, 2]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:   [[OT:%[^ ]+]] = mogg._tensor.create([[OP]], [[OSH]]) {device = #M.device_ref<"cpu", 0>, elementType = f32, io = 0 : i32, mut = true} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK-NEXT:   [[LSH:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[2, 2]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:   mogg._call.foreach @builtin_primitives.primitives._foreach shape([[LSH]]) context([[CTX]]) capturing([[IT0]], [[IT1]], [[OT]] : !mogg.tensor_, !mogg.tensor_, !mogg.tensor_) {
// CHECK-NEXT:   ^bb0([[IN0:%[^ :,]+]]: !mogg.tensor_, [[IN1:%[^ :,]+]]: !mogg.tensor_, [[OUT:%[^ :,]+]]: !mogg.tensor_, [[IDX:%[^ :,]+]]: !map.index<2>):
// CHECK-NEXT:     [[V0:%[^ ]+]] = mogg._tensor.load [[IN0]][[[IDX]]] : !mogg.tensor_, !map.index<2> -> f32
// CHECK-NEXT:     [[V1:%[^ ]+]] = mogg._tensor.load [[IN1]][[[IDX]]] : !mogg.tensor_, !map.index<2> -> f32
// CHECK-NEXT:     [[SUM:%[^ ]+]] = mogg._call.elementwise["testing_kernels.testing_kernels.AddElementwise"] ([[V0]], [[V1]]) {moOpName = "imposter_add_elementwise"} : f32, f32 -> f32
// CHECK-NEXT:     mogg._tensor.store [[OUT]][[[IDX]]] = [[SUM]] : !mogg.tensor_, !map.index<2>, f32
// CHECK-NEXT:   }
// CHECK-NEXT: }
mo.graph @add(%a: !mo.tensor<[2, 2], f32>, %b: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_elementwise"}(%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}
