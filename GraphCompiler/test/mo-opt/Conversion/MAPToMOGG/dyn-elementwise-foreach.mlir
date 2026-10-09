// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg | FileCheck %s

// CHECK-LABEL: mo.graph @add
// CHECK: mogg.kernel (%{{.*}} : !mo.tensor<[M, N], f32>, !mo.tensor<[M, N], f32>) -> !mo.tensor<[M, N], f32>
// CHECK-NEXT: inputs ([[IP0:%[^ :,]+]]: !mogg.pointer, [[IP1:%[^ :,]+]]: !mogg.pointer)
// CHECK-NEXT: outputs ([[OP:%[^ :,]+]]: !mogg.pointer)
// CHECK-NEXT: context ([[CTX:%[^ :,]+]]: !mogg.context)
// CHECK-NEXT: dims ([[DM:%[^ :,]+]]: !mogg.dim<M>, [[DN:%[^ :,]+]]: !mogg.dim<N>) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK-NEXT:   [[ISH0:%[^ ]+]] = mogg.shape.create([[DM]], [[DN]]) <shape = #mosh<ape[dyn, dyn]> : !mosh.ape> : (!mogg.dim<M>, !mogg.dim<N>) -> !mogg.shape
// CHECK-NEXT:   [[IT0:%[^ ]+]] = mogg.tensor.create([[IP0]], [[ISH0]]) <device = "cpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[M, N]> : !mosh.ape> : (!mogg.pointer, !mogg.shape) -> !mogg.tensor
// CHECK-NEXT:   [[ISH1:%[^ ]+]] = mogg.shape.create([[DM]], [[DN]]) <shape = #mosh<ape[dyn, dyn]> : !mosh.ape> : (!mogg.dim<M>, !mogg.dim<N>) -> !mogg.shape
// CHECK-NEXT:   [[IT1:%[^ ]+]] = mogg.tensor.create([[IP1]], [[ISH1]]) <device = "cpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[M, N]> : !mosh.ape> : (!mogg.pointer, !mogg.shape) -> !mogg.tensor
// CHECK-NEXT:   [[OSH:%[^ ]+]] = mogg.shape.create([[DM]], [[DN]]) <shape = #mosh<ape[dyn, dyn]> : !mosh.ape> : (!mogg.dim<M>, !mogg.dim<N>) -> !mogg.shape
// CHECK-NEXT:   [[OT:%[^ ]+]] = mogg.tensor.create([[OP]], [[OSH]]) <device = "cpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[M, N]> : !mosh.ape> : (!mogg.pointer, !mogg.shape) -> !mogg.tensor
// CHECK-NEXT:   [[LSH:%[^ ]+]] = mogg.shape.create([[DM]], [[DN]]) <shape = #mosh<ape[dyn, dyn]> : !mosh.ape> : (!mogg.dim<M>, !mogg.dim<N>) -> !mogg.shape
// CHECK-NEXT:   mogg.call.foreach @builtin_primitives.primitives._foreach shape([[LSH]]) context([[CTX]]) capturing([[IT0]], [[IT1]], [[OT]] : !mogg.tensor, !mogg.tensor, !mogg.tensor) device = "cpu" {
// CHECK-NEXT:   ^bb0([[IN0:%[^ :,]+]]: !mogg.tensor, [[IN1:%[^ :,]+]]: !mogg.tensor, [[OUT:%[^ :,]+]]: !mogg.tensor, [[IDX:%[^ :,]+]]: !map.index<2>):
// CHECK-NEXT:     [[V0:%[^ ]+]] = mogg.tensor.load [[IN0]][[[IDX]]] : !mogg.tensor, !map.index<2> -> f32
// CHECK-NEXT:     [[V1:%[^ ]+]] = mogg.tensor.load [[IN1]][[[IDX]]] : !mogg.tensor, !map.index<2> -> f32
// CHECK-NEXT:     [[SUM:%[^ ]+]] = mogg.call.elementwise @testing_kernels.testing_kernels.AddElementwise([[V0]], [[V1]]) moOpName = "imposter_add_elementwise" : f32, f32 -> f32
// CHECK-NEXT:     mogg.tensor.store [[OUT]][[[IDX]]] = [[SUM]] : !mogg.tensor, !map.index<2>, f32
// CHECK-NEXT:   }
// CHECK-NEXT: }
mo.graph @add<M, N>(%a: !mo.tensor<[M, N], f32>, %b: !mo.tensor<[M, N], f32>) -> !mo.tensor<[M, N], f32> {
  %c = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>](%a, %b) : (!mo.tensor<[M, N], f32>, !mo.tensor<[M, N], f32>) -> !mo.tensor<[M, N], f32>
  mo.output %c : !mo.tensor<[M, N], f32>
}
