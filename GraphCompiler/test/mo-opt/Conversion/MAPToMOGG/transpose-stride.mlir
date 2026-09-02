// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --pass-pipeline='builtin.module(mogg-import-packages, mo-to-map, map-outline-structured-kernel-bodies, mo.graph(map.iter.opaque(map-loop-invariant-view-motion, canonicalize)), map-to-mogg)' | FileCheck %s

// Full pipeline from a real mo.transpose op through MOToMAP, outlining, view
// motion, and MAPToMOGG: map.index.transpose.stride lowers to
// mogg._tensor.create.transpose, which carries the view's own output shape
// and perm -- the input's shape/rank is read off the input tensor operand at
// Mojo-emission time, not duplicated onto this op. perm's own tensor operand
// (reconstructed as mogg._tensor.create like any other kernel input) goes
// unused once its value has been baked into the op as a compile-time
// constant attribute. The `canonicalize` right after LoopInvariantViewMotion
// drops the closure's now-dead original input capture (unused once the
// hoisted transpose view replaces the load) via
// `DropDeadCapturingClosureCaptures`.

// 2D swap: out.dim[0] = in.dim[1], out.dim[1] = in.dim[0].

// CHECK-LABEL: mo.graph @transposeSwap
// CHECK:       [[KERNEL:%[^ ]+]] = mogg._kernel ({{.*}} : !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>
// CHECK-NEXT:  inputs ([[IP:%[^ :,]+]]: !mogg.pointer_, [[PP:%[^ :,]+]]: !mogg.pointer_)
// CHECK-NEXT:  outputs ([[OP:%[^ :,]+]]: !mogg.pointer_)
// CHECK-NEXT:  context ([[CTX:%[^ :,]+]]: !mogg.context) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK-NEXT:    [[ISH:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[2, 3]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:    [[IT:%[^ ]+]] = mogg._tensor.create([[IP]], [[ISH]]) {{.*}}io = 1{{.*}} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK-NEXT:    [[PSH:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[2]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:    [[PT:%[^ ]+]] = mogg._tensor.create([[PP]], [[PSH]]) {{.*}}io = 1{{.*}} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK-NEXT:    [[OSH:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[3, 2]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:    [[OT:%[^ ]+]] = mogg._tensor.create([[OP]], [[OSH]]) {{.*}}io = 0{{.*}} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK-NEXT:    [[TRANS:%[^ ]+]] = mogg._tensor.create.transpose([[IT]]) {{.*}}outShape = #mosh<ape[3, 2]>{{.*}}perm = array<i64: 1, 0>{{.*}} : !mogg.tensor_ -> !mogg.tensor_
// CHECK-NEXT:    [[LSH:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[3, 2]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:    mogg._call.foreach @builtin_primitives.primitives._foreach shape([[LSH]]) context([[CTX]]) capturing([[OT]], [[TRANS]] : !mogg.tensor_, !mogg.tensor_) attributes {device = #M.device_ref<"cpu", 0>} {
// CHECK-NEXT:    ^bb0([[OUT:%[^ :,]+]]: !mogg.tensor_, [[TRANSARG:%[^ :,]+]]: !mogg.tensor_, [[IDX:%[^ :,]+]]: !map.index<2>):
// CHECK-NEXT:      [[VAL:%[^ ]+]] = mogg._tensor.load [[TRANSARG]]{{\[}}[[IDX]]] : !mogg.tensor_, !map.index<2> -> f32
// CHECK-NEXT:      mogg._tensor.store [[OUT]]{{\[}}[[IDX]]] = [[VAL]] : !mogg.tensor_, !map.index<2>, f32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
mo.graph @transposeSwap(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 2], f32> {
  %perm = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.transpose(%input, %perm) : (
    !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}

// 3D permutation: the generated mogg._tensor.create.transpose carries the
// full non-trivial perm; the input's own (differently-ordered) shape is
// inferred later, purely from its Mojo argument type at emission time.

// CHECK-LABEL: mo.graph @transpose3D
// CHECK: mogg._tensor.create.transpose(%{{.*}}) {{.*}}outShape = #mosh<ape[4, 2, 3]>{{.*}}perm = array<i64: 2, 0, 1>{{.*}} : !mogg.tensor_ -> !mogg.tensor_
mo.graph @transpose3D(%input: !mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[4, 2, 3], f32> {
  %perm = mo.constant {value = #M.dense_array<2, 0, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %res = mo.transpose(%input, %perm) : (
    !mo.tensor<[2, 3, 4], f32>, !mo.tensor<[3], si64>
  ) -> !mo.tensor<[4, 2, 3], f32>
  mo.output %res : !mo.tensor<[4, 2, 3], f32>
}
