// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --pass-pipeline='builtin.module(mogg-import-packages, mo-to-map, mo.graph(map-fuse-kernels), map-outline-structured-kernel-bodies, mo.graph(canonicalize), map-to-mogg)' | FileCheck %s

// Full pipeline from a real `mo.constant` fed to a broadcast: `LoadOp`'s
// canonicalizer turns the per-element read of the one-element constant into a
// `map.constant.scalar`, which lowers to `mogg._constant.scalar` inside the
// `mogg._call.foreach` region. The constant buffer is not a kernel operand at
// all -- no `!mogg.pointer_` input, no `mogg._tensor.create` for it.

// The elementwise-operand shape: the literal takes the place of one of the
// add's two loads, so the call comes out `f32, f32 -> f32` exactly as it
// would with two real loads.

// CHECK-LABEL: mo.graph @addConstant
// CHECK:       mogg._kernel ({{.*}} : !mo.tensor<[8, 64], f32>) -> !mo.tensor<[8, 64], f32>
// CHECK-NEXT:  inputs ([[IP:%[^ :,]+]]: !mogg.pointer_)
// CHECK-NEXT:  outputs ([[OP:%[^ :,]+]]: !mogg.pointer_)
// CHECK-NEXT:  context ([[CTX:%[^ :,]+]]: !mogg.context) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK-NEXT:    [[ISH:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[8, 64]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:    [[IT:%[^ ]+]] = mogg._tensor.create([[IP]], [[ISH]]) {{.*}}io = 1{{.*}} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK-NEXT:    [[OSH:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[8, 64]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:    [[OT:%[^ ]+]] = mogg._tensor.create([[OP]], [[OSH]]) {{.*}}io = 0{{.*}} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK-NEXT:    [[LSH:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[8, 64]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:    mogg._call.foreach @builtin_primitives.primitives._foreach shape([[LSH]]) context([[CTX]]) capturing([[IT]], [[OT]] : !mogg.tensor_, !mogg.tensor_) attributes {device = #M.device_ref<"cpu", 0>} {
// CHECK-NEXT:    ^bb0([[IN:%[^ :,]+]]: !mogg.tensor_, [[OUT:%[^ :,]+]]: !mogg.tensor_, [[IDX:%[^ :,]+]]: !map.index<2>):
// CHECK-NEXT:      [[CST:%[^ ]+]] = mogg._constant.scalar {value = 2.000000e+00 : f32} : f32
// CHECK-NEXT:      [[VAL:%[^ ]+]] = mogg._tensor.load [[IN]]{{\[}}[[IDX]]] : !mogg.tensor_, !map.index<2> -> f32
// CHECK-NEXT:      [[SUM:%[^ ]+]] = mogg._call.elementwise["builtin_kernels.elementwise.Add"] ([[VAL]], [[CST]]) {moOpName = "mo.add"} : f32, f32 -> f32
// CHECK-NEXT:      mogg._tensor.store [[OUT]]{{\[}}[[IDX]]] = [[SUM]] : !mogg.tensor_, !map.index<2>, f32
mo.graph @addConstant(%x: !mo.tensor<[8, 64], f32>) -> !mo.tensor<[8, 64], f32> {
  %cst = mo.constant {value = #M.dense_array<2.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%cst) : !mo.tensor<[], f32> -> !mo.tensor<[8, 64], f32>
  %r = mo.add(%x, %b) : !mo.tensor<[8, 64], f32>
  mo.output %r : !mo.tensor<[8, 64], f32>
}

// An unfused broadcast of a constant is a whole kernel whose only job was the
// read: it lowers to a kernel with no input pointer at all, storing the
// literal.

// CHECK-LABEL: mo.graph @broadcastConstant
// CHECK:       mogg._kernel () -> !mo.tensor<[8, 64], f32>
// CHECK-NEXT:  outputs ([[OP2:%[^ :,]+]]: !mogg.pointer_)
// CHECK:         mogg._call.foreach
// CHECK-NEXT:    ^bb0([[OUT2:%[^ :,]+]]: !mogg.tensor_, [[IDX2:%[^ :,]+]]: !map.index<2>):
// CHECK-NEXT:      [[CST2:%[^ ]+]] = mogg._constant.scalar {value = 5.000000e-01 : f32} : f32
// CHECK-NEXT:      mogg._tensor.store [[OUT2]]{{\[}}[[IDX2]]] = [[CST2]] : !mogg.tensor_, !map.index<2>, f32
mo.graph @broadcastConstant() -> !mo.tensor<[8, 64], f32> {
  %cst = mo.constant {value = #M.dense_array<5.000000e-01> : tensor<f32>} : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%cst) : !mo.tensor<[], f32> -> !mo.tensor<[8, 64], f32>
  mo.output %b : !mo.tensor<[8, 64], f32>
}
