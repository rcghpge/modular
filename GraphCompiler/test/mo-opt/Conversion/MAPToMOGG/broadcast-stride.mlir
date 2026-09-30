// RUN: mo-opt %s --pass-pipeline='builtin.module(mogg-import-packages, mo-to-map, map-outline-structured-kernel-bodies, mo.graph(map-livm, map.iter.opaque(canonicalize)), map-to-mogg)' | FileCheck %s

// Full pipeline from a real mo.static.broadcast_to op through MOToMAP,
// outlining, view motion, and MAPToMOGG: map.index.broadcast.stride lowers to
// mogg.tensor.create.broadcast, which only carries the view's own output
// shape -- the input's shape/rank is read off the input tensor operand at
// Mojo-emission time, not duplicated onto this op. The `canonicalize` right
// after LoopInvariantViewMotion drops the closure's now-dead original input
// capture (unused once the hoisted broadcast view replaces the load) via
// `DropDeadCapturingClosureCaptures`.

// Rank-preserving: in-shape dim 0 (size 1) broadcasts to 8.

// CHECK-LABEL: mo.graph @broadcastSameRank
// CHECK:       [[KERNEL:%[^ ]+]] = mogg.kernel ({{.*}} : !mo.tensor<[1, 64], f32>) -> !mo.tensor<[8, 64], f32>
// CHECK-NEXT:  inputs ([[IP:%[^ :,]+]]: !mogg.pointer_)
// CHECK-NEXT:  outputs ([[OP:%[^ :,]+]]: !mogg.pointer_)
// CHECK-NEXT:  context ([[CTX:%[^ :,]+]]: !mogg.context) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK-NEXT:    [[ISH:%[^ ]+]] = mogg.shape.create() {shape = #mosh<ape[1, 64]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:    [[IT:%[^ ]+]] = mogg.tensor.create([[IP]], [[ISH]]) {{.*}}io = 1{{.*}} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK-NEXT:    [[OSH:%[^ ]+]] = mogg.shape.create() {shape = #mosh<ape[8, 64]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:    [[OT:%[^ ]+]] = mogg.tensor.create([[OP]], [[OSH]]) {{.*}}io = 0{{.*}} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK-NEXT:    [[BCAST:%[^ ]+]] = mogg.tensor.create.broadcast([[IT]]) {{.*}}outShape = #mosh<ape[8, 64]>{{.*}} : !mogg.tensor_ -> !mogg.tensor_
// CHECK-NEXT:    [[LSH:%[^ ]+]] = mogg.shape.create() {shape = #mosh<ape[8, 64]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:    mogg.call.foreach @builtin_primitives.primitives._foreach shape([[LSH]]) context([[CTX]]) capturing([[OT]], [[BCAST]] : !mogg.tensor_, !mogg.tensor_) attributes {device = "cpu"} {
// CHECK-NEXT:    ^bb0([[OUT:%[^ :,]+]]: !mogg.tensor_, [[BCASTARG:%[^ :,]+]]: !mogg.tensor_, [[IDX:%[^ :,]+]]: !map.index<2>):
// CHECK-NEXT:      [[VAL:%[^ ]+]] = mogg.tensor.load [[BCASTARG]]{{\[}}[[IDX]]] : !mogg.tensor_, !map.index<2> -> f32
// CHECK-NEXT:      mogg.tensor.store [[OUT]]{{\[}}[[IDX]]] = [[VAL]] : !mogg.tensor_, !map.index<2>, f32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
mo.graph @broadcastSameRank(%input: !mo.tensor<[1, 64], f32>)
    -> !mo.tensor<[8, 64], f32> {
  %res = mo.static.broadcast_to(%input) : !mo.tensor<[1, 64], f32> -> !mo.tensor<[8, 64], f32>
  mo.output %res : !mo.tensor<[8, 64], f32>
}

// Rank-expanding: in has one fewer dim than out (numpy semantics). The
// generated mogg.tensor.create.broadcast is identical in shape to the
// rank-preserving case above -- $outShape is the only shape this op carries;
// the input's own (smaller) rank is inferred later, purely from its Mojo
// argument type at emission time.

// CHECK-LABEL: mo.graph @broadcastRankExpanding
// CHECK: mogg.tensor.create.broadcast(%{{.*}}) {{.*}}outShape = #mosh<ape[8, 64]>{{.*}} : !mogg.tensor_ -> !mogg.tensor_
mo.graph @broadcastRankExpanding(%input: !mo.tensor<[64], f32>)
    -> !mo.tensor<[8, 64], f32> {
  %res = mo.static.broadcast_to(%input) : !mo.tensor<[64], f32> -> !mo.tensor<[8, 64], f32>
  mo.output %res : !mo.tensor<[8, 64], f32>
}
