// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --pass-pipeline='builtin.module(mogg-import-packages, mo-to-map, map-outline-structured-kernel-bodies, mo.graph(map.iter.opaque(map-loop-invariant-view-motion, canonicalize)), map-to-mogg)' | FileCheck %s

// Full pipeline from a real mo.slice op through MOToMAP, outlining, view
// motion, and MAPToMOGG: map.index.slice.stride lowers to
// mogg._tensor.create.slice, and only the slice result and the output tensor
// are captured by mogg._call.foreach, since the loop body only references
// those two. The `canonicalize` right after LoopInvariantViewMotion drops
// the closure's now-dead original captures (the raw input tensor and the
// starts/stops/steps bounds, all unused once the hoisted view replaces the
// load) via `DropDeadCapturingClosureCaptures` -- without it the lowered
// kernel's parameter list would carry those unused pointers too.

// Static bounds (mo.constant starts/stops/steps): every bound is a literal.

// CHECK-LABEL: mo.graph @sliceStaticBounds
// CHECK:       [[KERNEL:%[^ ]+]] = mogg._kernel ({{.*}} : !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 32], f32>
// CHECK-NEXT:  inputs ([[IP0:%[^ :,]+]]: !mogg.pointer_, [[IP1:%[^ :,]+]]: !mogg.pointer_, [[IP2:%[^ :,]+]]: !mogg.pointer_, [[IP3:%[^ :,]+]]: !mogg.pointer_)
// CHECK-NEXT:  outputs ([[OP:%[^ :,]+]]: !mogg.pointer_)
// CHECK-NEXT:  context ([[CTX:%[^ :,]+]]: !mogg.context) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK-NEXT:    [[ISH0:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[10, 64]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:    [[IT0:%[^ ]+]] = mogg._tensor.create([[IP0]], [[ISH0]]) {{.*}}io = 1{{.*}} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK-NEXT:    [[ISH1:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[2]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:    [[IT1:%[^ ]+]] = mogg._tensor.create([[IP1]], [[ISH1]]) {{.*}}io = 1{{.*}} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK-NEXT:    [[ISH2:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[2]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:    [[IT2:%[^ ]+]] = mogg._tensor.create([[IP2]], [[ISH2]]) {{.*}}io = 1{{.*}} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK-NEXT:    [[ISH3:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[2]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:    [[IT3:%[^ ]+]] = mogg._tensor.create([[IP3]], [[ISH3]]) {{.*}}io = 1{{.*}} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK-NEXT:    [[OSH:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[3, 32]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:    [[OT:%[^ ]+]] = mogg._tensor.create([[OP]], [[OSH]]) {{.*}}io = 0{{.*}} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK-NEXT:    [[STRIDED:%[^ ]+]] = mogg._tensor.create.slice([[IT0]], [[IT1]], [[IT2]], [[IT3]]) {{.*}}outShape = #mosh<ape[3, 32]>{{.*}}staticStart = array<i64: 1, 32>{{.*}}staticStep = array<i64: 1, 1>{{.*}} : (!mogg.tensor_, !mogg.tensor_, !mogg.tensor_, !mogg.tensor_) -> !mogg.tensor_
// CHECK-NEXT:    [[LSH:%[^ ]+]] = mogg._shape.create() {shape = #mosh<ape[3, 32]> : !mosh.ape} : () -> !mogg.shape
// CHECK-NEXT:    mogg._call.foreach @builtin_primitives.primitives._foreach shape([[LSH]]) context([[CTX]]) capturing([[OT]], [[STRIDED]] : !mogg.tensor_, !mogg.tensor_) attributes {device = #M.device_ref<"cpu", 0>} {
// CHECK-NEXT:    ^bb0([[OUT:%[^ :,]+]]: !mogg.tensor_, [[STRIDEDARG:%[^ :,]+]]: !mogg.tensor_, [[IDX:%[^ :,]+]]: !map.index<2>):
// CHECK-NEXT:      [[VAL:%[^ ]+]] = mogg._tensor.load [[STRIDEDARG]]{{\[}}[[IDX]]] : !mogg.tensor_, !map.index<2> -> f32
// CHECK-NEXT:      mogg._tensor.store [[OUT]]{{\[}}[[IDX]]] = [[VAL]] : !mogg.tensor_, !map.index<2>, f32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
mo.graph @sliceStaticBounds(%input: !mo.tensor<[10, 64], f32>) -> !mo.tensor<[3, 32], f32> {
  %start = mo.constant {value = #M.dense_array<1, 32> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop = mo.constant {value = #M.dense_array<4, 64> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32], f32>
  mo.output %res : !mo.tensor<[3, 32], f32>
}

// Dynamic bounds (graph arguments, not mo.constant): every bound is
// unresolved at compile time.

// CHECK-LABEL: mo.graph @sliceDynamicBounds
// CHECK: mogg._tensor.create.slice(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}outShape = #mosh<ape[3, 32]>{{.*}}
mo.graph @sliceDynamicBounds(
    %input: !mo.tensor<[10, 64], f32>,
    %start: !mo.tensor<[2], si64>,
    %stop: !mo.tensor<[2], si64>,
    %step: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 32], f32> {
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32], f32>
  mo.output %res : !mo.tensor<[3, 32], f32>
}

// A constant start indexing a DYNAMIC input dim (mirrors `x[-8:]` off a
// dynamic `seq_len`): MOToMAP normalizes a start against its dim's extent only
// for static-input dims, so this start stays the raw literal `-8`. Its real
// offset `dim - 8` is not known until runtime, so its `staticStart` entry must
// lower to `?` (`kDynamic`, printed as the i64 min) -- not the raw `-8`, which
// the slice primitive cannot infer as a comptime offset (GEX-3964 / the
// mixed-shape LLM snifftest gap). The static dim1 start (0) stays static.

// CHECK-LABEL: mo.graph @sliceNegativeStartDynamicInDim
// CHECK: mogg._tensor.create.slice(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}staticStart = array<i64: -9223372036854775808, 0>{{.*}}staticStep = array<i64: 1, 1>{{.*}}
mo.graph @sliceNegativeStartDynamicInDim(
    %input: !mo.tensor<[?, 64], f32>,
    %stop: !mo.tensor<[2], si64>)
    -> !mo.tensor<[?, 64], f32> {
  %start = mo.constant {value = #M.dense_array<-8, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[?, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[?, 64], f32>
  mo.output %res : !mo.tensor<[?, 64], f32>
}

// A NON-negative constant start indexing a DYNAMIC input dim: `2` is already an
// absolute offset (no extent needed to resolve it), so it stays a comptime
// `staticStart` hint even though dim0 is dynamic -- only negatives, which need
// the unknown extent to normalize, must lower to `?`.

// CHECK-LABEL: mo.graph @slicePositiveStartDynamicInDim
// CHECK: mogg._tensor.create.slice(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}staticStart = array<i64: 2, 0>{{.*}}staticStep = array<i64: 1, 1>{{.*}}
mo.graph @slicePositiveStartDynamicInDim(
    %input: !mo.tensor<[?, 64], f32>,
    %stop: !mo.tensor<[2], si64>)
    -> !mo.tensor<[?, 64], f32> {
  %start = mo.constant {value = #M.dense_array<2, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[?, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[?, 64], f32>
  mo.output %res : !mo.tensor<[?, 64], f32>
}
