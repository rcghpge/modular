// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels --map-fuse-mutable-loads --map-outline-structured-kernel-bodies --map-to-mogg | FileCheck %s

// `map-fuse-mutable-loads` folds the load's mutable buffer into the loop, so
// the loaded and stored captures reach `mogg.call.foreach` as `!mogg.buffer_`
// rather than `!mogg.tensor_`.

// CHECK-LABEL: mo.graph @in_place_add
// CHECK: [[IN:%[^ ]+]] = mogg.tensor.create({{.*}}) {{{.*}}mut = true{{.*}}} : (!mogg.pointer_, !mogg.shape) -> !mogg.buffer_
// CHECK: [[RHS:%[^ ]+]] = mogg.tensor.create({{.*}}) {{{.*}}mut = false{{.*}}} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK: [[OUT:%[^ ]+]] = mogg.tensor.create({{.*}}) {{{.*}}mut = true{{.*}}} : (!mogg.pointer_, !mogg.shape) -> !mogg.buffer_
// CHECK: mogg.call.foreach {{.*}} capturing([[IN]], [[RHS]], [[OUT]] : !mogg.buffer_, !mogg.tensor_, !mogg.buffer_)
// CHECK-NEXT: ^bb0([[BIN:%[^ :,]+]]: !mogg.buffer_, [[BRHS:%[^ :,]+]]: !mogg.tensor_, [[BOUT:%[^ :,]+]]: !mogg.buffer_, [[IDX:%[^ :,]+]]: !map.index<2>):
// CHECK-NEXT:   mogg.tensor.load [[BIN]][[[IDX]]] : !mogg.buffer_, !map.index<2> -> f32
// CHECK: mogg.tensor.store [[BOUT]][[[IDX]]] = {{%[^ ]+}} : !mogg.buffer_, !map.index<2>, f32
mo.graph @in_place_add(%buf: !mo.buffer<[2, 2], f32>, %rhs: !mo.tensor<[2, 2], f32>) -> (!mo.buffer<[2, 2], f32>, !mo.chain) {
  %ch0 = mo.chain.create(%buf) : !mo.buffer<[2, 2], f32>
  %vbuf, %ch1 = mo.mutable.load(%buf, %ch0) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %res = mo.add(%vbuf, %rhs) : !mo.tensor<[2, 2], f32>
  %ch2 = mo.mutable.store(%buf, %res, %ch1) : (!mo.buffer<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.chain) -> (!mo.chain)
  mo.output %buf, %ch2 : !mo.buffer<[2, 2], f32>, !mo.chain
}
