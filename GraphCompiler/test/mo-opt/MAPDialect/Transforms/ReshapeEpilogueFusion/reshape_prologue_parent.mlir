// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels | FileCheck %s

// A reshape on a kernel's INPUT side fuses into its prologue, so
// `map.index.reshape` has to be legal there -- every sibling index op already
// is. Nothing else was needed: the emitter and the index transform both
// handle a prologue parent, only the op's parent list did not admit one.

// CHECK-LABEL: mo.graph @reshape_into_reduce_prologue
// CHECK: map.iter.prologue
// CHECK: map.index.reshape
mo.graph @reshape_into_reduce_prologue(%a: !mo.tensor<[1, 4], f32>) -> (!mo.tensor<[1], f32>) {
  %r = mo.static.reshape(%a) : !mo.tensor<[1, 4], f32> -> !mo.tensor<[4], f32>
  %m = mo.reduce.mean(%r) [axis = 0] : (!mo.tensor<[4], f32>) -> !mo.tensor<[1], f32>
  mo.output %m : !mo.tensor<[1], f32>
}
