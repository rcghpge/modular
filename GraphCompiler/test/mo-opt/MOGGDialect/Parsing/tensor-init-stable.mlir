// RUN: mo-opt --allow-unregistered-dialect %s | FileCheck %s

// A stable kernel-facing tensor is an ordinary `mogg.tensor.__init__` whose
// result `!mo.tensor` carries the `stable` metadata entry. The op is the same;
// the result type decides which constructor it lowers to.

// CHECK-LABEL: func.func @tensor_init_stable
func.func @tensor_init_stable(%buf: !mgp.buffer<<"gpu", 0>>) {
  // CHECK: mogg.tensor.__init__(%{{.*}}) {io = 1 : i32, mut = false} : (!mgp.buffer<<"gpu", 0>>) -> !mo.tensor<[4, 8], f32, {io = 1 : i32, mut = false, stable}>
  %t = mogg.tensor.__init__(%buf) {io = 1 : i32, mut = false}
      : (!mgp.buffer<<"gpu", 0>>) -> !mo.tensor<[4, 8], f32, {io = 1 : i32, mut = false, stable}>
  // CHECK: mogg.tensor.__init__(%{{.*}}) {io = 0 : i32, mut = true} : (!mgp.buffer<<"gpu", 0>>) -> !mo.buffer<[4, 8], f32, {io = 0 : i32, mut = true, stable}>
  %b = mogg.tensor.__init__(%buf) {io = 0 : i32, mut = true}
      : (!mgp.buffer<<"gpu", 0>>) -> !mo.buffer<[4, 8], f32, {io = 0 : i32, mut = true, stable}>
  return
}
