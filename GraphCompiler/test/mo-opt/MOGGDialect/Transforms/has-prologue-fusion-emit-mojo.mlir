// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-mogg --mogg-simplify-kernel-operands --mogg-fuse-elementwise --canonicalize --cse --mogg-fuse-views --canonicalize --cse --mogg-fuse-prologues --mogg-to-mgp --mogg-outline-kernels --mogg-add-try-except-blocks --mogg-emit-mojo --mgp-dump-mojo 2>&1 | FileCheck %s

// Verifies the `has_prologue_fusion` comptime parameter the graph compiler injects
// into kernels that declare it (getHasPrologueFusionValue in EmitMojoState.cpp).
// It is False iff nothing is fused into the kernel's input (the fusion tree is
// a single leaf), and True whenever any producer -- elementwise OR a
// coordinate-remapping view -- is folded in.

// (1) Plain softmax over a materialized input
// CHECK: Softmax.execute[{{.*}}has_prologue_fusion = False{{.*}}]
mo.graph @softmax_plain(%arg0: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mo.reduce.softmax(%arg0) {axis = 1 : index} : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// (2) An elementwise producer (mul) fuses into softmax's input lambda
// CHECK: Softmax.execute[{{.*}}has_prologue_fusion = True{{.*}}]
mo.graph @softmax_elementwise_fused(%arg0: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %m = mo.mul(%arg0, %arg0) : !mo.tensor<[2, 3], f32>
  %0 = mo.reduce.softmax(%m) {axis = 1 : index} : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// (3) A slice view fuses into softmax's input -> View group
// CHECK: Softmax.execute[{{.*}}has_prologue_fusion = True{{.*}}]
mo.graph @softmax_slice_fused(%arg0: !mo.tensor<[3, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %start = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop = mo.constant {value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %s = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[3, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  %0 = mo.reduce.softmax(%s) {axis = 1 : index} : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// (4) A transpose view fuses into softmax's input -> View group.
//     Transpose swaps strides, so the input is non-contiguous
// CHECK: Softmax.execute[{{.*}}has_prologue_fusion = True{{.*}}]
mo.graph @softmax_transpose_fused(%arg0: !mo.tensor<[3, 2], f32>) -> !mo.tensor<[2, 3], f32> {
  %perm = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %t = mo.transpose(%arg0, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  %0 = mo.reduce.softmax(%t) {axis = 1 : index} : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// (5) A broadcast view fuses into softmax's input -> View group.
// CHECK: Softmax.execute[{{.*}}has_prologue_fusion = True{{.*}}]
mo.graph @softmax_broadcast_fused(%arg0: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %b = mo.static.broadcast_to(%arg0) : !mo.tensor<[1, 3], f32> -> !mo.tensor<[2, 3], f32>
  %0 = mo.reduce.softmax(%b) {axis = 1 : index} : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// (6) The same flag is injected on LogSoftmax. Plain input -> leaf.
// CHECK: LogSoftmax.execute[{{.*}}has_prologue_fusion = False{{.*}}]
mo.graph @logsoftmax_plain(%arg0: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mo.reduce.logsoftmax(%arg0) {axis = 1 : index} : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// (7) LogSoftmax with an elementwise producer fused into its input.
// CHECK: LogSoftmax.execute[{{.*}}has_prologue_fusion = True{{.*}}]
mo.graph @logsoftmax_elementwise_fused(%arg0: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %m = mo.mul(%arg0, %arg0) : !mo.tensor<[2, 3], f32>
  %0 = mo.reduce.logsoftmax(%m) {axis = 1 : index} : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}
