// RUN: mo-opt --mo-to-mogg --fuse-elementwise --cse --canonicalize --fuse-bcasts --cse --canonicalize --translate-mogg-to-kgen --inline-lambdas --mo-to-primitives --lower-primitives --remove-non-lowerable-results --lower-abi --canonicalize -cse --jit-compile-kernels --lower-mogg-region-compile %s | FileCheck %s

// Check that with many broadcasts of the same value we still produce a unique kernel for each

mo.graph @many_bcasts_same_val(%arg0: !mo.tensor<[1, 100], f32>) -> !mo.tensor<[1, 65], f32> {
  %0 = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
  %bcast1 = mo.static.broadcast_to(%0) : !mo.tensor<[], f32> -> !mo.tensor<[100, 300], f32, {layout = #mo.layout<KN>}>
  %4 = mo.matmul(%arg0, %bcast1) : (!mo.tensor<[1, 100], f32>, !mo.tensor<[100, 300], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[1, 300], f32>
  %bcast2 = mo.static.broadcast_to(%0) : !mo.tensor<[], f32> -> !mo.tensor<[1, 300], f32>
  %6 = mo.add(%4, %bcast2) : !mo.tensor<[1, 300], f32>
  %bcast3 = mo.static.broadcast_to(%0) : !mo.tensor<[], f32> -> !mo.tensor<[300, 10], f32, {layout = #mo.layout<KN>}>
  %7 = mo.matmul(%6, %bcast3) : (!mo.tensor<[1, 300], f32>, !mo.tensor<[300, 10], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[1, 10], f32>

  %bcast4 = mo.static.broadcast_to(%0) : !mo.tensor<[], f32> -> !mo.tensor<[10, 65], f32, {layout = #mo.layout<KN>}>
  %8 = mo.matmul(%7, %bcast4) : (!mo.tensor<[1, 10], f32>, !mo.tensor<[10, 65], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[1, 65], f32>
  mo.output %8 : !mo.tensor<[1, 65], f32>
}


// CHECK: mgp.module.load
// CHECK-NOT: mgp.module.load
