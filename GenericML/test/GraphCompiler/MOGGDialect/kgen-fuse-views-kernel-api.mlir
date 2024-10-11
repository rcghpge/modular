// RUN: mo-opt --mo-to-mogg=kernel-library-paths="%kernel_api_lib" --fuse-views=kernel-library-paths="%kernel_api_lib" %s | FileCheck %s

mo.graph @back2backBroadcastReshape<N, M, K, J>(%arg0: !mo.tensor<[K, M], f32>, %arg1: !mo.tensor<[], f32>) -> (!mo.tensor<[K, M], f32>) {
  %bcast = mo.static.broadcast_to(%arg1) : !mo.tensor<[], f32> -> !mo.tensor<[J], f32>
  %reshape = mo.static.reshape(%bcast) : !mo.tensor<[J], f32> -> !mo.tensor<[K, M], f32>
  %1 = mo.add(%arg0, %reshape) : !mo.tensor<[K, M], f32>
  mo.output %1 : !mo.tensor<[K, M], f32>
}

// CHECK-LABEL:   mo.graph @back2backBroadcastReshape
// CHECK: summary = "mo.static.broadcast_to
// CHECK: summary = "mo.static.reshape{{.*}}mo.add
