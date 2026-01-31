// RUN: mo-opt --mo-to-mogg --mogg-fuse-affine-views --canonicalize --mogg-fuse-prologues-epilogues --canonicalize %s | FileCheck %s

// CHECK-LABEL: mo.graph @many_affine_views
// CHECK: summary = "mo.matmul{{.*}}mo.add_singleton_dim{{.*}}mo.add_singleton_dim{{.*}}mo.add_singleton_dim{{.*}}mo.relu{{.*}}"
mo.graph @many_affine_views(%arg0: !mo.tensor<[4, 8], f32>, %arg1: !mo.tensor<[8, 8], f32, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[1, 1, 8], f32>) -> !mo.tensor<[4, 1, 1, 1, 8], f32> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[4, 8], f32>, !mo.tensor<[8, 8], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[4, 8], f32>
  %1 = mo.add_singleton_dim[1] (%0) : (!mo.tensor<[4, 8], f32>) -> !mo.tensor<[4, 1, 8], f32>
  %2 = mo.add_singleton_dim[1] (%1) : (!mo.tensor<[4, 1, 8], f32>) -> !mo.tensor<[4, 1, 1, 8], f32>
  %3 = mo.add_singleton_dim[1] (%2) : (!mo.tensor<[4, 1, 1, 8], f32>) -> !mo.tensor<[4, 1, 1, 1, 8], f32>
  %relu = mo.relu(%3): !mo.tensor<[4, 1, 1, 1, 8], f32>
  mo.output %relu: !mo.tensor<[4, 1, 1, 1, 8], f32>
}
