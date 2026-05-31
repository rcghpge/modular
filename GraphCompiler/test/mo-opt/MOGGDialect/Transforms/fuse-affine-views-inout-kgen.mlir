// RUN: mo-opt --mogg-import-packages --mo-to-mogg --mogg-fuse-epilogues --canonicalize %s | FileCheck %s

// CHECK-LABEL: mo.graph @many_affine_views
// CHECK: fusionInfo = Epilogue(Affine("mo.matmul":{{.*}}, "mo.add_singleton_dim":{{.*}}, "mo.add_singleton_dim":{{.*}}, "mo.add_singleton_dim":{{.*}}), "mo.activation":{{.*}})
mo.graph @many_affine_views(%arg0: !mo.tensor<[4, 8], f32>, %arg1: !mo.tensor<[8, 8], f32, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[1, 1, 8], f32>) -> !mo.tensor<[4, 1, 1, 1, 8], f32> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[4, 8], f32>, !mo.tensor<[8, 8], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[4, 8], f32>
  %1 = mo.add_singleton_dim[1] (%0) : (!mo.tensor<[4, 8], f32>) -> !mo.tensor<[4, 1, 8], f32>
  %2 = mo.add_singleton_dim[1] (%1) : (!mo.tensor<[4, 1, 8], f32>) -> !mo.tensor<[4, 1, 1, 8], f32>
  %3 = mo.add_singleton_dim[1] (%2) : (!mo.tensor<[4, 1, 1, 8], f32>) -> !mo.tensor<[4, 1, 1, 1, 8], f32>
  %relu = mo.activation(%3) {activation = "relu"}: !mo.tensor<[4, 1, 1, 1, 8], f32>
  mo.output %relu: !mo.tensor<[4, 1, 1, 1, 8], f32>
}
