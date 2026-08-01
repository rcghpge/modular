
// RUN: mo-opt --mogg-import-packages --mo-to-mogg --canonicalize --mogg-fuse-epilogues --canonicalize --cse %s | FileCheck %s

// CHECK-LABEL: mo.graph @matmul_mixed

// CHECK: mogg.output_placeholders : !mo.tensor<[24, 24], si32>
// CHECK: mogg.scratch_placeholders : !mo.tensor<[24, 24], f32>
// CHECK: mogg.tensor.fused


mo.graph @matmul_mixed(%arg0: !mo.tensor<[24, 24], f32>, %arg1: !mo.tensor<[24, 24], f32, {layout = #mo.layout<KNkni>}>) -> !mo.tensor<[24, 24], si32> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[24, 24], f32>, !mo.tensor<[24, 24], f32, {layout = #mo.layout<KNkni>}>) -> !mo.tensor<[24, 24], f32>
  %1 = mo.cast(%0) : (!mo.tensor<[24, 24], f32>) -> !mo.tensor<[24, 24], si32>
  mo.output %1: !mo.tensor<[24, 24], si32>
}
