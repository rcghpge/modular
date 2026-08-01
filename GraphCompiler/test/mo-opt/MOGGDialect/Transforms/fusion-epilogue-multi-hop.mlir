// RUN: mo-opt --mogg-import-packages --mo-to-mogg --mogg-fuse-elementwise --mogg-simplify-kernel-operands --mogg-fuse-views --mogg-simplify-kernel-operands --mogg-fuse-prologues --mogg-simplify-kernel-operands --mogg-fuse-epilogues --mogg-simplify-kernel-operands --mogg-strip-kernel-decls %s | FileCheck %s

// Tests that FuseEpilogues fuses epilogues 4 hops away through alternating
// affine views. FuseEpilogues runs epilogue fusion and affine view fusion in a
// loop until convergence, enabling:
//   matmul -> add -> merge_dim -> add -> split_dim -> add -> merge_dim -> add -> split_dim -> add
// to become fully fused in a single kernel.

// CHECK-LABEL: mo.graph @matmul_add_4hop
// Single fused kernel: all ops absorbed
// CHECK: mogg.kernel
// CHECK-NOT: mogg.kernel{{[^.]}}
mo.graph @matmul_add_4hop(%arg0: !mo.tensor<[4, 8], f32>, %arg1: !mo.tensor<[8, 8], f32, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[4, 8], f32>, %arg3: !mo.tensor<[32], f32>, %arg4: !mo.tensor<[4, 8], f32>, %arg5: !mo.tensor<[32], f32>, %arg6: !mo.tensor<[4, 8], f32>) -> !mo.tensor<[4, 8], f32> {
  %mm = mo.matmul(%arg0, %arg1) : (!mo.tensor<[4, 8], f32>, !mo.tensor<[8, 8], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[4, 8], f32>
  %add1 = mo.add(%mm, %arg2) : !mo.tensor<[4, 8], f32>
  %merge1 = mo.merge_dim[0](%add1) : (!mo.tensor<[4, 8], f32>) -> !mo.tensor<[32], f32>
  %add2 = mo.add(%merge1, %arg3) : !mo.tensor<[32], f32>
  %split1 = mo.split_dim[0](%add2) : (!mo.tensor<[32], f32>) -> !mo.tensor<[4, 8], f32>
  %add3 = mo.add(%split1, %arg4) : !mo.tensor<[4, 8], f32>
  %merge2 = mo.merge_dim[0](%add3) : (!mo.tensor<[4, 8], f32>) -> !mo.tensor<[32], f32>
  %add4 = mo.add(%merge2, %arg5) : !mo.tensor<[32], f32>
  %split2 = mo.split_dim[0](%add4) : (!mo.tensor<[32], f32>) -> !mo.tensor<[4, 8], f32>
  %add5 = mo.add(%split2, %arg6) : !mo.tensor<[4, 8], f32>
  mo.output %add5 : !mo.tensor<[4, 8], f32>
}
