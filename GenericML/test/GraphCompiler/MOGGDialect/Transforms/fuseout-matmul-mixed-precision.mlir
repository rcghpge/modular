
// RUN: mo-opt --mo-to-mogg --canonicalize --fuse-in-out-lambdas --canonicalize --cse %s | FileCheck %s

// CHECK-LABEL: mo.graph @matmul_mixed

// CHECK: [[OUTPUT:%.*]] = mogg.output_placeholders : !mo.tensor<[24, 24], si32>
// CHECK: mogg.scratch_placeholders : !mo.tensor<[24, 24], f32>

// The output lambda
// CHECK: kgen.param.declare.region MOGG_{{.*}}_output_0_fn =
// Ensure we get rid of the original store
// CHECK-NOT: kgen.call @"tensor_internal::managed_tensor_slice::simd_store_into_managed_tensor_slice
// CHECK: kgen.call @"{{.*}}MOGGKernelAPI::Cast::execute
// CHECK-NEXT: kgen.call @"tensor_internal::managed_tensor_slice::simd_store_into_managed_tensor_slice
// CHECK-NEXT: kgen.return
// CHECK-NEXT: } {sym_name = "MOGG_{{.*}}_output_0_fn"}

// CHECK: mogg.output [[OUTPUT]] : !mo.tensor<[24, 24], si32>

mo.graph @matmul_mixed(%arg0: !mo.tensor<[24, 24], f32>, %arg1: !mo.tensor<[24, 24], f32, {layout = #mo.layout<KNkni>}>) -> !mo.tensor<[24, 24], si32> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[24, 24], f32>, !mo.tensor<[24, 24], f32, {layout = #mo.layout<KNkni>}>) -> !mo.tensor<[24, 24], f32>
  %1 = mo.cast(%0) : (!mo.tensor<[24, 24], f32>) -> !mo.tensor<[24, 24], si32>
  mo.output %1: !mo.tensor<[24, 24], si32>
}
