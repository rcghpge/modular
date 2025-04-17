// RUN: mo-opt %s --mo-to-mogg="kernel-library-paths=%kernel_api_test_lib" | FileCheck %s
// TODO(GEX-1573): Convert the following sniff test to a complete test.
// RUN: MOGG_USE_EXP_KERNELS=1 mo-opt %s --mo-to-mogg="kernel-library-paths=%kernel_api_test_lib"

// CHECK-LABEL: mo.graph @supports_scalars
// CHECK: kgen.call @"{{.*}}MOGGKernelAPI::get_scalar_from_managed_tensor_slice
// CHECK: kgen.call @"{{.*}}MOGGKernelAPITest::SupportsScalarKernel::execute
mo.graph @supports_scalars(%a : !mo.tensor<[1], f32>) -> !mo.tensor<[1], f32> {
  %0 = mo.custom {symbol = "supports_scalar_kernel"}(%a, %a) : (!mo.tensor<[1], f32>, !mo.tensor<[1], f32>) -> !mo.tensor<[1], f32>
  mo.output %0 : !mo.tensor<[1], f32>
}
