// RUN: mo-opt %s --mo-to-mogg=kernel-library-paths="%kernel_api_test_lib" --fuse-in-out-lambdas=kernel-library-paths="%kernel_api_test_lib" --cse | FileCheck %s

// CHECK: module {
// CHECK:   kgen.extern.generator @"MOGGKernelAPITestLib::MOGGKernelAPITest::TestCustomOp::execute

// CHECK:       kgen.call @"MOGGKernelAPILib::MOGGKernelAPI::Ceil::execute
// CHECK:       kgen.call @"MOGGKernelAPITestLib::MOGGKernelAPITest::TestCustomOp::execute
// CHECK:       kgen.call @"MOGGKernelAPILib::MOGGKernelAPI::ReLU::execute
// CHECK:       mogg.output %{{.*}} : !mo.tensor<[2], f32>

mo.graph @input_output_fusion(%arg0: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
  %0 = mo.ceil(%arg0) : !mo.tensor<[2], f32>
  %1 = mo.custom {symbol = "test_custom_op"}(%0) : (!mo.tensor<[2], f32>) -> !mo.tensor<[2], f32>
  %2 = mo.relu(%1) : !mo.tensor<[2], f32>
  mo.output %2 : !mo.tensor<[2], f32>
}
