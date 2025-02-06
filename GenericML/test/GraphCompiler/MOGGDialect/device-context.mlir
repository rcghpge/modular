// RUN: mo-opt %s --mo-to-mogg=kernel-library-paths="%kernel_api_test_lib" --verify-diagnostics | FileCheck %s

// CHECK: %[[CTX:.*]] = mogg.device_context_placeholder {device = #M.device_ref<"gpu", 0>}
// CHECK: kgen.call @"{{.*}}MOGGKernelAPITest::SingleDeviceContext::execute{{.*}}"(%[[CTX]]
mo.graph @single_device_context(%arg0: !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0> {
  %0 = mo.custom {symbol = "single_device_context"}(%arg0) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4], f32, gpu:0>
}

// -----

// CHECK-LABEL: mo.graph @multi_device_context
// CHECK: %[[CTX0:.*]] = mogg.device_context_placeholder {device = #M.device_ref<"gpu", 0>}
// CHECK: %[[CTX1:.*]] = mogg.device_context_placeholder {device = #M.device_ref<"gpu", 1>}
// CHECK: kgen.call @"{{.*}}MOGGKernelAPITest::MultiDeviceContext::execute{{.*}}"(%[[CTX0]], %{{.*}}, %[[CTX1]]
mo.graph @multi_device_context(
    %arg0: !mo.tensor<[4], f32, gpu:0>
) -> !mo.tensor<[4], f32, gpu:1> {
  %0 = mo.custom {symbol = "multi_device_context"}(%arg0) : (
    !mo.tensor<[4], f32, gpu:0>
  ) -> !mo.tensor<[4], f32, gpu:1>
  mo.output %0 : !mo.tensor<[4], f32, gpu:1>
}


// -----

// CHECK-LABEL: mo.graph @multi_device_context_dedup
// CHECK: %[[CTX2:.*]] = mogg.device_context_placeholder {device = #M.device_ref<"gpu", 0>}
// CHECK: %[[CTX3:.*]] = mogg.device_context_placeholder {device = #M.device_ref<"gpu", 1>}
// CHECK-NOT: mogg.device_context_placeholder
// CHECK: kgen.call @"{{.*}}MOGGKernelAPITest::MultiDeviceContextDedup::execute{{.*}}"(%[[CTX2]], %{{.*}}, %[[CTX3]]
mo.graph @multi_device_context_dedup(
    %arg0: !mo.tensor<[4], f32, gpu:0>,
    %arg1: !mo.tensor<[4], f32, gpu:1>
) -> !mo.tensor<[4], f32, gpu:0> {
  %0 = mo.custom {symbol = "multi_device_context_dedup"}(%arg0, %arg1) : (
    !mo.tensor<[4], f32, gpu:0>,
    !mo.tensor<[4], f32, gpu:1>
  ) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4], f32, gpu:0>
}
