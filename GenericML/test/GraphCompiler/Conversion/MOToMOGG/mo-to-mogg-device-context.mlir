// RUN: mo-opt %s --mo-to-mogg=kernel-library-paths="%kernel_api_test_lib" | FileCheck %s
// TODO(GEX-1573): Convert the following sniff test to a complete test.
// RUN: MOGG_USE_EXP_KERNELS=1 mo-opt %s --mo-to-mogg="kernel-library-paths=%kernel_api_test_lib"

// CHECK: %[[CTX:.*]] = mogg.device_context_placeholder {device = #M.device_ref<"gpu", 0>}
// CHECK: %[[TUPLE:.*]] = pop.array.create [%[[CTX]]]
// CHECK: kgen.call @"{{.*}}MOGGKernelAPITest::SingleDeviceContext::execute{{.*}}"(%[[TUPLE]]
mo.graph @single_device_context(%arg0: !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0> {
  %0 = mo.custom {symbol = "single_device_context"}(%arg0) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4], f32, gpu:0>
}

// -----

// CHECK-LABEL: mo.graph @multi_device_context
// CHECK: %[[CTX0:.*]] = mogg.device_context_placeholder {device = #M.device_ref<"gpu", 0>}
// CHECK: %[[CTX1:.*]] = mogg.device_context_placeholder {device = #M.device_ref<"gpu", 1>}
// CHECK: %[[TUPLE:.*]] = pop.array.create [%[[CTX0]], %[[CTX1]]]
// CHECK: kgen.call @"{{.*}}MOGGKernelAPITest::MultiDeviceContext::execute{{.*}}"(%[[TUPLE]]
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
// CHECK: %[[TUPLE:.*]] = pop.array.create [%[[CTX2]], %[[CTX3]]]
// CHECK: kgen.call @"{{.*}}MOGGKernelAPITest::MultiDeviceContextDedup::execute{{.*}}"(%[[TUPLE]]
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

// -----

// CHECK-LABEL: mo.graph @single_variadic_device_context
// CHECK: %[[CTX:.*]] = mogg.device_context_placeholder {device = #M.device_ref<"gpu", 0>}
// CHECK: %[[ARR:.*]] = pop.array.create [%[[CTX]]
// CHECK: kgen.call @"{{.*}}MOGGKernelAPITest::VariadicDeviceContext::execute{{.*}}%[[ARR]]
// CHECK-NOT: mogg.device_context_placeholder

mo.graph @single_variadic_device_context(
    %arg0: !mo.tensor<[4], f32, gpu:0>
) -> !mo.tensor<[4], f32, gpu:0> {
  %0 = mo.custom {symbol = "variadic_device_context"}(%arg0) : (
    !mo.tensor<[4], f32, gpu:0>
  ) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4], f32, gpu:0>
}

// -----

// CHECK-LABEL: mo.graph @multi_variadic_device_context
// CHECK: %[[CTX0:.*]] = mogg.device_context_placeholder {device = #M.device_ref<"gpu", 0>}
// CHECK: %[[CTX1:.*]] = mogg.device_context_placeholder {device = #M.device_ref<"gpu", 1>}
// CHECK: %[[ARR:.*]] = pop.array.create [%[[CTX0]], %[[CTX1]]
// CHECK: kgen.call @"{{.*}}MOGGKernelAPITest::VariadicDeviceContext::execute{{.*}}%[[ARR]]
// CHECK-NOT: mogg.device_context_placeholder

mo.graph @multi_variadic_device_context(
    %arg0: !mo.tensor<[4], f32, gpu:0>,
    %arg1: !mo.tensor<[4], f32, gpu:1>
) -> !mo.tensor<[4], f32, gpu:0> {
  %0 = mo.custom {symbol = "variadic_device_context"}(
    %arg0, %arg1
  ) : (
    !mo.tensor<[4], f32, gpu:0>,
    !mo.tensor<[4], f32, gpu:1>
  ) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4], f32, gpu:0>
}
