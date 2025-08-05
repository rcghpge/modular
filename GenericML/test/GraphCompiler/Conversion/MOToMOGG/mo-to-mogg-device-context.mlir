// RUN: mo-opt %s --mo-to-mogg=kernel-library-paths="%kernel_api_test_lib" | FileCheck %s
// TODO(GEX-1573): Convert the following sniff test to a complete test.

// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.execute["single_device_context"]
mo.graph @single_device_context(%arg0: !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0> {
  %0 = mo.custom {symbol = "single_device_context", device = #M.device_ref<"cpu", 0>}(%arg0) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4], f32, gpu:0>
}

// -----

// CHECK-LABEL: mo.graph @multi_device_context
// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.execute["multi_device_context"]
mo.graph @multi_device_context(
    %arg0: !mo.tensor<[4], f32, gpu:0>
) -> !mo.tensor<[4], f32, gpu:1> {
  %0 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "multi_device_context"}(%arg0) : (
    !mo.tensor<[4], f32, gpu:0>
  ) -> !mo.tensor<[4], f32, gpu:1>
  mo.output %0 : !mo.tensor<[4], f32, gpu:1>
}


// -----

// CHECK-LABEL: mo.graph @multi_device_context_dedup
// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.execute["multi_device_context_dedup"]
mo.graph @multi_device_context_dedup(
    %arg0: !mo.tensor<[4], f32, gpu:0>,
    %arg1: !mo.tensor<[4], f32, gpu:1>
) -> !mo.tensor<[4], f32, gpu:0> {
  %0 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "multi_device_context_dedup"}(%arg0, %arg1) : (
    !mo.tensor<[4], f32, gpu:0>,
    !mo.tensor<[4], f32, gpu:1>
  ) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4], f32, gpu:0>
}

// -----

// CHECK-LABEL: mo.graph @single_variadic_device_context
// CHECK: mogg.experimental.kernel
// CHECK: mogg.as_tuple
// CHECK: mogg.call.execute["variadic_device_context"]

mo.graph @single_variadic_device_context(
    %arg0: !mo.tensor<[4], f32, gpu:0>
) -> !mo.tensor<[4], f32, gpu:0> {
  %0 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "variadic_device_context"}(%arg0) : (
    !mo.tensor<[4], f32, gpu:0>
  ) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4], f32, gpu:0>
}

// -----

// CHECK-LABEL: mo.graph @multi_variadic_device_context
// CHECK: mogg.experimental.kernel
// CHECK: mogg.as_tuple
// CHECK: mogg.call.execute["variadic_device_context"]

mo.graph @multi_variadic_device_context(
    %arg0: !mo.tensor<[4], f32, gpu:0>,
    %arg1: !mo.tensor<[4], f32, gpu:1>
) -> !mo.tensor<[4], f32, gpu:0> {
  %0 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "variadic_device_context"}(
    %arg0, %arg1
  ) : (
    !mo.tensor<[4], f32, gpu:0>,
    !mo.tensor<[4], f32, gpu:1>
  ) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4], f32, gpu:0>
}
