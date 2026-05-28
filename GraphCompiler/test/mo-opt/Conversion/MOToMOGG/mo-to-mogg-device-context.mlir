// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-mogg | FileCheck %s
// TODO(GEX-1573): Convert the following sniff test to a complete test.

// CHECK: mogg.kernel
// CHECK: mogg.call.execute["{{.*}}.SingleDeviceContext"]
mo.graph @single_device_context(%arg0: !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0> {
  %0 = mo.custom {symbol = "single_device_context", device = #M.device_ref<"cpu", 0>}(%arg0) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4], f32, gpu:0>
}

// -----

// CHECK-LABEL: mo.graph @multi_device_context
// CHECK: mogg.kernel
// CHECK: mogg.call.execute["{{.*}}.MultiDeviceContext"]
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
// CHECK: mogg.kernel
// CHECK: mogg.call.execute["{{.*}}.MultiDeviceContextDedup"]
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
// CHECK: mogg.kernel
// CHECK: mogg.as_tuple
// CHECK: mogg.call.execute["{{.*}}.VariadicDeviceContext"]

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
// CHECK: mogg.kernel
// CHECK: mogg.as_tuple
// CHECK: mogg.call.execute["{{.*}}.VariadicDeviceContext"]

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
