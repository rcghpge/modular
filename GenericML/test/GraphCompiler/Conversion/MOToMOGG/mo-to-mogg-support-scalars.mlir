// RUN: mo-opt %s --mo-to-mogg="kernel-library-paths=%kernel_api_test_lib" | FileCheck %s
// TODO(GEX-1573): Convert the following sniff test to a complete test.

// CHECK-LABEL: mo.graph @supports_scalars
// CHECK: mogg.experimental.kernel
// CHECK: mogg.as_scalar
// CHECK: mogg.call.execute["supports_scalar_kernel"]
mo.graph @supports_scalars(%a : !mo.tensor<[], f32>) -> !mo.tensor<[], f32> {
  %0 = mo.custom {symbol = "supports_scalar_kernel", device = #M.device_ref<"cpu", 0>}(%a, %a) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], f32>
  mo.output %0 : !mo.tensor<[], f32>
}
