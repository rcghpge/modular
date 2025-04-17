// RUN: mo-opt %s --mo-to-mogg=kernel-library-paths="%kernel_api_test_lib" --verify-diagnostics

mo.graph @too_many_devices(
    %arg0: !mo.tensor<[4], f32, gpu:0>,
    %arg1: !mo.tensor<[4], f32, gpu:1>
) -> !mo.tensor<[4], f32, gpu:2> {
  // expected-error @+1 {{'mo.custom' op [MO_TO_MOGG] Device context argument count (2) must match unique devices (3)}}
  %0 = mo.custom {symbol = "multi_device_context_dedup"}(%arg0, %arg1) : (
    !mo.tensor<[4], f32, gpu:0>,
    !mo.tensor<[4], f32, gpu:1>
  ) -> !mo.tensor<[4], f32, gpu:2>
  mo.output %0 : !mo.tensor<[4], f32, gpu:2>
}
