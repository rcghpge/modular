// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-mogg --verify-diagnostics

mo.graph @too_many_devices(
    %arg0: !mo.tensor<[4], f32, gpu:0>,
    %arg1: !mo.tensor<[4], f32, gpu:1>
) -> !mo.tensor<[4], f32, gpu:2> {
  %0 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "multi_device_context_dedup"}(%arg0, %arg1) : (
    !mo.tensor<[4], f32, gpu:0>,
    !mo.tensor<[4], f32, gpu:1>
  ) -> !mo.tensor<[4], f32, gpu:2>
  mo.output %0 : !mo.tensor<[4], f32, gpu:2>
}
