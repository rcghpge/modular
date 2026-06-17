// RUN: mo-opt --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-mogg %s --verify-diagnostics

mo.graph @gpu_only_kernel_on_cpu(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    // expected-error @+1 {{no kernel matching device info for 'gpu_only_kernel' on device #M.device_info<"cpu", "unknown", "unknown", "unknown">}}
    %out = mo.custom {symbol = "gpu_only_kernel", device = #M.device_ref<"cpu", 0>} (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    mo.output %out : !mo.tensor<[4], f32>
}
