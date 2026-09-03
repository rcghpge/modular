// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_lib default-device-label=gpu compiled-framework-label=mof" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// End-to-end proof of the trait fused-load parameter-closure path. `execute`
// takes trait args and an `@parameter` closure captures the access-enabled
// view returned by `unsafe_fused_view()`, then passes it down to the device
// launch as a comptime parameter via `@__copy_capture`.
//
// The `mo.add` producer is fused into the trait arg so real fusion state exists.
// With an all-ones input, `x.load(i) == x[i] + x[i] == 2`, and the kernel
// negates, so a working composition returns all -2.
// CHECK: 'trait_fused_load_capture_parameter_closure_graph' returned tensor<4xf32> [-2.000000e+00, -2.000000e+00, -2.000000e+00, -2.000000e+00]
mo.graph @trait_fused_load_capture_parameter_closure_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %p = mo.add(%x_gpu, %x_gpu) : !mo.tensor<[4], f32, gpu:0>
    %z_gpu = mo.custom {symbol = "trait_fused_load_capture_parameter_closure", device = #M.device_ref<"gpu", 0>} (%p) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %z, %ch2 = mo.transfer[%ch1] %z_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
    mo.output %z : !mo.tensor<[4], f32>
}
