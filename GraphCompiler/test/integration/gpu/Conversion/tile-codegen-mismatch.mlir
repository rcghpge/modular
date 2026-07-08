// Consistency assert for the tile programming model: when a device is configured
// for tile codegen but the resolved primary kernel is not a tile kernel, the
// compiler must emit a clear error rather than binding a tile fusion struct to a
// SIMD kernel. Here the built-in mo.matmul (a SIMD compute-output kernel) runs
// on a device whose `#M.device_info` has `tile_based_fusion = true`, and whose
// arch (sm_90a) has no built-in tile mo.matmul registration, so the assert
// fires. (The transitional tile mo.matmul is registered only for sm_100a; see
// GEX-3941 / GEX-3919. On sm_100a the selection filter would instead pick that
// tile kernel and no mismatch would occur.)
//
// RUN: not mo-opt --gc-pipeline-pre-jit="default-device-label=gpu dump-mojo-text constant-fold=false" %s 2>&1 | FileCheck %s

// CHECK: device 'gpu' is configured for tile codegen (tile_based_fusion) but kernel
// CHECK-SAME: declares a SIMD compute output (_FusedComputeOutputTensor)

module attributes {mo.device_info_mapping = {gpu = #M.device_info<"gpu", "cuda", "sm_90a", "NVIDIA H100", tile_based_fusion = true>}} {
mo.graph @matmul_tile_mismatch_gpu(
    %A: !mo.tensor<[256, 256], f32>,
    %B: !mo.tensor<[256, 256], f32>
) -> !mo.tensor<[256, 256], f32> {
    %ch0 = mo.chain.create()
    %A_gpu, %ch1 = mo.transfer[%ch0] %A : !mo.tensor<[256, 256], f32> to <"gpu", 0>
    %B_gpu, %ch2 = mo.transfer[%ch1] %B : !mo.tensor<[256, 256], f32> to <"gpu", 0>
    %mm = mo.matmul(%A_gpu, %B_gpu) { device = #M.device_ref<"gpu", 0> } : (!mo.tensor<[256, 256], f32, gpu:0>, !mo.tensor<[256, 256], f32, gpu:0>) -> !mo.tensor<[256, 256], f32, gpu:0>
    %result_cpu, %ch3 = mo.transfer[%ch2] %mm : !mo.tensor<[256, 256], f32, gpu:0> to <"cpu", 0>
    mo.output %result_cpu : !mo.tensor<[256, 256], f32>
}
}
