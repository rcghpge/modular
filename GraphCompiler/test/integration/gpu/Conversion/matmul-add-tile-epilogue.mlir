// Verify that a matmul kernel declaring _FusedComputeOutputTileTensor as its
// output causes the GC to select the ComputeTile epilogue path and emit a
// FusionStruct conforming to ComputeOutputFusionTile.
//
// The stub matmul in testing_kernels/MOGGTileKernelAPITest.mojo overrides the
// built-in mo.matmul via kernel-library-paths, signalling the tile IOSpec.
//
// RUN: mo-opt --gc-pipeline-pre-jit="default-device-label=gpu dump-mojo-text constant-fold=false kernel-library-paths=%tile_kernel_api_test_lib" %s 2>&1 | FileCheck %s

// CHECK: _bind_to_fused_compute_output_tile
// CHECK: ComputeOutputFusionTile

mo.graph @matmul_add_tile_gpu(
    %A: !mo.tensor<[256, 256], f32>,
    %B: !mo.tensor<[256, 256], f32>,
    %bias: !mo.tensor<[1, 256], f32>
) -> !mo.tensor<[256, 256], f32> {
    %ch0 = mo.chain.create()
    %A_gpu, %ch1 = mo.transfer[%ch0] %A : !mo.tensor<[256, 256], f32> to <"gpu", 0>
    %B_gpu, %ch2 = mo.transfer[%ch1] %B : !mo.tensor<[256, 256], f32> to <"gpu", 0>
    %bias_gpu, %ch3 = mo.transfer[%ch2] %bias : !mo.tensor<[1, 256], f32> to <"gpu", 0>
    %mm = mo.matmul(%A_gpu, %B_gpu) { device = #M.device_ref<"gpu", 0> } : (!mo.tensor<[256, 256], f32, gpu:0>, !mo.tensor<[256, 256], f32, gpu:0>) -> !mo.tensor<[256, 256], f32, gpu:0>
    %bcast_bias = mo.static.broadcast_to(%bias_gpu) : !mo.tensor<[1, 256], f32, gpu:0> -> !mo.tensor<[256, 256], f32, gpu:0>
    %result = mo.add(%mm, %bcast_bias) { device = #M.device_ref<"gpu", 0> } : !mo.tensor<[256, 256], f32, gpu:0>
    %result_cpu, %ch4 = mo.transfer[%ch3] %result : !mo.tensor<[256, 256], f32, gpu:0> to <"cpu", 0>
    mo.output %result_cpu : !mo.tensor<[256, 256], f32>
}
