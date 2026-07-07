// A graph containing only mo.matmul, compiled with tile-codegen. There
// is no epilogue op, so the matmul's output fusion is the identity: the emitted
// tile compute method returns the matmul's output tile unchanged. Verifies that
// the graph-wide option selects the ComputeTile path and that the primary
// kernel's tile IOSpec passes the consistency assert.
//
// The stub matmul in testing_kernels/tile_stubs/MOGGTileKernelAPITest.mojo
// overrides the built-in mo.matmul via kernel-library-paths.
//
// Tile codegen is selected per-device via the `tile_based_fusion` attribute on the
// device's `#M.device_info` entry (the module's `mo.device_info_mapping`), which
// the graph API sets from the `tile_codegen` compile option.
//
// RUN: mo-opt --gc-pipeline-pre-jit="default-device-label=gpu dump-mojo-text constant-fold=false kernel-library-paths=%tile_kernel_api_test_lib" %s 2>&1 | FileCheck %s

// Tile programming model selected -> tile fusion bound, tile trait, tile method.
// CHECK: _bind_to_fused_compute_output_tile
// CHECK: ComputeOutputFusionTile
// CHECK: def compute[dtype : DType, rank : Int, LayoutType : TensorLayout, copier : TileCopier]
// CHECK-SAME: val: TileTensor[dtype, LayoutType, MutAnyOrigin]
// CHECK-SAME: -> TileTensor[dtype, LayoutType, MutAnyOrigin]

// Identity output fusion: no elementwise op, no SIMD load/store.
// CHECK-NOT: simd_load_from_tensor_pointer
// CHECK-NOT: rebind[SIMD

module attributes {mo.device_info_mapping = {gpu = #M.device_info<"gpu", "cuda", "sm_100a", "NVIDIA B200", tile_based_fusion = true>}} {
mo.graph @matmul_tile_gpu(
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
