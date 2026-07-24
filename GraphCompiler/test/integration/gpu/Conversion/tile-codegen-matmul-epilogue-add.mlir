// Verify that compiling with tile-codegen selects the ComputeTile
// epilogue path and emits a FusionStruct conforming to ComputeOutputFusionTile.
// The primary kernel's tile IOSpec is asserted to agree with the graph-wide
// option.
//
// The stub matmul in kernels/tile_stubs/MOGGTileKernelAPITest.mojo
// overrides the built-in mo.matmul via kernel-library-paths and declares the
// tile IOSpec that the assert checks against.
//
// Tile codegen is selected per-device via the `tile_based_fusion` attribute on the
// device's `#M.device_info` entry (the module's `mo.device_info_mapping`), which
// the graph API sets from the `tile_codegen` compile option.
//
// RUN: mo-opt --gc-pipeline-pre-jit="default-device-label=gpu dump-mojo-text constant-fold=false kernel-library-paths=%tile_kernel_api_test_lib" %s 2>&1 | FileCheck %s

// Stub matmul selects the tile epilogue path and binds the tile fusion.
// CHECK: _bind_to_fused_compute_output_tile
// CHECK: ComputeOutputFusionTile

// The bias aux input is captured as a single `TileTensor` field (derived from
// its `spec_1`), not decomposed ptr/shape/strides.
// CHECK: var tile_1: {{.*}}TileTensor[{{.*}}Layout[shape_types = Self.spec_1.static_layout._shape_types{{.*}}]{{.*}}Origin]

// FusionStruct emits a TileTensor-based compute method.
// The method signature uses TileTensor params (not SIMD); the tile trait/type
// names are import-qualified so the generated Mojo compiles.
// CHECK: def compute[dtype : DType, rank : Int, LayoutType : {{.*}}TensorLayout, Copier : {{.*}}TileCopier]
// CHECK-SAME: tile_coords
// CHECK-SAME: val: {{.*}}TileTensor[dtype, LayoutType, MutAnyOrigin])

// The body takes the bias tile from its captured `TileTensor` field
// (`self.tile_1.tile[...]`, no ptr/shape/strides reconstruction), stages it into
// a copier-dst-space tile, and combines it with the matmul output tile `val` —
// copier.copy + Add.elementwise, no SIMD.
// CHECK-NOT: simd_load_from_tensor_pointer
// CHECK-NOT: rebind[SIMD
// CHECK: stack_allocation[LayoutType.static_product, dtype, address_space = Copier.dst_address_space]()
// CHECK: self.tile_1.tile[
// CHECK: copier.copy(
// CHECK: Add.elementwise(val,
// CHECK: return

module attributes {mo.device_info_mapping = {gpu = #M.device_info<"gpu", "cuda", "sm_100a", "NVIDIA B200", tile_based_fusion = true>}} {
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
}
