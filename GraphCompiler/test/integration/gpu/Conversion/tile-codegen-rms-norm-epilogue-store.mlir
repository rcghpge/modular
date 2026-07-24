// Verify that compiling with tile-codegen selects the OutputTile (store)
// epilogue path and emits a FusionStruct conforming to OutputFusionTile whose
// `store` method OWNS the store (the primary kernel does not store).
//
// The store variant is exercised via `mo.reduce.rms_norm` (not `mo.matmul`):
// the graph compiler registers one primary kernel per op, so a kernel author
// chooses store or compute fusion per op. `mo.matmul`'s tile kernel uses the
// compute variant (see tile-codegen-matmul-epilogue-add.mlir); the store stub
// here declares `mo.reduce.rms_norm` with a `_FusedOutputTileTensor` output,
// which flips the fused epilogue onto a `TileTensor` `store` method.
//
// The stub lives in kernels/tile_stubs/MOGGTileStoreKernelAPITest.mojo (same
// library as the matmul compute stub; a different op, so no registration
// collision) and is pulled in via kernel-library-paths.
//
// Tile codegen is selected per-device via the `tile_based_fusion` attribute on
// the device's `#M.device_info` entry, which the graph API sets from the
// `tile_codegen` compile option.

// RUN: mo-opt --gc-pipeline-pre-jit="default-device-label=gpu dump-mojo-text constant-fold=false kernel-library-paths=%tile_kernel_api_test_lib" %s 2>&1 | FileCheck %s

// Stub rms_norm selects the tile store path and binds the tile store fusion.
// CHECK: _bind_to_fused_output_tile
// CHECK: managed_tensor_slice.OutputFusionTile

// The output tensor (capture 0) and the broadcast scale aux input (capture 1)
// are each captured as a single `TileTensor` field (derived from their
// `spec_N`), not decomposed ptr/shape/strides.
// CHECK: var tile_0: {{.*}}TileTensor[{{.*}}Layout[shape_types = Self.spec_0.static_layout._shape_types{{.*}}]{{.*}}Origin]
// CHECK: var tile_1: {{.*}}TileTensor[{{.*}}Layout[shape_types = Self.spec_1.static_layout._shape_types{{.*}}]{{.*}}Origin]

// FusionStruct emits a TileTensor-based `store` method (not `compute`): the
// tile signature takes `tile_coords`, the aux-load `Copier`, a kernel-supplied
// `StoreCopier`, and the output tile `val`, and is terminal -- no TileTensor
// return. Both copiers are method parameters (chosen by the kernel), so the
// emitter never synthesizes a store copier of its own.
// CHECK: def store[dtype : DType, rank : Int, LayoutType : {{.*}}TensorLayout, Copier : {{.*}}TileCopier, StoreCopier : {{.*}}TileCopier]
// CHECK-SAME: tile_coords
// CHECK-SAME: store_copier: StoreCopier
// CHECK-SAME: val: {{.*}}TileTensor[dtype, LayoutType, MutAnyOrigin])

// The body takes the scale tile from its captured `TileTensor` field
// (`self.tile_1.tile[...]`), stages it into a copier-dst-space tile, combines it
// with the norm output tile `val` (Mul.elementwise, no SIMD), then gathers the
// fragment into the output tile -- taken from `self.tile_0.tile[...]` (no
// ptr/shape/strides reconstruction) -- via the passed-in `store_copier`.
// CHECK-NOT: simd_load_from_tensor_pointer
// CHECK-NOT: simd_store_into_tensor_pointer
// CHECK-NOT: rebind[SIMD
// CHECK: stack_allocation[LayoutType.static_product, dtype, address_space = Copier.dst_address_space]()
// CHECK: self.tile_1.tile[
// CHECK: copier.copy(
// CHECK: Mul.elementwise(val,
// CHECK: store_copier.copy(self.tile_0.tile[

module attributes {mo.device_info_mapping = {gpu = #M.device_info<"gpu", "cuda", "sm_100a", "NVIDIA B200", tile_based_fusion = true>}} {
mo.graph @rms_norm_mul_tile_gpu(
    %x: !mo.tensor<[256, 256], f32>,
    %gamma: !mo.tensor<[256], f32>,
    %scale: !mo.tensor<[1, 256], f32>
) -> !mo.tensor<[256, 256], f32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[256, 256], f32> to <"gpu", 0>
    %gamma_gpu, %ch2 = mo.transfer[%ch1] %gamma : !mo.tensor<[256], f32> to <"gpu", 0>
    %scale_gpu, %ch3 = mo.transfer[%ch2] %scale : !mo.tensor<[1, 256], f32> to <"gpu", 0>
    %eps = mo.constant {value = #M.dense_array<0.00001> : tensor<f32>} : !mo.tensor<[], f32>
    %woff = mo.constant {value = #M.dense_array<0.0> : tensor<f32>} : !mo.tensor<[], f32>
    %norm = mo.reduce.rms_norm(%x_gpu, %gamma_gpu, %eps, %woff) {multiply_before_cast = true} : (!mo.tensor<[256, 256], f32, gpu:0>, !mo.tensor<[256], f32, gpu:0>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[256, 256], f32, gpu:0>
    %scale_bcast = mo.static.broadcast_to(%scale_gpu) : !mo.tensor<[1, 256], f32, gpu:0> -> !mo.tensor<[256, 256], f32, gpu:0>
    %result = mo.mul(%norm, %scale_bcast) { device = #M.device_ref<"gpu", 0> } : !mo.tensor<[256, 256], f32, gpu:0>
    %result_cpu, %ch4 = mo.transfer[%ch3] %result : !mo.tensor<[256, 256], f32, gpu:0> to <"cpu", 0>
    mo.output %result_cpu : !mo.tensor<[256, 256], f32>
}
}
