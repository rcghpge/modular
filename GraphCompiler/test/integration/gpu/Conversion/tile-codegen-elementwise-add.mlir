// Verify that compiling a standalone elementwise graph with tile-codegen
// selects the ElementwiseTile path: it emits a `foreach_fusion_tile` driver
// call (no `simd_width`) plus a `FusionStruct` conforming to
// `ElementwiseFusionTile` whose `compute(tile_coords, copier, dst)` method
// `copier.copy`s each input into its own `dst.layout`-shaped staging tile and
// returns `Add.elementwise(...)` of the two distinct tiles (no SIMD load/store).
//
// The `foreach_fusion_tile` call is emitted in the main kernel stub, which the
// emitter writes BEFORE the fusion struct definitions (structDefsBuilder is
// concatenated after the main builder), hence the CHECK order below.
//
// Tile codegen is selected per-device via the `tile_based_fusion` attribute on the
// device's `#M.device_info` entry (the module's `mo.device_info_mapping`).
//
// RUN: mo-opt --gc-pipeline-pre-jit="default-device-label=gpu dump-mojo-text constant-fold=false" %s 2>&1 | FileCheck %s

// No SIMD codegen anywhere under the tile programming model.
// CHECK-NOT: simd_load_from_tensor_pointer
// CHECK-NOT: rebind[SIMD

// Driven by the tile foreach driver, with target/_trace_name but no simd_width.
// CHECK: foreach_fusion_tile[target = "gpu", _trace_name = "mo.add"]
// CHECK-SAME: tensor =
// CHECK-SAME: elem =
// CHECK-SAME: ctx =
// CHECK-NOT: simd_width

// The fusion struct conforms to the tile elementwise trait.
// CHECK: struct FusionStruct{{.*}}(managed_tensor_slice.ElementwiseFusionTile)

// The compute method uses TileTensor params (not SIMD) with a `dst` dest tile.
// CHECK: def compute[dtype : DType, rank : Int, LayoutType : TensorLayout, copier : TileCopier]
// CHECK-SAME: tile_coords
// CHECK-SAME: dst: TileTensor[dtype, LayoutType, MutAnyOrigin]
// CHECK-SAME: -> TileTensor[dtype, LayoutType, MutAnyOrigin]

// Each input is copier.copy'd into its OWN dst.layout-shaped staging tile:
// two distinct stack allocations / dest tiles, offset by tile_coords and the
// per-input strides. The Add then combines the two DISTINCT tiles (a + b, not
// a + a -- guards against the CSE-aliasing bug where both inputs collapse onto
// one staging tile).
// CHECK: stack_allocation[LayoutType.static_product, dtype, address_space = copier.dst_address_space]()
// CHECK: [[DST0:var[_0-9]+]] = TileTensor({{.*}}unsafe_origin_cast[MutAnyOrigin](), dst.layout)
// CHECK: tile_coords[0] * dst.dim[0]() * self.strides_0[0]
// CHECK: copier.copy([[DST0]],
// CHECK: stack_allocation[LayoutType.static_product, dtype, address_space = copier.dst_address_space]()
// CHECK: [[DST1:var[_0-9]+]] = TileTensor({{.*}}unsafe_origin_cast[MutAnyOrigin](), dst.layout)
// CHECK: tile_coords[0] * dst.dim[0]() * self.strides_1[0]
// CHECK: copier.copy([[DST1]],
// CHECK: Add.elementwise([[DST0]], [[DST1]])
// CHECK: return

module attributes {mo.device_info_mapping = {gpu = #M.device_info<"gpu", "cuda", "sm_100a", "NVIDIA B200", tile_based_fusion = true>}} {
mo.graph @add_tile_gpu(
    %A: !mo.tensor<[256, 256], f32>,
    %B: !mo.tensor<[256, 256], f32>
) -> !mo.tensor<[256, 256], f32> {
    %ch0 = mo.chain.create()
    %A_gpu, %ch1 = mo.transfer[%ch0] %A : !mo.tensor<[256, 256], f32> to <"gpu", 0>
    %B_gpu, %ch2 = mo.transfer[%ch1] %B : !mo.tensor<[256, 256], f32> to <"gpu", 0>
    %result = mo.add(%A_gpu, %B_gpu) { device = #M.device_ref<"gpu", 0> } : !mo.tensor<[256, 256], f32, gpu:0>
    %result_cpu, %ch4 = mo.transfer[%ch2] %result : !mo.tensor<[256, 256], f32, gpu:0> to <"cpu", 0>
    mo.output %result_cpu : !mo.tensor<[256, 256], f32>
}
}
