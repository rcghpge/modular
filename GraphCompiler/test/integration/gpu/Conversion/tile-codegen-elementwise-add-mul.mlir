// Verify that a FUSED elementwise chain (mo.add + mo.mul) compiled under
// tile-codegen emits a single ElementwiseTile fusion whose compute chains the
// tile *.elementwise ops: each of the three inputs is copier.copy'd into its
// own dst.layout-shaped staging tile, Add.elementwise combines the first two,
// and Mul.elementwise consumes the Add result (a tile) plus the third input --
// all on TileTensors, driven by one foreach_fusion_tile call, no SIMD.
//
// This exercises the emitter's multi-op chaining and requires tile overloads
// for BOTH Add and Mul (see max/kernels/.../builtin_kernels/elementwise.mojo).
//
// Tile codegen is selected per-device via the `tile_based_fusion` attribute on the
// device's `#M.device_info` entry (the module's `mo.device_info_mapping`).
//
// RUN: mo-opt --gc-pipeline-pre-jit="default-device-label=gpu dump-mojo-text constant-fold=false" %s 2>&1 | FileCheck %s

// No SIMD codegen anywhere under the tile programming model.
// CHECK-NOT: simd_load_from_tensor_pointer
// CHECK-NOT: rebind[SIMD

// One tile foreach driver call for the whole fused chain, no simd_width.
// CHECK: foreach_fusion_tile[target = "gpu", _trace_name = "Elementwise(mo.add, mo.mul)"]
// CHECK-SAME: tensor =
// CHECK-SAME: elem =
// CHECK-SAME: ctx =
// CHECK-NOT: simd_width

// One fusion struct conforming to the tile elementwise trait.
// CHECK: struct FusionStruct{{.*}}(managed_tensor_slice.ElementwiseFusionTile)
// CHECK: def compute[dtype : DType, rank : Int, LayoutType : {{.*}}TensorLayout, Copier : {{.*}}TileCopier]
// CHECK-SAME: dst: {{.*}}TileTensor[dtype, LayoutType, MutAnyOrigin])

// Three DISTINCT inputs: each reconstructed as its own full input tile (distinct
// `spec`/`ptr`), copier.copy'd into its own staging tile, then re-viewed as a
// GENERIC / `MutAnyOrigin` tile for the elementwise chain.
// CHECK: [[SRC0:var[_0-9]+]] = {{.*}}InputTensor[static_spec = Self.spec_0]({{.*}}self.ptr_0{{.*}}).to_tile_tensor().tile[
// CHECK: copier.copy({{.*}}, [[SRC0]].address_space_cast[Copier.src_address_space]())
// CHECK: [[T0:var[_0-9]+]] = {{.*}}TileTensor({{.*}}GENERIC]().unsafe_origin_cast[MutAnyOrigin](), dst.layout)
// CHECK: [[SRC1:var[_0-9]+]] = {{.*}}InputTensor[static_spec = Self.spec_1]({{.*}}self.ptr_1{{.*}}).to_tile_tensor().tile[
// CHECK: copier.copy({{.*}}, [[SRC1]].address_space_cast[Copier.src_address_space]())
// CHECK: [[T1:var[_0-9]+]] = {{.*}}TileTensor({{.*}}GENERIC]().unsafe_origin_cast[MutAnyOrigin](), dst.layout)
// CHECK: [[SRC2:var[_0-9]+]] = {{.*}}InputTensor[static_spec = Self.spec_2]({{.*}}self.ptr_2{{.*}}).to_tile_tensor().tile[
// CHECK: copier.copy({{.*}}, [[SRC2]].address_space_cast[Copier.src_address_space]())
// CHECK: [[T2:var[_0-9]+]] = {{.*}}TileTensor({{.*}}GENERIC]().unsafe_origin_cast[MutAnyOrigin](), dst.layout)

// The chain: Add combines the first two tiles; Mul consumes the Add RESULT
// (a tile) and the third tile -- proving the intermediate is passed as a tile,
// not recomputed or aliased.
// CHECK: [[ADD:var[_0-9]+]] = builtin_kernels.elementwise.Add.elementwise([[T0]], [[T1]])
// CHECK: builtin_kernels.elementwise.Mul.elementwise([[ADD]], [[T2]])
// CHECK: return

module attributes {mo.device_info_mapping = {gpu = #M.device_info<"gpu", "cuda", "sm_100a", "NVIDIA B200", tile_based_fusion = true>}} {
mo.graph @addmul_tile_gpu(
    %A: !mo.tensor<[256, 256], f32>,
    %B: !mo.tensor<[256, 256], f32>,
    %C: !mo.tensor<[256, 256], f32>
) -> !mo.tensor<[256, 256], f32> {
    %ch0 = mo.chain.create()
    %A_gpu, %ch1 = mo.transfer[%ch0] %A : !mo.tensor<[256, 256], f32> to <"gpu", 0>
    %B_gpu, %ch2 = mo.transfer[%ch1] %B : !mo.tensor<[256, 256], f32> to <"gpu", 0>
    %C_gpu, %ch3 = mo.transfer[%ch2] %C : !mo.tensor<[256, 256], f32> to <"gpu", 0>
    %t = mo.add(%A_gpu, %B_gpu) { device = #M.device_ref<"gpu", 0> } : !mo.tensor<[256, 256], f32, gpu:0>
    %r = mo.mul(%t, %C_gpu) { device = #M.device_ref<"gpu", 0> } : !mo.tensor<[256, 256], f32, gpu:0>
    %r_cpu, %ch5 = mo.transfer[%ch3] %r : !mo.tensor<[256, 256], f32, gpu:0> to <"cpu", 0>
    mo.output %r_cpu : !mo.tensor<[256, 256], f32>
}
}
