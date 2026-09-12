// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-to-mogg --mogg-conform-call-operands | FileCheck %s

// ConformCallOperands wraps each trait-bound call operand in a
// `mogg._trait.create` carrying the distilled contract, so a raw
// `!mogg.tensor_` (which reconstructs as a `ManagedTensorSlice`, conforming to
// none of `Tensor`/`DenseTensor`/`TileTensorable`) reaches the kernel as the
// conforming trait argument. The wrap is materialized in the IR here; the call
// op's own emit stays a dumb forward. `trait_negate` declares both its input
// and its output against `DenseTensor`, so both operands are wrapped -- with the
// input carrying `role = input` and the output `role = output` -- and the
// execution device rides on the op for local alignment at emit.

// CHECK-LABEL: mo.graph @trait_negate_graph
// CHECK: %[[INTENSOR:.*]] = mogg._tensor.create(%{{.*}}, %{{.*}}) {{.*}}mut = false{{.*}} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK: %[[OUTTENSOR:.*]] = mogg._tensor.create(%{{.*}}, %{{.*}}) {{.*}}mut = true{{.*}} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK: %[[INARG:.*]] = mogg._trait.create(%[[INTENSOR]]) {contract = #mogg.tensor_arg<{role = input, fused = false, access = none, capabilities = none}>, device = {{.*}}} : (!mogg.tensor_) -> !mogg.tensor_
// CHECK: %[[OUTARG:.*]] = mogg._trait.create(%[[OUTTENSOR]]) {contract = #mogg.tensor_arg<{role = output, fused = false, access = none, capabilities = none}>, device = {{.*}}} : (!mogg.tensor_) -> !mogg.tensor_
// CHECK: mogg._call.opaque @{{.*}}TraitNegate inputs(%[[INARG]] : !mogg.tensor_) outputs(%[[OUTARG]] : !mogg.tensor_) context(
mo.graph @trait_negate_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %z = mo.custom {symbol = "trait_negate", device = #M.device_ref<"cpu", 0>} (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %z : !mo.tensor<[4], f32>
}
