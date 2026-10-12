// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-pipeline --map-to-mogg --mogg-pipeline --mogg-to-mgp | FileCheck %s
// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-pipeline --map-to-mogg --mogg-pipeline --mogg-to-mgp --mogg-outline-async-routines --mogg-add-try-except-blocks --mogg-emit-mojo --mogg-dump-mojo -o /dev/null 2>&1 | FileCheck %s --check-prefix=EMIT

// A trait kernel whose input is `Some[Input & DenseTensor & Stable]`
// (`trait_stable_negate`). The contract carries `stable = true` on the
// `mogg.trait.create` that wraps the input, and emit assembles a
// `_MoggStableTensorArg` (slot-backed, `StableAddrEngine`) instead of the
// pointer-backed `_MoggDenseTensorArg`.

// CHECK-LABEL: mogg.func @__custom__trait_stable_negate
// The output stays a plain dense arg; the input wrapper carries the stable
// contract.
// CHECK-DAG: mogg.trait.create{{.*}}contract = <{role = output, fused = false, access = none, capabilities = none}>
// CHECK-DAG: mogg.trait.create{{.*}}contract = <{role = input, fused = false, access = none, capabilities = none, stable = true}>

// The stable wrapper is defined once, after the dense defs it depends on for
// `_MoggTensorIO`, with the slot engine and field-wise device encoding.
// EMIT-DAG: struct _MoggTensorIO
// EMIT-DAG: struct _MoggStableTensorArg[
// EMIT-DAG: comptime Engine = StableAddrEngine[element_width=1]
// EMIT-DAG: self._addr = StableAddr(
// EMIT-DAG: encoder.encode_fields[Self](self, target)
//
// The stub assembles the stable input and the dense output from their buffer
// captures, then calls the trait kernel.
// EMIT-DAG: _MoggStableTensorArg[type_of(var___{{[0-9]+}}).dtype, type_of(var___{{[0-9]+}}), {{[0-9]+}}, _MoggTensorIO.INPUT]
// EMIT-DAG: _MoggDenseTensorArg[type_of(var___{{[0-9]+}}).dtype, type_of(var___{{[0-9]+}}), {{[0-9]+}}, _MoggTensorIO.OUTPUT
// EMIT-DAG: TraitStableNegate.execute[target = "gpu"]

mo.graph @trait_stable_negate_graph(%x: !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0> {
  %z = mo.custom [symbol = "trait_stable_negate", device = #M.device_ref<"gpu", 0>] (%x) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %z : !mo.tensor<[4], f32, gpu:0>
}
