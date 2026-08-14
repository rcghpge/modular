// RUN: not mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_pkg" --mo-to-mogg --mogg-fuse-elementwise --mogg-fuse-epilogues --mogg-simplify-kernel-operands --mogg-to-mgp 2>&1 | FileCheck %s

// Epilogue fusion captures the producer's legacy input in a lambda attached to
// its trait-bound output. The same value remains an explicit input, so
// derivation must reject the incompatible representations.

// CHECK: error: kernel I/O value derives conflicting tensor-arg contracts: call operand #{{[0-9]+}} requires #mogg.tensor_arg
// CHECK: note: call operand #{{[0-9]+}} first required legacy `ManagedTensorSlice`
mo.graph @derive_capture_conflict(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %0 = mo.custom {symbol = "trait_fused_store_legacy_input", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %1 = mo.add(%0, %x) : !mo.tensor<[4], f32>
  mo.output %1 : !mo.tensor<[4], f32>
}
