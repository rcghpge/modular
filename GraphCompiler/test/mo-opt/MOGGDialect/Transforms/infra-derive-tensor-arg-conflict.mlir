// RUN: not mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_pkg" --mo-to-mogg --mogg-simplify-kernel-operands --mogg-to-mgp 2>&1 | FileCheck %s

// A mixed legacy+trait kernel called with the same value twice:
// `SimplifyKernelOperands` merges the duplicate operands, leaving one
// exec-body argument that feeds a legacy (contract-free) execute position AND
// a trait fused position. One kernel I/O value cannot carry two different
// contracts, so derivation reports the conflict instead of silently applying
// either representation (GEX-4071; the deleted positional side-table crashed
// on an alignment assert here at best, and mislabeled the operand at worst).
// TODO(GEX-4071): lowering one init op per distinct contract from the shared
// buffer (or refusing the dedup) would make this case compile.

// CHECK: error: kernel I/O value derives conflicting tensor-arg contracts: call operand #{{[0-9]+}} requires
// CHECK: note: call operand #{{[0-9]+}} first required
mo.graph @derive_conflicting_dup_operand(%x: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mo.custom {symbol = "trait_mixed_legacy_fused", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x, %x) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}
