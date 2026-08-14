// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_pkg mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-mogg --mogg-fuse-elementwise --mogg-fuse-prologues --mogg-simplify-kernel-operands --mogg-to-mgp 2>&1 | FileCheck %s

// Trait contracts are derived from the kernel declaration at boundary
// lowering, keyed by execute-call operand position (GEX-4071). These cases
// mutate the kernel operand list AFTER mo-to-mogg -- exactly the transforms
// that silently corrupted the deleted positional side-table -- and check the
// derived `mogg.tensor_arg.__init__` routing survives them. No contract array
// is stamped on `mogg.kernel`.

// `SimplifyKernelOperands` merges the duplicate operand into one exec-body
// argument feeding two execute positions; both positions declare the same
// dense input contract, so the merged argument derives it unambiguously and
// exactly one input init op (plus the output's) is built.
// CHECK-LABEL: mgp.model @derive_dup_operand
// CHECK-COUNT-2: mogg.tensor_arg.__init__
// CHECK-NOT: mogg.tensor_arg.__init__
// CHECK-NOT: mogg.input_tensor_arg_contracts
mo.graph @derive_dup_operand(%x: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mo.custom {symbol = "trait_add_explicit", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x, %x) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// Prologue fusion of a two-operand producer: the fused input's `mo.add`
// producer leaves TWO boundary capture arguments for ONE declared fused
// operand. Each capture reaches the execute call only through the bind
// lambda, so each derives the fused operand's contract.
// CHECK-LABEL: mgp.model @derive_fused_two_operand_producer
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = input, fused = true, access = load, capabilities = load}>
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = input, fused = true, access = load, capabilities = load}>
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = output, fused = false, access = none, capabilities = none}>
// CHECK-NOT: mogg.input_tensor_arg_contracts
mo.graph @derive_fused_two_operand_producer(%a: !mo.tensor<[2, 3], f32>, %b: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %p = mo.add(%a, %b) : !mo.tensor<[2, 3], f32>
  %0 = mo.custom {symbol = "trait_fused_load", target = "cpu", device = #M.device_ref<"cpu", 0>}(%p) : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}
