// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-mogg --mogg-fuse-elementwise --mogg-fuse-epilogues --mogg-simplify-kernel-operands --mogg-to-mgp 2>&1 | FileCheck %s

// Epilogue fusion inserts EXTRA consumer operands on the producer that reach
// the kernel call terminator only as `mogg.tensor.load`s inside the output
// bind lambda. Such a capture inherits the fused output operand's contract
// (it is emitted by that operand's fusion machinery, so it must be on the
// trait lane), but only weakly: a direct execute-operand use wins (GEX-4071).

// The extra `%y` never reaches the execute call directly, so it inherits the
// fused-store output's contract; the producer input `%x` keeps its declared
// dense input contract. Both the real output (`mut = true`) and the inherited
// extra (`mut = false`) carry the fused-store contract.
// CHECK-LABEL: mgp.model @derive_epilogue_extra
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}role = output, fused = true, access = store{{.*}}mut = false
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}role = input, fused = false, access = none{{.*}}mut = false
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}role = output, fused = true, access = store{{.*}}mut = true
mo.graph @derive_epilogue_extra(%x: !mo.tensor<[4], f32>, %y: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %0 = mo.custom {symbol = "trait_fused_store_double", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %1 = mo.add(%0, %y) : !mo.tensor<[4], f32>
  mo.output %1 : !mo.tensor<[4], f32>
}

// `SimplifyKernelOperands` merges the epilogue extra with the producer's own
// input: one exec-body argument feeds the dense input execute position AND
// the in-lambda load. The direct operand position determines the contract
// (dense input); the weak lambda-capture vote only cross-checks the lane.
// This used to hard-error with a contract conflict.
// CHECK-LABEL: mgp.model @derive_epilogue_extra_dedup
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}role = input, fused = false, access = none
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}role = output, fused = true, access = store
mo.graph @derive_epilogue_extra_dedup(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %0 = mo.custom {symbol = "trait_fused_store_double", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %1 = mo.add(%0, %x) : !mo.tensor<[4], f32>
  mo.output %1 : !mo.tensor<[4], f32>
}
