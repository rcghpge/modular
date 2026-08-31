// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-mogg --mogg-simplify-kernel-operands --mogg-to-mgp 2>&1 | FileCheck %s

// A mixed legacy+trait kernel called with the same value twice:
// `SimplifyKernelOperands` merges the duplicate operands, leaving one
// exec-body argument that feeds a legacy (contract-free) execute position AND
// a trait dense position. The lowering materializes one init per derived
// contract lane over the shared raw buffer, instead of rejecting the mix: the
// same buffer backs a `mogg.tensor_arg.__init__` for the trait position and
// a `mogg.tensor.__init__` for the legacy one, both inside the stub. (The
// fused-lane mix keeps the conflict diagnostic — see
// infra-derive-tensor-arg-conflict.mlir.)
//
// Related to GEX-4071.

// CHECK-LABEL: mogg.kernel.stub @custom__trait_dual_lane_add
// CHECK-DAG: mogg.tensor_arg.__init__(%[[BUF:arg[0-9]+]]) {contract = #mogg.tensor_arg<{role = input, fused = false, access = none, capabilities = none}>
// CHECK-DAG: mogg.tensor.__init__(%[[BUF]])
// CHECK-NOT: error: kernel I/O value derives conflicting tensor-arg contracts
mo.graph @lower_per_use_dual_lane(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %0 = mo.custom {symbol = "trait_dual_lane_add", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x, %x) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}
