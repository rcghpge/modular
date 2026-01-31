// TODO(GEX-2314): Support opaque elementwise in the new dialect.
// REQUIRES: disabled

// RUN: mo-opt %s -o %t.mlir --mo-to-mgp="kernel-library-paths=%kernel_api_test_lib" --mlir-print-ir-after="fuse-elementwise" 2>&1 | FileCheck --check-prefix=CHECK-IR %s

// CHECK-IR-LABEL: @elementwise_opaque_fusion
// CHECK-IR: mogg.kernel
// CHECK-IR: mogg.kernel
// CHECK-IR: summary = "mo.add:{{[0-9]+}}+custom__fused_write_to_tensor_wrapper:{{[0-9]+}}"
// CHECK-IR: mogg.kernel


!tt1 = !mo.tensor<[5, 5], f32>
!tt2 = !mo.tensor<[3, 3], f32>
!ot = !mo.opaque<"TensorWrapper">
!ct = !mo.chain

mo.graph @elementwise_opaque_fusion(%input: !tt1, %add_input1: !tt2, %add_input2: !tt2) {

  %ch0 = mo.chain.create()

  %result = mo.custom {symbol = "to_tensor_wrapper", device = #M.device_ref<"cpu", 0>} (%input) : (!tt1) -> !ot

  %add_result = mo.add(%add_input1, %add_input2) : !tt2
  %ch1 = mo.custom {symbol = "fused_write_to_tensor_wrapper", device = #M.device_ref<"cpu", 0>} (%result, %add_result, %ch0) : (!ot, !tt2, !ct) -> !ct

  %ch2 = mo.custom {symbol = "print_tensor_wrapper", device = #M.device_ref<"cpu", 0>} (%result, %ch1) : (!ot, !ct) -> !ct

  mo.output
}
