// RUN: mo-opt %s -o %t.mlir --mo-to-mgp="kernel-library-paths=%kernel_api_test_lib" --allow-unregistered-dialect --mlir-print-ir-after="mogg-fuse-prologues-epilogues" --mlir-disable-threading 2>&1 | FileCheck %s --check-prefix=CHECK-IR


// CHECK-IR-LABEL: mo.graph @print_shape
// CHECK-IR-NEXT: %[[OUT:.*]] = mogg.kernel({{.+}})
// CHECK-IR: mo.output %[[OUT]]

// CHECK-LABEL: Running 'print_shape':
// CHECK: input.shape = (1, 2, 3)
// CHECK: output.shape = (1, 2, 3)
mo.graph @print_shape<D0, D1>(%0: !mo.tensor<[D0, D1, 3], f32>) -> (!mo.tensor<[D0, D1, 3], f32>) {
  %1 = mo.custom {symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>} (%0, %0) : (!mo.tensor<[D0, D1, 3], f32>, !mo.tensor<[D0, D1, 3], f32>) -> !mo.tensor<[D0, D1, 3], f32>
  %2 = mo.custom {symbol = "print_shape_fused", device = #M.device_ref<"cpu", 0>} (%1) : (!mo.tensor<[D0, D1, 3], f32>) -> !mo.tensor<[D0, D1, 3], f32>
  mo.output %2: !mo.tensor<[D0, D1, 3], f32>
}
