// RUN: mo-opt %s --mo-to-mgp=constant-fold=false | FileCheck %s --check-prefix=CHECK-IR
// RUN: mt --execute --result-output-style=full %s | FileCheck %s --check-prefix=CHECK-MT


mo.graph @rms_norm() -> !mo.tensor<[3, 2], f32> {
  %input = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<3x2xf32>} : !mo.tensor<[3, 2], f32>
  %gamma = mo.constant {value = #M.dense_array<1.1, 2.1> : tensor<2xf32>} : !mo.tensor<[2], f32>
  %eps   = mo.constant {value = #M.dense_array<0.00001> : tensor<f32>} : !mo.tensor<[], f32>
  %weight_offset = mo.constant {value = #M.dense_array<0.0> : tensor<f32>} : !mo.tensor<[], f32>
  %rnorm = mo.custom {symbol = "rms_norm", device = #M.device_ref<"cpu", 0>}(%input, %gamma, %eps, %weight_offset) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[3, 2], f32>
  %res   = mo.add(%rnorm, %input) : !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}

// CHECK-MT: 'rms_norm' returned tensor<3x2xf32> [1.6957{{[0-9]*}}e+00, 4.6563{{[0-9]*}}e+00, 3.9333{{[0-9]*}}e+00, 6.3758{{[0-9]*}}e+00, 5.9958{{[0-9]*}}e+00, 8.2815{{[0-9]*}}e+00]
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}custom__rms_norm{{.*}}mo.add{{.*}}
