// RUN: mo-opt %s --mo-to-mogg="kernel-library-paths=%extra_test_lib" | FileCheck %s

// Ensures we support scalars. Notice there is no fusion lambda for the `mogg.call`
// CHECK-LABEL: mo.graph @supports_scalars
// CHECK: mogg.call "supports_scalar_kernel" [](%arg1, %1) : !mo.tensor<[1], f32>, !mo.tensor<[1], f32>
mo.graph @supports_scalars(%a : !mo.tensor<[1], f32>) -> !mo.tensor<[1], f32> {
  %0 = mo.custom {symbol = "supports_scalar_kernel"}(%a) : (!mo.tensor<[1], f32>) -> !mo.tensor<[1], f32>
  mo.output %0 : !mo.tensor<[1], f32>
}
