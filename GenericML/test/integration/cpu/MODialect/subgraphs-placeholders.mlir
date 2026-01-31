// RUN: mo-opt %s --mo-to-mgp=constant-fold=false -o %t.mlir

mo.graph @add(%arg0: !mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>) attributes {isSubgraph} {
  %1 = mo.constant.external {align = 16 : ui64, device = #M.device_ref<"cpu", 0>, name = "foo", isPlaceholder = true } : !mo.tensor<[5], f32>
  %add = mo.add(%arg0, %1) : !mo.tensor<[5], f32>
  mo.output %add : !mo.tensor<[5], f32>
}

mo.graph @combined() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<-1.0, 2.0, -3.0, -4.0, 5.0> : tensor<5xf32>} : !mo.tensor<[5], f32>
  %add = mo.call @add(%0) {prefix = "prefix"} : (!mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
  mo.output %add : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'combined' returned tensor<5xf32> [2.000000e+00, 3.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+01]
