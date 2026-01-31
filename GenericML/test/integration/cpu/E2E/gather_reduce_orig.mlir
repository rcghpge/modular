// RUN: mo-opt %s --mo-shape-inference --mo-to-mgp=constant-fold=false -o %t.mlir
// RUN: python3 %mo_vs_numpy_py -n GatherReduceConstAxis -f %t.mlir | FileCheck %s

// CHECK: All MO values match numpy

// This is the original gather reduce pattern.

mo.graph @serving_default<D0, D1, D2, D3 -> D4, D5, D6>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D2, D3], si32>) -> !mo.tensor<[D4, D5, D6], f32> attributes {argument_names = ["input_1", "input_2"], result_names = ["lambda"]} {
  %0 = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %1 = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %2 = mo.gather<() -> D7, D8, D9>(%arg0, %arg1, %1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[D7, D8, D9], f32>
  %3 = mo.constant {value = #M.dense_array<> : tensor<0xsi64>} : !mo.tensor<[0], si64>
  %4 = mo.reshape(%0, %3) : (!mo.tensor<[1], si64>, !mo.tensor<[0], si64>) -> !mo.tensor<[], si64>
  %5 = mo.reduce.add<() -> D10, D11, D12>(%2, %4) : (!mo.tensor<[D7, D8, D9], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[D10, D11, D12], f32>
  mo.output<D10, D11, D12> %5 : !mo.tensor<[D10, D11, D12], f32>
}
