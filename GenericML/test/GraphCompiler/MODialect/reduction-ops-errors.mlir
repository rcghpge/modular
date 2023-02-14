// RUN: mo-opt %s --split-input-file --verify-diagnostics

//===----------------------------------------------------------------------===//
// Mean
//===----------------------------------------------------------------------===//

mo.graph @mean_non_scalar_axis(%input: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 1], f32>) {
  %axis = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2, 1], si64>
  // expected-error@+1 {{'mo.mean' op operand #1 must be ranked scalar tensor, but got '!mo.tensor<[2, 1], si64>'}}
  %output = mo.mean(%input, %axis) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 1], si64>) -> !mo.tensor<[2, 1], f32>
  mo.output %output : !mo.tensor<[2, 1], f32>
}

// -----
