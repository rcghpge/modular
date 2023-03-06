// RUN: mo-opt %s --split-input-file --verify-diagnostics

//===----------------------------------------------------------------------===//
// Matrix Band
//===----------------------------------------------------------------------===//

mo.graph @matrix_band_part() -> !mo.tensor<[1, 2, 3], f32> {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<1x2x3xf32>} : !mo.tensor<[1, 2, 3], f32>
  %1 = mo.constant {value = #M.dense_array<-1> : tensor<si32>} : !mo.tensor<[], si32>
  %2 = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  // expected-error@+1 {{first operand and the result must have the same type}}
  %3 = mo.matrix_band_part(%0, %1, %2) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[2, 3], f32>
  mo.output %3 : !mo.tensor<[2, 3], f32>
}

// -----

mo.graph @matrix_band_part() -> !mo.tensor<[1, 2, 3], f32> {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<1x2x3xf32>} : !mo.tensor<[1, 2, 3], f32>
  %1 = mo.constant {value = #M.dense_array<-1.0> : tensor<f32>} : !mo.tensor<[], f32>
  %2 = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  // expected-error@+1 {{'mo.matrix_band_part' op operand #1 must be ranked scalar index tensor, but got '!mo.tensor<[], f32>'}}
  %3 = mo.matrix_band_part(%0, %1, %2) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[1, 2, 3], f32>
  mo.output %3 : !mo.tensor<[1, 2, 3], f32>
}

// -----

mo.graph @matrix_band_part() -> !mo.tensor<[1, 2, 3], f32> {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<1x2x3xf32>} : !mo.tensor<[1, 2, 3], f32>
  %1 = mo.constant {value = #M.dense_array<-1> : tensor<si32>} : !mo.tensor<[], si32>
  %2 = mo.constant {value = #M.dense_array<0.0> : tensor<f32>} : !mo.tensor<[], f32>
  // expected-error@+1 {{'mo.matrix_band_part' op operand #2 must be ranked scalar index tensor, but got '!mo.tensor<[], f32>'}}
  %3 = mo.matrix_band_part(%0, %1, %2) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[], si32>, !mo.tensor<[], f32>) -> !mo.tensor<[1, 2, 3], f32>
  mo.output %3 : !mo.tensor<[1, 2, 3], f32>
}

