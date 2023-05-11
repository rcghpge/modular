// RUN: mo-opt %s --split-input-file --verify-diagnostics

//===----------------------------------------------------------------------===//
// Matrix Band
//===----------------------------------------------------------------------===//

mo.graph @linalg_band_part() -> !mo.tensor<[1, 2, 3], f32> {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<1x2x3xf32>} : !mo.tensor<[1, 2, 3], f32>
  %1 = mo.constant {value = #M.dense_array<-1> : tensor<si32>} : !mo.tensor<[], si32>
  %2 = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %3 = mo.constant {value = #M.dense_array<0> : tensor<ui8>} : !mo.tensor<[], bool>
  // expected-error@+1 {{first operand and the result must have the same type}}
  %4 = mo.linalg.band_part(%0, %1, %2, %3) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[2, 3], f32>
  mo.output %4 : !mo.tensor<[2, 3], f32>
}

// -----

mo.graph @linalg_band_part() -> !mo.tensor<[1, 2, 3], f32> {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<1x2x3xf32>} : !mo.tensor<[1, 2, 3], f32>
  %1 = mo.constant {value = #M.dense_array<-1.0> : tensor<f32>} : !mo.tensor<[], f32>
  %2 = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %3 = mo.constant {value = #M.dense_array<0> : tensor<ui8>} : !mo.tensor<[], bool>
  // expected-error@+1 {{'mo.linalg.band_part' op operand #1 must be ranked scalar index tensor, but got '!mo.tensor<[], f32>'}}
  %4 = mo.linalg.band_part(%0, %1, %2, %3) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[], f32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[1, 2, 3], f32>
  mo.output %4 : !mo.tensor<[1, 2, 3], f32>
}

// -----

mo.graph @linalg_band_part() -> !mo.tensor<[1, 2, 3], f32> {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<1x2x3xf32>} : !mo.tensor<[1, 2, 3], f32>
  %1 = mo.constant {value = #M.dense_array<-1> : tensor<si32>} : !mo.tensor<[], si32>
  %2 = mo.constant {value = #M.dense_array<0.0> : tensor<f32>} : !mo.tensor<[], f32>
  %3 = mo.constant {value = #M.dense_array<0> : tensor<ui8>} : !mo.tensor<[], bool>
  // expected-error@+1 {{'mo.linalg.band_part' op operand #2 must be ranked scalar index tensor, but got '!mo.tensor<[], f32>'}}
  %4 = mo.linalg.band_part(%0, %1, %2, %3) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[], si32>, !mo.tensor<[], f32>, !mo.tensor<[], bool>) -> !mo.tensor<[1, 2, 3], f32>
  mo.output %4 : !mo.tensor<[1, 2, 3], f32>
}

// -----

mo.graph @linalg_band_part() -> !mo.tensor<[1, 2, 3], f32> {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<1x2x3xf32>} : !mo.tensor<[1, 2, 3], f32>
  %1 = mo.constant {value = #M.dense_array<-1> : tensor<si32>} : !mo.tensor<[], si32>
  %2 = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %3 = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  // expected-error@+1 {{'mo.linalg.band_part' op operand #3 must be rank 0 and boolean element type, but got '!mo.tensor<[], si32>'}}
  %4 = mo.linalg.band_part(%0, %1, %2, %3) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[1, 2, 3], f32>
  mo.output %4 : !mo.tensor<[1, 2, 3], f32>
}

// -----

mo.graph @linalg_band_part() -> !mo.tensor<[1, 2, 3], f32> {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<1x2x3xf32>} : !mo.tensor<[1, 2, 3], f32>
  %1 = mo.constant {value = #M.dense_array<-1> : tensor<si32>} : !mo.tensor<[], si32>
  %2 = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %3 = mo.constant {value = #M.dense_array<0> : tensor<1xui8>} : !mo.tensor<[1], ui8>
  // expected-error@+1 {{'mo.linalg.band_part' op operand #3 must be rank 0 and boolean element type, but got '!mo.tensor<[1], ui8>'}}
  %4 = mo.linalg.band_part(%0, %1, %2, %3) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[1], ui8>) -> !mo.tensor<[1, 2, 3], f32>
  mo.output %4 : !mo.tensor<[1, 2, 3], f32>
}

// -----

//===----------------------------------------------------------------------===//
// Solve
//===----------------------------------------------------------------------===//

mo.graph @linalg_solve() -> !mo.tensor<[1, 2, 2], f32> {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<1x2x2xf32>} : !mo.tensor<[1, 2, 2], f32>
  %1 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<1x2x3xf32>} : !mo.tensor<[1, 2, 3], f32>
  // expected-error@+1 {{'mo.linalg.solve' op incompatible input/result matmul shapes}}
  %4 = mo.linalg.solve(%0, %1) : (!mo.tensor<[1, 2, 2], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 2], f32>
  mo.output %4 : !mo.tensor<[1, 2, 2], f32>
}

// -----

mo.graph @linalg_solve_mismatched_batches() -> !mo.tensor<[2, 2, 3], f32> {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<1x2x2xf32>} : !mo.tensor<[1, 2, 2], f32>
  %1 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<2x2x3xf32>} : !mo.tensor<[2, 2, 3], f32>
  // expected-error@+1 {{mo.linalg.solve' op incompatible input/result batch dimensions}}
  %4 = mo.linalg.solve(%0, %1) : (!mo.tensor<[1, 2, 2], f32>, !mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 2, 3], f32>
  mo.output %4 : !mo.tensor<[2, 2, 3], f32>
}

// -----

mo.graph @linalg_solve_non_square() -> !mo.tensor<[1, 3, 3], f32> {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<1x2x3xf32>} : !mo.tensor<[1, 3, 2], f32>
  %1 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<1x2x3xf32>} : !mo.tensor<[1, 2, 3], f32>
  // expected-error@+1 {{first operand must be a batch of square matrices}}
  %4 = mo.linalg.solve(%0, %1) : (!mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 3, 3], f32>
  mo.output %4 : !mo.tensor<[1, 3, 3], f32>
}

// -----

mo.graph @linalg_solve_wrong_dtype() -> !mo.tensor<[1, 2, 3], f64> {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<1x2x2xf32>} : !mo.tensor<[1, 2, 2], f32>
  %1 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<1x2x3xf32>} : !mo.tensor<[1, 2, 3], f64>
  // expected-error@+1 {{'mo.linalg.solve' op requires the same element type for all operands and results}}
  %4 = mo.linalg.solve(%0, %1) : (!mo.tensor<[1, 2, 2], f32>, !mo.tensor<[1, 2, 3], f64>) -> !mo.tensor<[1, 2, 3], f64>
  mo.output %4 : !mo.tensor<[1, 2, 3], f64>
}

// -----

mo.graph @linalg_solve_mismatched_ranks() -> !mo.tensor<[2, 3], f32> {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<1x2x2xf32>} : !mo.tensor<[1, 2, 2], f32>
  %1 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<1x2x3xf32>} : !mo.tensor<[2, 3], f32>
  // expected-error@+1 {{input and output ranks must be all dynamic, or all static and match}}
  %4 = mo.linalg.solve(%0, %1) : (!mo.tensor<[1, 2, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %4 : !mo.tensor<[2, 3], f32>
}

// -----

