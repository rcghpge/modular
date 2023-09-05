// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK: mo.graph @abs(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> no_inline {
mo.graph @abs(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.abs(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.abs(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @add_folder()
mo.graph @add_folder() -> (!mo.tensor<[2, 2], si32>) {
  // CHECK: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<-1, 2, 0, 4> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>
  %0 = mo.constant {value = #M.dense_array<-1, 0, 1, 2> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>
  %1 = mo.constant {value = #M.dense_array<0, 2, -1, 2> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>
  %2 = mo.add(%0, %1) : !mo.tensor<[2, 2], si32>
  // CHECK-NEXT: mo.output %[[VAL0]] : !mo.tensor<[2, 2], si32>
  mo.output %2 : !mo.tensor<[2, 2], si32>
}

// CHECK: mo.graph @relu(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> no_inline {
mo.graph @relu(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.relu(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.relu(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @gelu(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> no_inline {
mo.graph @gelu(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.gelu(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.gelu(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @erf(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> no_inline {
mo.graph @erf(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.erf(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.erf(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @tanh(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> no_inline {
mo.graph @tanh(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.tanh(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.tanh(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @exp(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> no_inline {
mo.graph @exp(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.exp(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.exp(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @sigmoid(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> no_inline {
mo.graph @sigmoid(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.sigmoid(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.sigmoid(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @rsqrt(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> no_inline {
mo.graph @rsqrt(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.rsqrt(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.rsqrt(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @sqrt(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> no_inline {
mo.graph @sqrt(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.sqrt(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.sqrt(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @cast(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], si32> no_inline {
mo.graph @cast(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], si32>) {
  // CHECK-NEXT: mo.cast(%[[ARG0]]) : (!mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], si32>
  %0 = mo.cast(%arg0) : (!mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], si32>
  mo.output %0 : !mo.tensor<[4, 5], si32>
}

// CHECK: mo.graph @cast_int_folder()
mo.graph @cast_int_folder() -> (!mo.tensor<[2], si32>, !mo.tensor<[3], ui64>, !mo.tensor<[2, 2], ui32>, !mo.tensor<[5], si64>) {
  // CHECK-DAG: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  %0 = mo.constant {value = #M.dense_array<1, 2> : tensor<2xui64>} : !mo.tensor<[2], ui64>
  %1 = mo.cast(%0) : (!mo.tensor<[2], ui64>) -> !mo.tensor<[2], si32>

  // CHECK-DAG: %[[VAL1:.*]] = mo.constant {value = #M.dense_array<18446744073709551615, 0, 1> : tensor<3xui64>} : !mo.tensor<[3], ui64>
  %2 = mo.constant {value = #M.dense_array<-1, 0, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %3 = mo.cast(%2) : (!mo.tensor<[3], si64>) -> !mo.tensor<[3], ui64>

  // CHECK-DAG: %[[VAL2:.*]] = mo.constant {value = #M.dense_array<4294967295, 0, 1, 2> : tensor<2x2xui32>} : !mo.tensor<[2, 2], ui32>
  %4 = mo.constant {value = #M.dense_array<-1, 0, 1, 2> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>
  %5 = mo.cast(%4) : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2], ui32>

  // CHECK-DAG: %[[VAL3:.*]] = mo.constant {value = #M.dense_array<-2, -1, 0, 1, 2> : tensor<5xsi64>} : !mo.tensor<[5], si64>
  %6 = mo.constant {value = #M.dense_array<-2, -1, 0, 1, 2> : tensor<5xsi32>} : !mo.tensor<[5], si32>
  %7 = mo.cast(%6) : (!mo.tensor<[5], si32>) -> !mo.tensor<[5], si64>

  // CHECK-NEXT: %[[VAL0]], %[[VAL1]], %[[VAL2]], %[[VAL3]] : !mo.tensor<[2], si32>, !mo.tensor<[3], ui64>, !mo.tensor<[2, 2], ui32>, !mo.tensor<[5], si64>
  mo.output %1, %3, %5, %7 : !mo.tensor<[2], si32>, !mo.tensor<[3], ui64>, !mo.tensor<[2, 2], ui32>, !mo.tensor<[5], si64>
}

// CHECK: mo.graph @cast_int_float_mixed_folder()
mo.graph @cast_int_float_mixed_folder() -> (!mo.tensor<[2], f32>, !mo.tensor<[3], f64>) {
  // CHECK-DAG: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<1, 2> : tensor<2xui64>} : !mo.tensor<[2], ui64>
  // CHECK-DAG: %[[VAL1:.*]] = mo.cast(%[[VAL0]]) : (!mo.tensor<[2], ui64>) -> !mo.tensor<[2], f32>
  %0 = mo.constant {value = #M.dense_array<1, 2> : tensor<2xui64>} : !mo.tensor<[2], ui64>
  %1 = mo.cast(%0) : (!mo.tensor<[2], ui64>) -> !mo.tensor<[2], f32>

  // CHECK-DAG: %[[VAL2:.*]] = mo.constant {value = #M.dense_array<-1, 0, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  // CHECK-DAG: %[[VAL3:.*]] = mo.cast(%[[VAL2]]) : (!mo.tensor<[3], si64>) -> !mo.tensor<[3], f64>
  %2 = mo.constant {value = #M.dense_array<-1, 0, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %3 = mo.cast(%2) : (!mo.tensor<[3], si64>) -> !mo.tensor<[3], f64>

  // CHECK-NEXT: %[[VAL1]], %[[VAL3]] : !mo.tensor<[2], f32>, !mo.tensor<[3], f64>
  mo.output %1, %3 : !mo.tensor<[2], f32>, !mo.tensor<[3], f64>
}

// Check casting between signed and unsigned types to see if sign extension works.
// CHECK: mo.graph @cast_int_between_signs() -> (!mo.tensor<[4], si32>, !mo.tensor<[4], ui32>, !mo.tensor<[7], si32>, !mo.tensor<[7], ui32>) no_inline {
mo.graph @cast_int_between_signs() -> (!mo.tensor<[4], si32>, !mo.tensor<[4], ui32>, !mo.tensor<[7], si32>, !mo.tensor<[7], ui32>) {
  // Each of these is the maximum and minimum value for the dtype, and a few other numbers
  %ui8 = mo.constant {value = #M.dense_array<255, 100, 1, 0> : tensor<4xui8>} : !mo.tensor<[4], ui8>
  %si8 = mo.constant {value = #M.dense_array<127, 100, 1, 0, -128, -100, -1> : tensor<7xsi8>} : !mo.tensor<[7], si8>

  // CHECK-DAG: %[[UI8_TO_SI32:.*]] = mo.constant {value = #M.dense_array<255, 100, 1, 0> : tensor<4xsi32>} : !mo.tensor<[4], si32>
  %ui8_to_si32 = mo.cast(%ui8): (!mo.tensor<[4], ui8>) -> !mo.tensor<[4], si32>

  // CHECK-DAG: %[[UI8_TO_UI32:.*]] = mo.constant {value = #M.dense_array<255, 100, 1, 0> : tensor<4xui32>} : !mo.tensor<[4], ui32>
  %ui8_to_ui32 = mo.cast(%ui8): (!mo.tensor<[4], ui8>) -> !mo.tensor<[4], ui32>

  // CHECK-DAG: %[[SI8_TO_SI32:.*]] = mo.constant {value = #M.dense_array<127, 100, 1, 0, -128, -100, -1> : tensor<7xsi32>} : !mo.tensor<[7], si32>
  %si8_to_si32 = mo.cast(%si8): (!mo.tensor<[7], si8>) -> !mo.tensor<[7], si32>

  // Note casting negative elements to positive is undefined behavior so we don't check it
  // CHECK-DAG: %[[SI8_TO_UI32:.*]] = mo.constant {value = #M.dense_array<127, 100, 1, 0
  %si8_to_ui32 = mo.cast(%si8): (!mo.tensor<[7], si8>) -> !mo.tensor<[7], ui32>

  // CHECK: mo.output %[[UI8_TO_SI32]], %[[UI8_TO_UI32]], %[[SI8_TO_SI32]], %[[SI8_TO_UI32]]
  mo.output %ui8_to_si32, %ui8_to_ui32, %si8_to_si32, %si8_to_ui32: !mo.tensor<[4], si32>, !mo.tensor<[4], ui32>, !mo.tensor<[7], si32>, !mo.tensor<[7], ui32>
}

// CHECK: mo.graph @log(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> no_inline {
mo.graph @log(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.log(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.log(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @log1p(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> no_inline {
mo.graph @log1p(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.log1p(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.log1p(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @floor(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> no_inline {
mo.graph @floor(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.floor(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.floor(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @ceil(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> no_inline {
mo.graph @ceil(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.ceil(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.ceil(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @trunc(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
mo.graph @trunc(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.trunc(%[[ARG0]]) : !mo.tensor<[2, 3], f32>
  %0 = mo.trunc(%arg0) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @round(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
mo.graph @round(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.round(%[[ARG0]]) : !mo.tensor<[2, 3], f32>
  %0 = mo.round(%arg0) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @roundeven(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
mo.graph @roundeven(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.roundeven(%[[ARG0]]) : !mo.tensor<[2, 3], f32>
  %0 = mo.roundeven(%arg0) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @isnan(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], bool> no_inline {
mo.graph @isnan(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], bool>) {
  // CHECK-NEXT: mo.is_nan(%[[ARG0]]) :  (!mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], bool>
  %0 = mo.is_nan(%arg0) : (!mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], bool>
  mo.output %0 : !mo.tensor<[4, 5], bool>
}

// CHECK: mo.graph @isinf(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], bool> no_inline {
mo.graph @isinf(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], bool>) {
  // CHECK-NEXT: mo.is_inf(%[[ARG0]]) :  (!mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], bool>
  %0 = mo.is_inf(%arg0) : (!mo.tensor<[4, 5], f32>) ->  !mo.tensor<[4, 5], bool>
  mo.output %0 : !mo.tensor<[4, 5], bool>
}

// CHECK: mo.graph @not(%[[ARG0:.*]]: !mo.tensor<[4, 5], bool>) -> !mo.tensor<[4, 5], bool> no_inline {
mo.graph @not(%arg0: !mo.tensor<[4, 5], bool>) -> (!mo.tensor<[4, 5], bool>) {
  // CHECK-NEXT: mo.not(%[[ARG0]]) : !mo.tensor<[4, 5], bool>
  %0 = mo.not(%arg0) : !mo.tensor<[4, 5], bool>
  mo.output %0 : !mo.tensor<[4, 5], bool>
}
