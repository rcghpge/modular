// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK: mo.graph @abs(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> {
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

// CHECK: mo.graph @relu(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> {
mo.graph @relu(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.relu(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.relu(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @gelu(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> {
mo.graph @gelu(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.gelu(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.gelu(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @erf(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> {
mo.graph @erf(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.erf(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.erf(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @tanh(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> {
mo.graph @tanh(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.tanh(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.tanh(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @exp(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> {
mo.graph @exp(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.exp(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.exp(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @sigmoid(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> {
mo.graph @sigmoid(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.sigmoid(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.sigmoid(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @rsqrt(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> {
mo.graph @rsqrt(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.rsqrt(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.rsqrt(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @sqrt(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> {
mo.graph @sqrt(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: mo.sqrt(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.sqrt(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @cast(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], si32> {
mo.graph @cast(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], si32>) {
  // CHECK-NEXT: mo.cast(%[[ARG0]]) : (!mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], si32>
  %0 = mo.cast(%arg0) : (!mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], si32>
  mo.output %0 : !mo.tensor<[4, 5], si32>
}

// CHECK: mo.graph @cast_folder()
mo.graph @cast_folder() -> (!mo.tensor<[2], si32>, !mo.tensor<[3], ui64>, !mo.tensor<[2, 2], ui32>, !mo.tensor<[5], si64>) {
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
