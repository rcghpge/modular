// RUN: mo-opt -split-input-file --mo-symbolic-optimize %s | FileCheck %s

// Basic case: reshape expands a batch dimension, slice is on the last axis.
//
// BEFORE: [6, 8] -> slice -> [6, 4] -> reshape -> [2, 3, 4]
// AFTER:  [6, 8] -> reshape -> [2, 3, 8] -> slice -> [2, 3, 4]

// CHECK-LABEL: mo.graph @swim_reshape_through_slice_basic
// CHECK-DAG:   %[[STEP:.*]] = mo.constant {value = #M.dense_array<1, 1, 1>
// CHECK-DAG:   %[[STOP:.*]] = mo.constant {value = #M.dense_array<9223372036854775807, 9223372036854775807, 4>
// CHECK-DAG:   %[[START:.*]] = mo.constant {value = #M.dense_array<0, 0, 0>
// CHECK:       %[[RESHAPE:.*]] = mo.static.reshape(%arg0) : !mo.tensor<[6, 8], f32> -> !mo.tensor<[2, 3, 8], f32>
// CHECK:       %[[SLICE:.*]] = mo.slice(%[[RESHAPE]], %[[START]], %[[STOP]], %[[STEP]]) : (!mo.tensor<[2, 3, 8], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 3, 4], f32>
// CHECK:       mo.output %[[SLICE]]

mo.graph @swim_reshape_through_slice_basic(%arg0: !mo.tensor<[6, 8], f32>) -> !mo.tensor<[2, 3, 4], f32> {
  %start = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop = mo.constant {value = #M.dense_array<9223372036854775807, 4> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %slice = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[6, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[6, 4], f32>
  %reshape = mo.static.reshape(%slice) : !mo.tensor<[6, 4], f32> -> !mo.tensor<[2, 3, 4], f32>
  mo.output %reshape : !mo.tensor<[2, 3, 4], f32>
}

// -----

// No conversion: slice result is used by more than one user.

// CHECK-LABEL: mo.graph @no_conversion_multiple_users
// CHECK:       mo.slice
// CHECK:       mo.static.reshape
// CHECK:       mo.static.reshape

mo.graph @no_conversion_multiple_users(%arg0: !mo.tensor<[6, 8], f32>) -> (!mo.tensor<[2, 3, 4], f32>, !mo.tensor<[2, 3, 4], f32>) {
  %start = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop = mo.constant {value = #M.dense_array<9223372036854775807, 4> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %slice = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[6, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[6, 4], f32>
  %reshape0 = mo.static.reshape(%slice) : !mo.tensor<[6, 4], f32> -> !mo.tensor<[2, 3, 4], f32>
  %reshape1 = mo.static.reshape(%slice) : !mo.tensor<[6, 4], f32> -> !mo.tensor<[2, 3, 4], f32>
  mo.output %reshape0, %reshape1 : !mo.tensor<[2, 3, 4], f32>, !mo.tensor<[2, 3, 4], f32>
}

// -----

// No conversion: slice operates on two dimensions simultaneously.

// CHECK-LABEL: mo.graph @no_conversion_multidimensional_slice
// CHECK:       mo.slice
// CHECK:       mo.static.reshape

mo.graph @no_conversion_multidimensional_slice(%arg0: !mo.tensor<[6, 8], f32>) -> !mo.tensor<[3, 2, 2], f32> {
  %start = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop = mo.constant {value = #M.dense_array<3, 4> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %slice = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[6, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 4], f32>
  %reshape = mo.static.reshape(%slice) : !mo.tensor<[3, 4], f32> -> !mo.tensor<[3, 2, 2], f32>
  mo.output %reshape : !mo.tensor<[3, 2, 2], f32>
}

// -----

// No conversion: start is not a constant.

// CHECK-LABEL: mo.graph @no_conversion_dynamic_start
// CHECK:       mo.slice
// CHECK:       mo.static.reshape

mo.graph @no_conversion_dynamic_start(%arg0: !mo.tensor<[6, 8], f32>, %start: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3, 4], f32> {
  %stop = mo.constant {value = #M.dense_array<9223372036854775807, 4> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %slice = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[6, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[6, 4], f32>
  %reshape = mo.static.reshape(%slice) : !mo.tensor<[6, 4], f32> -> !mo.tensor<[2, 3, 4], f32>
  mo.output %reshape : !mo.tensor<[2, 3, 4], f32>
}

// -----

// Slice on first dimension, reshape expands last dimension.
//
// BEFORE: [6, 4] -> slice -> [3, 4] -> reshape -> [3, 2, 2]
// AFTER:  [6, 4] -> reshape -> [6, 2, 2] -> slice -> [3, 2, 2]

// CHECK-LABEL: mo.graph @swim_reshape_through_slice_first_dim
// CHECK-DAG:   %[[STEP:.*]] = mo.constant {value = #M.dense_array<1, 1, 1>
// CHECK-DAG:   %[[STOP:.*]] = mo.constant {value = #M.dense_array<4, 9223372036854775807, 9223372036854775807>
// CHECK-DAG:   %[[START:.*]] = mo.constant {value = #M.dense_array<1, 0, 0>
// CHECK:       %[[RESHAPE:.*]] = mo.static.reshape(%arg0) : !mo.tensor<[6, 4], f32> -> !mo.tensor<[6, 2, 2], f32>
// CHECK:       %[[SLICE:.*]] = mo.slice(%[[RESHAPE]], %[[START]], %[[STOP]], %[[STEP]]) : (!mo.tensor<[6, 2, 2], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[3, 2, 2], f32>
// CHECK:       mo.output %[[SLICE]]

mo.graph @swim_reshape_through_slice_first_dim(%arg0: !mo.tensor<[6, 4], f32>) -> !mo.tensor<[3, 2, 2], f32> {
  %start = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop = mo.constant {value = #M.dense_array<4, 9223372036854775807> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %slice = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[6, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 4], f32>
  %reshape = mo.static.reshape(%slice) : !mo.tensor<[3, 4], f32> -> !mo.tensor<[3, 2, 2], f32>
  mo.output %reshape : !mo.tensor<[3, 2, 2], f32>
}

// -----

// Expansion case: the sliced dimension expands into multiple output dimensions
// in the reshape. The slice boundaries are aligned to the contraction factor.
//
// BEFORE: [2, 8] -> slice -> [2, 4] -> reshape -> [2, 2, 2]
// AFTER:  [2, 8] -> reshape -> [2, 4, 2] -> slice -> [2, 2, 2]

// CHECK-LABEL: mo.graph @swim_reshape_through_slice_expansion
// CHECK-DAG:   %[[STEP:.*]] = mo.constant {value = #M.dense_array<1, 1, 1>
// CHECK-DAG:   %[[STOP:.*]] = mo.constant {value = #M.dense_array<9223372036854775807, 2, 9223372036854775807>
// CHECK-DAG:   %[[START:.*]] = mo.constant {value = #M.dense_array<0, 0, 0>
// CHECK:       %[[RESHAPE:.*]] = mo.static.reshape(%arg0) : !mo.tensor<[2, 8], f32> -> !mo.tensor<[2, 4, 2], f32>
// CHECK:       %[[SLICE:.*]] = mo.slice(%[[RESHAPE]], %[[START]], %[[STOP]], %[[STEP]]) : (!mo.tensor<[2, 4, 2], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
// CHECK:       mo.output %[[SLICE]]

mo.graph @swim_reshape_through_slice_expansion(%arg0: !mo.tensor<[2, 8], f32>) -> !mo.tensor<[2, 2, 2], f32> {
  %start = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop = mo.constant {value = #M.dense_array<9223372036854775807, 4> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %slice = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 4], f32>
  %reshape = mo.static.reshape(%slice) : !mo.tensor<[2, 4], f32> -> !mo.tensor<[2, 2, 2], f32>
  mo.output %reshape : !mo.tensor<[2, 2, 2], f32>
}

// -----

// No conversion: expansion case with step != 1.

// CHECK-LABEL: mo.graph @no_conversion_expansion_step_not_one
// CHECK:       mo.slice
// CHECK:       mo.static.reshape

mo.graph @no_conversion_expansion_step_not_one(%arg0: !mo.tensor<[2, 8], f32>) -> !mo.tensor<[2, 2, 2], f32> {
  %start = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop = mo.constant {value = #M.dense_array<9223372036854775807, 8> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %slice = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 4], f32>
  %reshape = mo.static.reshape(%slice) : !mo.tensor<[2, 4], f32> -> !mo.tensor<[2, 2, 2], f32>
  mo.output %reshape : !mo.tensor<[2, 2, 2], f32>
}

// -----

// No conversion: expansion case with unaligned start.

// CHECK-LABEL: mo.graph @no_conversion_expansion_unaligned_start
// CHECK:       mo.slice
// CHECK:       mo.static.reshape

mo.graph @no_conversion_expansion_unaligned_start(%arg0: !mo.tensor<[2, 8], f32>) -> !mo.tensor<[2, 2, 2], f32> {
  %start = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop = mo.constant {value = #M.dense_array<9223372036854775807, 5> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %slice = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 4], f32>
  %reshape = mo.static.reshape(%slice) : !mo.tensor<[2, 4], f32> -> !mo.tensor<[2, 2, 2], f32>
  mo.output %reshape : !mo.tensor<[2, 2, 2], f32>
}
