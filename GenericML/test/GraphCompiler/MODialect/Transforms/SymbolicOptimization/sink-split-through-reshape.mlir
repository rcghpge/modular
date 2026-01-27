// RUN: mo-opt %s --mo-symbolic-optimize | FileCheck %s

// Test that verifies the SinkSplitThroughStaticReshapeLikeOpsPattern correctly
// creates host constants for split sizes and axis.

// CHECK-LABEL: mo.graph @sink_split_through_reshape
// CHECK-DAG: %[[SIZES:.*]] = mo.constant {value = #M.dense_array<2, 2> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-DAG: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: %[[RESHAPE:.*]] = mo.static.reshape(%arg0) : !mo.tensor<[2, 8], f32> -> !mo.tensor<[2, 4, 2], f32>
// CHECK-NEXT: %[[SPLIT:.*]]:2 = mo.split[%[[AXIS]] : !mo.tensor<[], si64>] (%[[RESHAPE]], %[[SIZES]]) : (!mo.tensor<[2, 4, 2], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>)
// CHECK-NEXT: mo.output %[[SPLIT]]#0, %[[SPLIT]]#1 : !mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>

mo.graph @sink_split_through_reshape(%in: !mo.tensor<[2, 8], f32>) -> (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>) {
  %0 = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %1 = mo.constant {value = #M.dense_array<4, 4> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %2:2 = mo.split[%0 : !mo.tensor<[], si32>] (%in, %1) : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 4], f32>, !mo.tensor<[2, 4], f32>)
  %3 = mo.static.reshape(%2#0) : !mo.tensor<[2, 4], f32> -> !mo.tensor<[2, 2, 2], f32>
  %4 = mo.static.reshape(%2#1) : !mo.tensor<[2, 4], f32> -> !mo.tensor<[2, 2, 2], f32>
  mo.output %3, %4 : !mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>
}

// -----

// Test with device placement to ensure constants are still created as host constants
// CHECK-LABEL: mo.graph @sink_split_through_reshape_with_device
// CHECK-DAG: %[[SIZES:.*]] = mo.constant {value = #M.dense_array<2, 2> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-DAG: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: %[[RESHAPE:.*]] = mo.static.reshape(%arg0) : !mo.tensor<[2, 8], f32, dev:0> -> !mo.tensor<[2, 4, 2], f32, dev:0>
// CHECK-NEXT: %[[SPLIT:.*]]:2 = mo.split[%[[AXIS]] : !mo.tensor<[], si64>] (%[[RESHAPE]], %[[SIZES]]) : (!mo.tensor<[2, 4, 2], f32, dev:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2, 2], f32, dev:0>, !mo.tensor<[2, 2, 2], f32, dev:0>)
// CHECK-NEXT: mo.output %[[SPLIT]]#0, %[[SPLIT]]#1 : !mo.tensor<[2, 2, 2], f32, dev:0>, !mo.tensor<[2, 2, 2], f32, dev:0>

mo.graph @sink_split_through_reshape_with_device(%in: !mo.tensor<[2, 8], f32, dev:0>) -> (!mo.tensor<[2, 2, 2], f32, dev:0>, !mo.tensor<[2, 2, 2], f32, dev:0>) {
  %0 = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %1 = mo.constant {value = #M.dense_array<4, 4> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %2:2 = mo.split[%0 : !mo.tensor<[], si32>] (%in, %1) : (!mo.tensor<[2, 8], f32, dev:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 4], f32, dev:0>, !mo.tensor<[2, 4], f32, dev:0>)
  %3 = mo.static.reshape(%2#0) : !mo.tensor<[2, 4], f32, dev:0> -> !mo.tensor<[2, 2, 2], f32, dev:0>
  %4 = mo.static.reshape(%2#1) : !mo.tensor<[2, 4], f32, dev:0> -> !mo.tensor<[2, 2, 2], f32, dev:0>
  mo.output %3, %4 : !mo.tensor<[2, 2, 2], f32, dev:0>, !mo.tensor<[2, 2, 2], f32, dev:0>
} 