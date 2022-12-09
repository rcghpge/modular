// RUN: mgp-opt %s --mo-to-primitives | FileCheck %s

// CHECK:      mgp.model @constants_with_multiple_returns : mgp init {
// CHECK:        %0 = grt.chain.create
// CHECK:        %1 = mgp.context.create 0
// CHECK:        bef.output %0, %1 : !bef.chain, !mgp.context
// CHECK:      } execute (
// CHECK-SAME:   %[[ARG0:.*]]: !bef.chain, %[[ARG1:.*]]: !mgp.context,
// CHECK-SAME:   %[[ARG2:.*]]: !mgp.cpu.tensor<2xf32>, %[[ARG3:.*]]: !mgp.cpu.tensor<5x1xf32>
// CHECK-SAME: ) -> (
// CHECK-SAME:   !bef.chain, !mgp.cpu.tensor<2xf32>, !mgp.cpu.tensor<3xf32>,
// CHECK-SAME:   !mgp.cpu.tensor<3xf32>, !mgp.cpu.tensor<5x1xf32>, !mgp.cpu.tensor<5x1xf32>, !mgp.cpu.tensor<4xsi8>)
// CHECK-SAME: {
// CHECK-DAG:    %[[VAL0:.*]] = mgp.cpu.buffer.constant #M.dense_array<0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64> : !M.array<12xui8>
// CHECK-DAG:    %[[VAL1:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<3xf32>
// CHECK:        %[[VAL2:.*]] = mgp.cpu.tensor.create %[[ARG0]] %[[VAL0]] %[[VAL1]] : !mgp.cpu.tensor<3xf32>
// CHECK-DAG:    %[[VAL3:.*]] = mgp.cpu.buffer.constant dense_resource<mtfl_buffer> : !M.array<4xui8>
// CHECK-DAG:    %[[VAL4:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<4xsi8>
// CHECK:        %[[VAL5:.*]] = mgp.cpu.tensor.create %[[ARG0]] %[[VAL3]] %[[VAL4]] : !mgp.cpu.tensor<4xsi8>
// CHECK:        bef.output %[[ARG0]], %[[ARG2]], %[[VAL2]], %[[VAL2]], %[[ARG3]], %[[ARG3]], %[[VAL5]]
// CHECK:      }
mo.graph @constants_with_multiple_returns(%arg0: !mo.tensor<2xf32>, %arg1: !mo.tensor<5x1xf32>) -> (
    !mo.tensor<2xf32>, !mo.tensor<3xf32>, !mo.tensor<3xf32>, !mo.tensor<5x1xf32>, !mo.tensor<5x1xf32>, !mo.tensor<4xsi8>
) {
  %0 = "mo.constant"() {value = #M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>} : () -> !mo.tensor<3xf32>
  %1 = "mo.constant"() {value = dense_resource<mtfl_buffer> : tensor<4xsi8>} : () -> !mo.tensor<4xsi8>
  mo.output %arg0, %0, %0, %arg1, %arg1, %1 : !mo.tensor<2xf32>, !mo.tensor<3xf32>, !mo.tensor<3xf32>, !mo.tensor<5x1xf32>, !mo.tensor<5x1xf32>, !mo.tensor<4xsi8>
}

{-#
  dialect_resources: {
    builtin: {
      mtfl_buffer: "0x4000000001234567"
    }
  }
#-}
