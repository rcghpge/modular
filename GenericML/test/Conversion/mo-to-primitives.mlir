// RUN: mo-opt %s --mo-to-primitives --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: mgp.model @constants_with_multiple_returns : tfl init {
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
// CHECK-DAG:    %[[VAL0:.*]] = mgp.cpu.buffer.constant #M.aligned_bytes<{{.*}} :"0x0000803F0000004000004040">
// CHECK-DAG:    %[[VAL1:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<3xf32>
// CHECK:        %[[VAL2:.*]] = mgp.cpu.tensor.create %[[ARG0]], %[[VAL0]], %[[VAL1]] : !mgp.cpu.tensor<3xf32>
// CHECK-DAG:    %[[VAL3:.*]] = mgp.cpu.buffer.constant dense_resource<mtfl_buffer> : !mgp.aligned_bytes<4, 64>
// CHECK-DAG:    %[[VAL4:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<4xsi8>
// CHECK:        %[[VAL5:.*]] = mgp.cpu.tensor.create %[[ARG0]], %[[VAL3]], %[[VAL4]] : !mgp.cpu.tensor<4xsi8>
// CHECK:        bef.output %[[ARG0]], %[[ARG2]], %[[VAL2]], %[[VAL2]], %[[ARG3]], %[[ARG3]], %[[VAL5]]
// CHECK:      }

mo.graph @constants_with_multiple_returns(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[5, 1], f32>) -> (
    !mo.tensor<[2], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[4], si8>
) {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>} : !mo.tensor<[3], f32>
  %1 = mo.constant {value = dense_resource<mtfl_buffer> : tensor<4xsi8>} : !mo.tensor<[4], si8>
  mo.output %arg0, %0, %0, %arg1, %arg1, %1 : !mo.tensor<[2], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[4], si8>
}

// CHECK-LABEL: mgp.model @scalar_arg : tfl init {
// CHECK:    %0 = grt.chain.create
// CHECK:    %1 = mgp.context.create 0
// CHECK:    bef.output %0, %1 : !bef.chain, !mgp.context
// CHECK:  } execute (%arg0: !bef.chain, %arg1: !mgp.context, %arg2: !mgp.cpu.tensor<f32>) -> (!bef.chain, !mgp.cpu.tensor<f32>) {
// CHECK:    bef.output %arg0, %arg2 : !bef.chain, !mgp.cpu.tensor<f32>
// CHECK:  }
mo.graph @scalar_arg(%arg0: !mo.tensor<[], f32>) -> (!mo.tensor<[], f32>) {
  mo.output %arg0 : !mo.tensor<[], f32>
}

{-#
  dialect_resources: {
    builtin: {
      mtfl_buffer: "0x4000000001234567"
    }
  }
#-}
