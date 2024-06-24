// RUN: mo-opt %s --add-devices --mo-to-primitives | FileCheck %s

// CHECK-LABEL:  mgp.model @constants_with_multiple_returns : mof
// CHECK:  } execute (
// CHECK-SAME: %[[ARG0:.*]]: !mef.chain, %[[ARG1:.*]]: !mgp.context,
// CHECK-SAME: %[[ARG2:[^:]*]]: !mgp.tensor<2xf32>, %[[ARG3:.*]]: !mgp.tensor<5x1xf32>
// CHECK-SAME: ) -> (!mef.chain, !mgp.tensor<2xf32>, !mgp.tensor<3xf32>, !mgp.tensor<3xf32>,
// CHECK-SAME: !mgp.tensor<5x1xf32>, !mgp.tensor<5x1xf32>, !mgp.tensor<4xsi8>) {
// CHECK-DAG:    %[[VAL0:.*]] = mgp.buffer.constant %{{.*}}, %[[ARG1]][1], #M.aligned_bytes<"0x0000803F0000004000004040", align {{.*}}> : !M.array<12xui8>
// CHECK-DAG:    %[[VAL1:.*]] = mgp.tensor_spec.create  : !mgp.tensor_spec<3xf32>
// CHECK-DAG:    %[[VAL2:.*]] = mgp.chain.host_to_device %[[ARG0]], %[[ARG1]][1]
// CHECK-DAG:    %[[VAL3:.*]] = mgp.tensor.create %[[VAL2]], %[[ARG1]][1], %[[VAL0]], %[[VAL1]] : !mgp.tensor<3xf32>
// CHECK-DAG:    %[[VAL4:.*]] = mgp.buffer.constant %{{.*}}, %[[ARG1]][1], dense_resource<mtfl_buffer> : !M.aligned_bytes<4, align 64>
// CHECK-DAG:    %[[VAL5:.*]] = mgp.tensor_spec.create  : !mgp.tensor_spec<4xsi8>
// CHECK-DAG:    %[[VAL6:.*]] = mgp.chain.host_to_device %[[ARG0]], %[[ARG1]][1]
// CHECK:    %[[VAL7:.*]] = mgp.tensor.create %[[VAL6]], %[[ARG1]][1], %[[VAL4]], %[[VAL5]] : !mgp.tensor<4xsi8>
// CHECK:    mef.output %[[ARG0]], %[[ARG2]], %[[VAL3]], %[[VAL3]], %[[ARG3]], %[[ARG3]], %[[VAL7]] : !mef.chain, !mgp.tensor<2xf32>, !mgp.tensor<3xf32>, !mgp.tensor<3xf32>, !mgp.tensor<5x1xf32>, !mgp.tensor<5x1xf32>, !mgp.tensor<4xsi8>
// CHECK:  }

mo.graph @constants_with_multiple_returns(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[5, 1], f32>) -> (
    !mo.tensor<[2], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[4], si8>
) {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>} : !mo.tensor<[3], f32>
  %1 = mo.constant {value = dense_resource<mtfl_buffer> : tensor<4xsi8>} : !mo.tensor<[4], si8>
  mo.output %arg0, %0, %0, %arg1, %arg1, %1 : !mo.tensor<[2], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[4], si8>
}

// CHECK-LABEL: mgp.model @foo : mof
// CHECK: } execute (
// CHECK-SAME: %[[ARG0:.*]]: !mef.chain, %[[ARG1:.*]]: !mgp.context,
// CHECK-SAME: %[[ARG2:[^:]*]]: !mgp.tensor<1x?xf32>, %[[ARG3:.*]]: !mgp.tensor<?xf32>) -> !mef.chain {
// CHECK:   %[[VAL0:.*]] = mgp.tensor.extract.tensor_spec %[[ARG1]][1], %[[ARG2]] : !mgp.tensor<1x?xf32>
// CHECK:   %[[VAL1:.*]] = mgp.tensor_spec.dims.dynamic %[[VAL0]] : !mgp.tensor_spec<1x?xf32>
// CHECK:   %[[VAL2:.*]] = mgp.tensor.extract.tensor_spec %[[ARG1]][1], %[[ARG3]] : !mgp.tensor<?xf32>
// CHECK:   %[[VAL3:.*]] = mgp.tensor_spec.dims.dynamic %[[VAL2]] : !mgp.tensor_spec<?xf32>
// CHECK:   mef.output %[[ARG0:.*]] : !mef.chain
mo.graph @foo<N>(%arg0: !mo.tensor<[1, N], f32>, %arg1: !mo.tensor<[N], f32>) {
  mo.output
}

// CHECK-LABEL: mgp.model @scalar_arg : mof
// CHECK: } execute (
// CHECK-SAME: %[[ARG0:.*]]: !mef.chain, %[[ARG1:.*]]: !mgp.context,
// CHECK-SAME: %[[ARG2:[^:]*]]: !mgp.tensor<f32>) -> (!mef.chain, !mgp.tensor<f32>) {
// CHECK:   mef.output %[[ARG0]], %[[ARG2]] : !mef.chain, !mgp.tensor<f32>
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

