// RUN: mo-opt %s --mo-to-primitives --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: mgp.model @constants_with_multiple_returns : tf init {
// CHECK-NEXT:        %0 = grt.chain.create ()
// CHECK-NEXT:        %1 = mgp.context.create 1
// CHECK-NEXT:        %2 = "mop.create_jit_context"(%0, %1) {slot = 0 : ui32}
// CHECK-NEXT:        mef.output %2, %1 : !mef.chain, !mgp.context
// CHECK-NEXT:      } execute (
// CHECK-SAME:   %[[ARG0:.*]]: !mef.chain, %[[ARG1:.*]]: !mgp.context,
// CHECK-SAME:   %[[ARG2:.*]]: !mgp.cpu.tensor<2xf32>, %[[ARG3:.*]]: !mgp.cpu.tensor<5x1xf32>
// CHECK-SAME: ) -> (
// CHECK-SAME:   !mef.chain, !mgp.cpu.tensor<2xf32>, !mgp.cpu.tensor<3xf32>,
// CHECK-SAME:   !mgp.cpu.tensor<3xf32>, !mgp.cpu.tensor<5x1xf32>, !mgp.cpu.tensor<5x1xf32>, !mgp.cpu.tensor<4xsi8>)
// CHECK-SAME: {
// CHECK-DAG:    %[[VAL0:.*]] = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000803F0000004000004040", align {{.*}}>
// CHECK-DAG:    %[[VAL1:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<3xf32>
// CHECK:        %[[VAL2:.*]] = mgp.cpu.tensor.create %[[ARG0]], %[[VAL0]], %[[VAL1]] : !mgp.cpu.tensor<3xf32>
// CHECK-DAG:    %[[VAL3:.*]] = mgp.cpu.buffer.constant dense_resource<mtfl_buffer> : !M.aligned_bytes<4, align 64>
// CHECK-DAG:    %[[VAL4:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<4xsi8>
// CHECK:        %[[VAL5:.*]] = mgp.cpu.tensor.create %[[ARG0]], %[[VAL3]], %[[VAL4]] : !mgp.cpu.tensor<4xsi8>
// CHECK:        mef.output %[[ARG0]], %[[ARG2]], %[[VAL2]], %[[VAL2]], %[[ARG3]], %[[ARG3]], %[[VAL5]]
// CHECK:      }

mo.graph @constants_with_multiple_returns(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[5, 1], f32>) -> (
    !mo.tensor<[2], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[4], si8>
) {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>} : !mo.tensor<[3], f32>
  %1 = mo.constant {value = dense_resource<mtfl_buffer> : tensor<4xsi8>} : !mo.tensor<[4], si8>
  mo.output %arg0, %0, %0, %arg1, %arg1, %1 : !mo.tensor<[2], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[4], si8>
}

// CHECK-LABEL: mgp.model @foo : tf init {
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK-SAME: %[[ARG2:.*]]: !mgp.cpu.tensor<1x?xf32>, %[[ARG3:.*]]: !mgp.cpu.tensor<?xf32>) -> !mef.chain {
// CHECK:   %[[BUF1:.*]], %[[SPEC1:.*]] = mgp.cpu.tensor.extract %[[ARG2]] : !mgp.cpu.tensor<1x?xf32>
// CHECK:   %[[VAL0:.*]] = mgp.cpu.tensor_spec.dims.dynamic %[[SPEC1]] : !mgp.cpu.tensor_spec<1x?xf32>
// CHECK:   %[[BUF2:.*]], %[[SPEC2:.*]] = mgp.cpu.tensor.extract %[[ARG3]] : !mgp.cpu.tensor<?xf32>
// CHECK:   %[[VAL1:.*]] = mgp.cpu.tensor_spec.dims.dynamic %[[SPEC2]] : !mgp.cpu.tensor_spec<?xf32>
// CHECK:   mef.output %[[CHAIN]] : !mef.chain
// CHECK: }
mo.graph @foo<N>(%arg0: !mo.tensor<[1, N], f32>, %arg1: !mo.tensor<[N], f32>) {
  mo.output
}

// CHECK-LABEL: mgp.model @scalar_arg : tf init {
// CHECK-NEXT:    %0 = grt.chain.create ()
// CHECK-NEXT:    %1 = mgp.context.create 1
// CHECK-NEXT:        %2 = "mop.create_jit_context"(%0, %1) {slot = 0 : ui32}
// CHECK:    mef.output %2, %1 : !mef.chain, !mgp.context
// CHECK:  } execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.cpu.tensor<f32>) -> (!mef.chain, !mgp.cpu.tensor<f32>) {
// CHECK:    mef.output %arg0, %arg2 : !mef.chain, !mgp.cpu.tensor<f32>
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

