// RUN: mo-opt %s --mo-to-mogg --mo-to-primitives --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: mgp.model @concat_static_output
// CHECK:  } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]
// CHECK: %[[OUT_CHAIN:.*]] = "mop.kernels.concat"(%[[CHAIN]]
// CHECK-NEXT: %[[OUTPUT_TENSOR:.*]] =  mgp.cpu.tensor.create %[[OUT_CHAIN]]
// CHECK-NEXT: mef.output %[[CHAIN]], %[[OUTPUT_TENSOR]] : !mef.chain, !mgp.cpu.tensor<6xsi32>
mo.graph @concat_static_output(%arg1 : !mo.tensor<[1], si32>) -> (!mo.tensor<[6], si32>) {
  %ax = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %arg2 = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  %arg3 = mo.constant {value = #M.dense_array<3, 3, 3> : tensor<3xsi32>} : !mo.tensor<[3], si32>
  %out = mo.concat[%ax: !mo.tensor<[], si32>](%arg1, %arg2, %arg3) : (
    !mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>
  ) -> !mo.tensor<[6], si32>
  mo.output %out : !mo.tensor<[6], si32>
}

// CHECK-LABEL: mgp.model @concat_dyn_output
// CHECK:  } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]
// CHECK: "mop.shape.concat"({{.*}}) :
// CHECK-SAME: (!mgp.cpu.tensor_spec<1xsi32>, !mgp.cpu.tensor_spec<2xsi32>, !mgp.cpu.tensor_spec<3xsi32>, !mgp.cpu.tensor<si32>) -> !mgp.cpu.tensor_spec<?xsi32>
// CHECK: %[[OUT_CHAIN:.*]] = "mop.kernels.concat"(%[[CHAIN]]
// CHECK-NEXT: %[[OUTPUT_TENSOR:.*]] =  mgp.cpu.tensor.create %[[OUT_CHAIN]]
// CHECK-NEXT: mef.output %[[CHAIN]], %[[OUTPUT_TENSOR]] : !mef.chain, !mgp.cpu.tensor<?xsi32>
mo.graph @concat_dyn_output() -> (!mo.tensor<[?], si32>) {
  %ax = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %arg1 = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %arg2 = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  %arg3 = mo.constant {value = #M.dense_array<3, 3, 3> : tensor<3xsi32>} : !mo.tensor<[3], si32>
  %out = mo.concat[%ax: !mo.tensor<[], si32>](%arg1, %arg2, %arg3) : (
    !mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>
  ) -> !mo.tensor<[?], si32>
  mo.output %out : !mo.tensor<[?], si32>
}
