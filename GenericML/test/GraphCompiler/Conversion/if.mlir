// RUN: mo-opt %s --add-devices --mo-to-mogg --mo-to-primitives --allow-unregistered-dialect | FileCheck %s

// CHECK: mgp.model @if_op
// CHECK:  } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// CHECK: %[[COND:.*]] = mgp.cpu.tensor.create %[[CHAIN]], %[[BUF1:.*]], %[[SPEC1:.*]] : !mgp.cpu.tensor<bool>
// CHECK: %[[VAL1:.*]] = mgp.cpu.tensor.create %[[CHAIN]], %[[BUF2:.*]], %[[SPEC2:.*]] : !mgp.cpu.tensor<2xf32>
// CHECK: %[[VAL2:.*]] = mgp.cpu.tensor.create %[[CHAIN]], %[[BUF3:.*]], %[[SPEC4:.*]] : !mgp.cpu.tensor<2xf32>
// CHECK: %[[BUF3:.*]], %[[SPEC3:.*]] = mgp.cpu.tensor.extract %[[COND]]
// CHECK: %[[BOOL:.*]] = mgp.to_grt_bool(%[[CHAIN]], %[[BUF3]])
// CHECK: %[[OUT:.*]]:2 = grt.if %[[BOOL]] (%[[CHAIN]]: !mef.chain, %[[VAL1]]: !mgp.cpu.tensor<2xf32>, %[[VAL2]]: !mgp.cpu.tensor<2xf32>)
// CHECK: mef.output %[[CHAIN]], %[[VAL1]]
// CHECK: } else {
// CHECK: mef.output %[[CHAIN]], %[[VAL2]]
// CHECK: }
// CHECK: mef.output %[[OUT]]#0, %[[OUT]]#1 : !mef.chain, !mgp.cpu.tensor<2xf32>
mo.graph @if_op() -> (!mo.tensor<[2], f32>) {
  %cond = mo.constant {value = #M.dense_array<1> : tensor<1xi1>} : !mo.tensor<[], bool>
  %val1 = mo.constant {value = #M.dense_array<1.0, 2.0> : tensor<2xf32>} : !mo.tensor<[2], f32>
  %val2 = mo.constant {value = #M.dense_array<3.0, 4.0> : tensor<2xf32>} : !mo.tensor<[2], f32>
  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
    mo.yield %val1 : !mo.tensor<[2], f32>
  } else {
    mo.yield %val2 : !mo.tensor<[2], f32>
  }
  mo.output %0 : !mo.tensor<[2], f32>
}
