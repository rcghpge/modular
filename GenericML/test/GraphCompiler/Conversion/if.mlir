// RUN: mo-opt %s --mo-to-mogg --mo-to-primitives | FileCheck %s

// CHECK: mgp.model @if_op
// CHECK:  } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// CHECK: %[[COND:.*]] = mgp.tensor.create %[[CHAIN1:.*]], %[[CTX1:.*]], %[[BUF1:.*]], %[[SPEC1:.*]] : !mgp.tensor<bool>
// CHECK: %[[VAL1:.*]] = mgp.tensor.create %[[CHAIN2:.*]], %[[CTX1]], %[[BUF2:.*]], %[[SPEC2:.*]] : !mgp.tensor<2xf32>
// CHECK: %[[VAL2:.*]] = mgp.tensor.create %[[CHAIN3:.*]], %[[CTX1]], %[[BUF3:.*]], %[[SPEC4:.*]] : !mgp.tensor<2xf32>
// CHECK: %[[CHAIN_MERGE:.*]] = grt.chain.create (%[[CHAIN]], %[[CHAIN]])
// CHECK: %[[BOOL_BUF:.*]] = mgp.tensor.extract.buffer %[[CTX1]], %[[COND]] : !mgp.tensor<bool>
// CHECK: %[[BOOL:.*]] = mgp.buffer.to_bool %[[CHAIN4:.*]], %[[CTX1]], %[[BOOL_BUF]] : !mgp.buffer<1>
// CHECK: %[[OUT:.*]]:2 = grt.if %[[BOOL]] (%[[CHAIN]]: !mef.chain, %[[VAL1]]: !mgp.tensor<2xf32>, %[[CHAIN_MERGE]]: !mef.chain, %[[VAL2]]: !mgp.tensor<2xf32>)
// CHECK: mef.output %[[CHAIN]], %[[VAL1]]
// CHECK: } else {
// CHECK: mef.output %[[CHAIN_MERGE]], %[[VAL2]]
// CHECK: }
// CHECK: mef.output %[[OUT]]#0, %[[OUT]]#1 : !mef.chain, !mgp.tensor<2xf32>
mo.graph @if_op() -> (!mo.tensor<[2], f32>) {
  %cond = mo.constant {value = #M.dense_array<1> : tensor<1xi8>} : !mo.tensor<[], bool>
  %val1 = mo.constant {value = #M.dense_array<1.0, 2.0> : tensor<2xf32>} : !mo.tensor<[2], f32>
  %val2 = mo.constant {value = #M.dense_array<3.0, 4.0> : tensor<2xf32>} : !mo.tensor<[2], f32>
  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
    mo.yield %val1 : !mo.tensor<[2], f32>
  } else {
    mo.yield %val2 : !mo.tensor<[2], f32>
  }
  mo.output %0 : !mo.tensor<[2], f32>
}

// CHECK: mgp.model @if_op_scalar_cond
// CHECK: [[BOOL:%.*]] = grt.constant.i1
// CHECK: grt.if [[BOOL]]
mo.graph @if_op_scalar_cond() -> (!mo.tensor<[2], f32>) {
  %cond = mo.constant.scalar {value = 1 : i1} : !mo.scalar<bool>
  %val1 = mo.constant {value = #M.dense_array<1.0, 2.0> : tensor<2xf32>} : !mo.tensor<[2], f32>
  %val2 = mo.constant {value = #M.dense_array<3.0, 4.0> : tensor<2xf32>} : !mo.tensor<[2], f32>
  %0 = mo.if(%cond : !mo.scalar<bool>) -> !mo.tensor<[2], f32> {
    mo.yield %val1 : !mo.tensor<[2], f32>
  } else {
    mo.yield %val2 : !mo.tensor<[2], f32>
  }
  mo.output %0 : !mo.tensor<[2], f32>
}


// CHECK-LABEL: mgp.model @if_op_multi_if
// CHECK: } execute (%[[CH0:.*]]: !mef.chain,

// CHECK: %[[IF_OUT:.*]]:2 = grt.if
// CHECK: %[[TRUE_CH0:.*]] = mogg.generic.compiled.region (%[[CH0]]
// CHECK: mef.output %[[TRUE_CH0]]
// } else {
// CHECK: %[[FALSE_CH0:.*]] = mogg.generic.compiled.region (%[[CH0]]
// CHECK: mef.output %[[FALSE_CH0]]

// CHECK: grt.if
// CHECK: %[[CH1:.*]] = grt.chain.create (%[[IF_OUT]]#0, %[[IF_OUT]]#0, %[[CH0]])
// CHECK: %[[TRUE_CH1:.*]] = mogg.generic.compiled.region (%[[CH1]]
// CHECK: mef.output %[[TRUE_CH1]]
// } else {
// CHECK: %[[CH1:.*]] = grt.chain.create (%[[IF_OUT]]#0, %[[CH0]])
// CHECK: %[[FALSE_CH1:.*]] = mogg.generic.compiled.region (%[[CH1]]
// CHECK: mef.output %[[FALSE_CH1]]

mo.graph @if_op_multi_if(%arg0: !mo.tensor<[2], f32>) -> (!mo.tensor<[2], f32>) {
  %cond = mo.constant {value = #M.dense_array<1> : tensor<1xi8>} : !mo.tensor<[], bool>
  %if_0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
    %1 = mo.mul(%arg0, %arg0): !mo.tensor<[2], f32>
    mo.yield %1 : !mo.tensor<[2], f32>
  } else {
    %1 = mo.relu(%arg0): !mo.tensor<[2], f32>
    mo.yield %1 : !mo.tensor<[2], f32>
  }

  %if_1 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
    %1 = mo.mul(%if_0, %if_0): !mo.tensor<[2], f32>
    mo.yield %1 : !mo.tensor<[2], f32>
  } else {
    %1 = mo.relu(%if_0): !mo.tensor<[2], f32>
    mo.yield %1 : !mo.tensor<[2], f32>
  }
  mo.output %if_1 : !mo.tensor<[2], f32>
}