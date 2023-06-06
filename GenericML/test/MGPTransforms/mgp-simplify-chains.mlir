// RUN: mgp-opt %s --allow-unregistered-dialect --simplify-chains | FileCheck %s

// CHECK-LABEL: mgp.model @chains : tf init {
mgp.model @chains : tf init {
  %0 = grt.chain.create
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
// CHECK: execute (%[[CH0:.*]]: !mef.chain,
} execute (%ch0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
  // %ch1 -> %ch0
  %ch1 = grt.chain.merge (%ch0)
  // %ch2 -> %ch0
  %ch2 = grt.chain.merge (%ch1, %ch1)
  // CHECK: %[[CH3:.*]] = "some.op1"(%[[CH0]])
  %ch3 = "some.op1"(%ch2) : (!mef.chain) -> !mef.chain
  // %ch4 -> %ch3
  %ch4 = grt.chain.merge (%ch0, %ch3)
  // CHECK: %[[CH5:.*]] = "some.op2"(%[[CH3]])
  %ch5 = "some.op2"(%ch4) : (!mef.chain) -> !mef.chain
  // CHECK: %[[CH6:.*]] = "some.op3"(%[[CH3]])
  %ch6 = "some.op3"(%ch4) : (!mef.chain) -> !mef.chain
  // CHECK: %[[CH7:.*]] = "some.op4"(%[[CH3]])
  %ch7 = "some.op4"(%ch4) : (!mef.chain) -> !mef.chain
  // CHECK: %[[CH8:.*]] = grt.chain.merge (%[[CH5]], %[[CH6]])
  %ch8 = grt.chain.merge (%ch5, %ch6)
  // %ch9 -> grt.chain.merge(%ch5, %ch6, %ch7)
  // CHECK: %[[CH9:.*]] = grt.chain.merge (%[[CH5]], %[[CH6]], %[[CH7]])
  %ch9 = grt.chain.merge (%ch4, %ch8, %ch7)
  // %ch10 is always ready
  // CHECK: %[[CH10:.*]] = grt.chain.merge ()
  %ch10 = grt.chain.merge ()
  // CHECK: %[[CH11:.*]] = grt.chain.merge (%[[CH5]], %[[CH6]], %[[CH7]])
  %ch11 = grt.chain.merge (%ch10, %ch9)
  %c = "some.op5"() : () -> i1
  // %ch12 is always ready
  %ch12 = grt.if %c (%ch10 : !mef.chain) -> !mef.chain {
    mef.output %ch10 : !mef.chain
  } else {
    // %ch13 is always ready
    %ch13 = grt.chain.create
    mef.output %ch13 : !mef.chain
  }
  // %ch14 -> expansion of %ch11
  // CHECK: %[[CH14:.*]] = grt.chain.merge (%[[CH5]], %[[CH6]], %[[CH7]])
  %ch14 = grt.chain.merge (%ch10, %ch11, %ch12)
  // CHECK: mef.output %[[CH14]]
  mef.output %ch14 : !mef.chain
}
