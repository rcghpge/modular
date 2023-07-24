// RUN: mgp-opt %s --allow-unregistered-dialect --async-dataflow-dependencies-test --simplify-chains --cse --mlir-disable-threading | FileCheck %s

// CHECK-LABEL: mgp.model @chains : tf init {
mgp.model @chains : tf init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
// CHECK: execute (%[[CH0:.*]]: !mef.chain,
} execute (%ch0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
  // %ch1 -> %ch0
  %ch1 = grt.chain.create (%ch0)
  // %ch2 -> %ch0
  %ch2 = grt.chain.create (%ch1, %ch1)
  // CHECK: %[[CH3:.*]] = "some.op1"(%[[CH0]])
  %ch3 = "some.op1"(%ch2) : (!mef.chain) -> !mef.chain
  // %ch4 -> %ch3
  %ch4 = grt.chain.create (%ch0, %ch3)
  // CHECK: %[[CH5:.*]] = "some.op2"(%[[CH3]])
  %ch5 = "some.op2"(%ch4) : (!mef.chain) -> !mef.chain
  // CHECK: %[[CH6:.*]] = "some.op3"(%[[CH3]])
  %ch6 = "some.op3"(%ch4) : (!mef.chain) -> !mef.chain
  // CHECK: %[[CH7:.*]] = "some.op4"(%[[CH3]])
  %ch7 = "some.op4"(%ch4) : (!mef.chain) -> !mef.chain
  // Dead
  %ch8 = grt.chain.create (%ch5, %ch6)
  // %ch9 -> grt.chain.create(%ch5, %ch6, %ch7)
  // Dead
  %ch9 = grt.chain.create (%ch4, %ch8, %ch7)
  // %ch10 -> %ch0
  %ch10 = grt.chain.create ()
  // Dead
  %ch11 = grt.chain.create (%ch10, %ch9)
  %c = "some.op5"() : () -> i1
  // %ch12 is always ready
  // CHECK: %[[CH12:.*]] = grt.if %{{.*}} (%[[CH0]]: !mef.chain) -> !mef.chain {
  %ch12 = grt.if %c (%ch10 : !mef.chain) -> !mef.chain {
    mef.output %ch10 : !mef.chain
  } else {
    // %ch13 is always ready
    // CHECK: %[[CH13:.*]] = grt.chain.create ()
    %ch13 = grt.chain.create ()
    mef.output %ch13 : !mef.chain
  }
  // %ch14 -> expansion of %ch11
  // CHECK: %[[CH14:.*]] = grt.chain.create (%[[CH5]], %[[CH6]], %[[CH7]])
  %ch14 = grt.chain.create (%ch10, %ch11, %ch12)
  // CHECK: mef.output %[[CH14]]
  mef.output %ch14 : !mef.chain
}

// CHECK-LABEL: mgp.model @arg0_not_always_ready
mgp.model @arg0_not_always_ready : tf init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  %2 = "some.op" (%0) : (!mef.chain) -> !mef.chain
  mef.output %2, %1 : !mef.chain, !mgp.context
// CHECK: execute (%[[CH0:.*]]: !mef.chain,
} execute (%ch0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
  // CHECK-NEXT: %{{.*}} = grt.chain.create ()
  %ch1 = grt.chain.create ()
  mef.output %ch1 : !mef.chain
}


// CHECK-LABEL: mgp.model @chain_merge_canonicalized
mgp.model @chain_merge_canonicalized : tf init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
// CHECK: execute
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
  // CHECK-NEXT: %[[CH0:.*]] = "some.op"()
  %ch0 = "some.op"() : () -> !mef.chain
  // CHECK-NEXT: %[[CH1:.*]] = "some.op"()
  %ch1 = "some.op"() : () -> !mef.chain

  // CHECK-NEXT: %[[RES:.*]] = grt.chain.create (%[[CH0:.*]], %[[CH1:.*]])
  %ch2 = grt.chain.create (%ch0, %ch1)
  %ch3 = grt.chain.create (%ch1, %ch0)

  // CHECK: %[[CH4:.*]] = "some.op"(%[[RES]], %[[CH3:.*]]) : (!mef.chain, !mef.chain) -> !mef.chain
  %ch4 = "some.op"(%ch2, %ch3) : (!mef.chain, !mef.chain) -> !mef.chain

  mef.output %ch4 : !mef.chain
}

// CHECK-LABEL: mgp.model @chain_merge_fold_empty_operands
mgp.model @chain_merge_fold_empty_operands : tf init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
// CHECK: execute (%[[CH0:.*]]: !mef.chain
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
  %ch1 = grt.chain.create ()
  // CHECK-NEXT: mef.output %[[CH0]] : !mef.chain
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @chain_merge_fold_consecutive
mgp.model @chain_merge_fold_consecutive : tf init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
// CHECK: execute
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
  // CHECK-NEXT: %[[CH0:.*]] = "some.op"()
  %ch0 = "some.op"() : () -> !mef.chain
  // CHECK-NEXT: %[[CH1:.*]] = "some.op"()
  %ch1 = "some.op"() : () -> !mef.chain
  // CHECK-NEXT: %[[CH2:.*]] = "some.op"()
  %ch2 = "some.op"() : () -> !mef.chain

  // CHECK-NEXT: %[[RES:.*]] = grt.chain.create (%[[CH0:.*]], %[[CH1:.*]], %[[CH2:.*]])
  %ch3 = grt.chain.create (%ch0, %ch1)
  %ch4 = grt.chain.create (%ch3, %ch2)

  // CHECK-NEXT: mef.output %[[RES]] : !mef.chain
  mef.output %ch4 : !mef.chain
}

// CHECK-LABEL: mgp.model @chain_merge_fold_operands_uniqued
mgp.model @chain_merge_fold_operands_uniqued : tf init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
// CHECK: execute
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
  // CHECK-NEXT: %[[RES:.*]] = "some.op"()
  %ch0 = "some.op"() : () -> !mef.chain

  %ch1 = grt.chain.create (%ch0, %ch0)
  %ch2 = grt.chain.create (%ch0, %ch1, %ch0)

  // CHECK-NEXT: mef.output %[[RES]] : !mef.chain
  mef.output %ch2 : !mef.chain
}
