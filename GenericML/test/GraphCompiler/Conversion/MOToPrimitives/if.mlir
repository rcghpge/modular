// RUN: mo-opt %s --mo-to-mogg --mo-to-primitives | FileCheck %s

// CHECK: mgp.model @if_op
// CHECK:  } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// CHECK: %[[COND:.*]] = mgp.tensor.create %[[CHAIN]], %[[BUF1:.*]], %[[SPEC1:.*]] : !mgp.tensor<bool>
// CHECK: %[[VAL1:.*]] = mgp.tensor.create %[[CHAIN]], %[[BUF2:.*]], %[[SPEC2:.*]] : !mgp.tensor<2xf32>
// CHECK: %[[VAL2:.*]] = mgp.tensor.create %[[CHAIN]], %[[BUF3:.*]], %[[SPEC4:.*]] : !mgp.tensor<2xf32>
// CHECK: %[[CHAIN_MERGE:.*]] = grt.chain.create (%[[CHAIN]], %[[CHAIN]])
// CHECK: %[[BOOL:.*]] = mgp.buffer.to_bool %[[CHAIN_MERGE]], %[[BUF1]] : !mgp.buffer
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
// CHECK: } execute (%[[ENTRY_CH:.*]]: !mef.chain,

// CHECK: %[[CH0:.*]] = grt.chain.create (%[[ENTRY_CH]], %[[ENTRY_CH]])

// CHECK: %[[IF_OUT:.*]]:2 = grt.if
// CHECK: %[[TRUE_CH0:.*]] = mogg.generic.compiled.region (%[[ENTRY_CH]]
// } else {
// CHECK: %[[FALSE_CH0:.*]] = mogg.generic.compiled.region (%[[CH0]]

// CHECK: %[[CND_CH:.*]] = grt.chain.create (%[[ENTRY_CH]], %[[IF_OUT]]#0)

// CHECK: grt.if
// CHECK: %[[CH1:.*]] = grt.chain.create (%[[IF_OUT]]#0, %[[IF_OUT]]#0, %[[IF_OUT]]#0)
// CHECK: %[[TRUE_CH1:.*]] = mogg.generic.compiled.region (%[[CH1]]
// } else {
// CHECK: %[[CH1:.*]] = grt.chain.create (%[[IF_OUT]]#0, %[[CND_CH]])
// CHECK: %[[FALSE_CH1:.*]] = mogg.generic.compiled.region (%[[CH1]]

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

// CHECK-LABEL: mgp.model @implicit_chaining
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[WEIGHTS:.*]]: !mgp.opaque<"weights_registry">, %[[CPU:.*]]: !mgp.device_context<<"cpu", 0>>

mo.graph @implicit_chaining(%init: !mo.chain, %cond: !mo.tensor<[], bool>, %arg0: !mo.buffer<[], ui32>) -> !mo.tensor<[10, 10], bf16> {

  // CHECK: %[[WEIGHT_NAME:.*]] = mgp.string.create "super_duper_weights"
  // CHECK: %[[CST:.*]] = mgp.buffer.constant.external %[[CHAIN]], %[[WEIGHTS]][%[[WEIGHT_NAME]],
  // CHECK: %[[GUARD:.*]] = grt.chain.create (%[[CHAIN]], %[[CHAIN]])

  // CHECK: %[[ALLOC:.*]] = mgp.buffer.alloc %[[GUARD]], %{{.*}}, %[[CPU]], align {{.*}} : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %constant = mo.constant.external {
    align = 1 : ui64,
    device = #M.device_ref<"cpu", 0>,
    isPlaceholder = true,
    name = "super_duper_weights"
  } : !mo.tensor<[10, 10], bf16>

  %val = mo.chain.create(%init) : !mo.chain

  // The mo.if operation will be implicitly chained during mo-to-primitives.
  // We want to make use that chaining does not interfere with the
  // mo.constant.external being hoisted to the init section.
  %unused = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.chain {
    %res = mo.chain.create(%val) : !mo.chain
    mo.yield %res : !mo.chain
  } else {
    mo.yield %init : !mo.chain
  }

  // %{{.*}} = mogg.generic.compiled.region (%{{.*}}, %[[ALLOC]], %[[CST]])
  %kernel_out = mogg.kernel (%constant) : (%arg18: !mo.tensor<[10, 10], bf16>)  -> !mo.tensor<[10, 10], bf16> {
    %out = mogg.output_placeholders : !mo.tensor<[10, 10], bf16>
    mogg.output %out : !mo.tensor<[10, 10], bf16>
  } {allocIndices = [], device = #M.device_ref<"cpu", 0>, isAsync = true, summary = ""}

  mo.output %kernel_out : !mo.tensor<[10, 10], bf16>
}

// CHECK-LABEL: mgp.model @if_output_params
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[WEIGHTS:.*]]: !mgp.opaque<"weights_registry">, %[[CPU:.*]]: !mgp.device_context<<"cpu", 0>>, %[[COND_TENSOR:.*]]: !mgp.tensor<bool>)
// CHECK:   %[[BOOL:.*]] = mgp.buffer.to_bool %[[CHAIN]]
// CHECK:   %[[IF_OUT:.*]]:3 = grt.if %[[BOOL]] (%[[CHAIN]]: !mef.chain) -> (!mef.chain, si64, si64) {
// CHECK:     %[[P1:.*]] = kgen.param.constant: si64 = <10>
// CHECK:     %[[P2:.*]] = kgen.param.constant: si64 = <20>
// CHECK:     mef.output %{{.*}}, %[[P1]], %[[P2]] : !mef.chain, si64, si64
// CHECK:   } else {
// CHECK:     %[[P3:.*]] = kgen.param.constant: si64 = <30>
// CHECK:     %[[P4:.*]] = kgen.param.constant: si64 = <40>
// CHECK:     mef.output %{{.*}}, %[[P3]], %[[P4]] : !mef.chain, si64, si64
// CHECK:   }
// CHECK:   %[[M:.*]] = index.casts %[[IF_OUT]]#1 : si64 to index
// CHECK:   %[[N:.*]] = index.casts %[[IF_OUT]]#2 : si64 to index
// CHECK:   %[[SPEC:.*]] = mgp.tensor_spec.create %[[M]], %[[N]] : !mgp.tensor_spec<?x?xf32>
// CHECK:   %[[TENSOR:.*]] = mgp.tensor.create %[[CHAIN]], %{{.*}}, %[[SPEC]] : !mgp.tensor<?x?xf32>
// CHECK:   mef.output %[[IF_OUT]]#0, %[[TENSOR]] : !mef.chain, !mgp.tensor<?x?xf32>
mo.graph @if_output_params(%cond: !mo.tensor<[], bool>) -> !mo.tensor<[M, N], f32> {
  mo.if<() -> M, N>(%cond : !mo.tensor<[], bool>) {
    kgen.param.declare L: si64 = <10>
    kgen.param.declare K: si64 = <20>
    mo.yield<L, K>
  } else {
    kgen.param.declare A: si64 = <30>
    kgen.param.declare B: si64 = <40>
    mo.yield<A, B>
  }
  %result = mo.constant {value = #M.dense_array<1.0> : tensor<1xf32>} : !mo.tensor<[M, N], f32>
  mo.output %result : !mo.tensor<[M, N], f32>
}
