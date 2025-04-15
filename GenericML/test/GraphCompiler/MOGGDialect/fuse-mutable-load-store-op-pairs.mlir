// RUN: mo-opt --mo-to-mogg --fuse-mutable-loads --fuse-mutable-stores %s | FileCheck %s

// CHECK-LABEL: mo.graph @single_add_buffer_rhs(
// CHECK-SAME: [[buffer:%.*]]: !mo.buffer<[10, 10], f32>, [[lhs:%.*]]: !mo.tensor<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create([[buffer]])
// CHECK-NOT: mo.mutable.load
// CHECK: [[ch2:%.*]] = mogg.kernel ([[lhs]], [[buffer]], [[ch0]])
// CHECK: [[kChain:%.*]] = "mogg.create_ready_chain"() : () -> !mo.chain
// CHECK: mogg.output [[kChain]] : !mo.chain
// CHECK: hasReadEffect = true, hasWriteEffect = true
// CHECK-SAME: inputFusionInterface = ["{{.*}}", "", ""]
// CHECK-SAME: outputFusionInterface = [""]
// CHECK-SAME: summary = "inplace_mo.add
// CHECK-NOT: mo.mutable.store
// CHECK: mo.mutable.load([[ch2]], [[buffer]])
mo.graph @single_add_buffer_rhs(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[10, 10], f32> {
  %0 = mo.chain.create(%arg0) : !mo.buffer<[10, 10], f32>

  %outChain, %outTensor = mo.mutable.load(%0, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %1 = mo.add(%arg1, %outTensor) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%outChain, %arg0, %1) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain

  %outChain_0, %outTensor_1 = mo.mutable.load(%2, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  mo.output %outTensor_1 : !mo.tensor<[10, 10], f32>
}

// CHECK-LABEL: mo.graph @sequenced_add(
// CHECK-SAME: [[buffer:%.*]]: !mo.buffer<[10, 10], f32>, [[rhs:%.*]]: !mo.tensor<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create([[buffer]])
// CHECK-NOT: mo.mutable.load
// CHECK: [[ch2:%.*]] = mogg.kernel ([[buffer]], [[rhs]], [[ch0]])
// CHECK: [[kChain:%.*]] = "mogg.create_ready_chain"() : () -> !mo.chain
// CHECK: mogg.output [[kChain]] : !mo.chain
// CHECK: hasReadEffect = true, hasWriteEffect = true
// CHECK-SAME: inputFusionInterface = ["", "{{.*}}", ""]
// CHECK-SAME: outputFusionInterface = [""]
// CHECK-SAME: summary = "inplace_mo.add
// CHECK-NOT: mo.mutable.store
// CHECK-NOT: mo.mutable.load
// CHECK: [[ch3:%.*]] = mogg.kernel ([[buffer]], [[rhs]], [[ch2]])
// CHECK: [[kChain:%.*]] = "mogg.create_ready_chain"() : () -> !mo.chain
// CHECK: mogg.output [[kChain]] : !mo.chain
// CHECK: hasReadEffect = true, hasWriteEffect = true
// CHECK-SAME: inputFusionInterface = ["", "{{.*}}", ""]
// CHECK-SAME: outputFusionInterface = [""]
// CHECK-SAME: summary = "inplace_mo.add
// CHECK-NOT: mo.mutable.store
// CHECK: mo.mutable.load([[ch3]], [[buffer]])
mo.graph @sequenced_add(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[10, 10], f32> {
  %0 = mo.chain.create(%arg0) : !mo.buffer<[10, 10], f32>

  %outChain, %outTensor = mo.mutable.load(%0, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %1 = mo.add(%outTensor, %arg1) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%outChain, %arg0, %1) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain

  %outChain_0, %outTensor_1 = mo.mutable.load(%2, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %3 = mo.add(%outTensor_1, %arg1) : !mo.tensor<[10, 10], f32>
  %4 = mo.mutable.store(%outChain_0, %arg0, %3) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain

  %outChain_2, %outTensor_3 = mo.mutable.load(%4, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)

  mo.output %outTensor_3 : !mo.tensor<[10, 10], f32>
}

// In this particular case mutable op fusion is not valid on the first mo.add as its result has multiple consumers
// CHECK-LABEL: mo.graph @no_valid_fusion(
// CHECK: mogg.kernel
// CHECK: summary = "mo.add
// CHECK: mo.mutable.store
// CHECK: mogg.kernel
// CHECK: summary = "mo.sub
// CHECK: mo.mutable.store
mo.graph @no_valid_fusion(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[10, 10], f32> {
  %0 = mo.chain.create(%arg0) : !mo.buffer<[10, 10], f32>

  %outChain, %outTensor = mo.mutable.load(%0, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %1 = mo.add(%outTensor, %arg1) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%outChain, %arg0, %1) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain

  %3 = mo.sub(%1, %arg1) : !mo.tensor<[10, 10], f32>
  %4 = mo.mutable.store(%2, %arg0, %3) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain

  %outChain_2, %outTensor_3 = mo.mutable.load(%4, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)

  mo.output %outTensor_3 : !mo.tensor<[10, 10], f32>
}
