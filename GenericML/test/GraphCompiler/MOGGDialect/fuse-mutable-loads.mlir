// RUN: mo-opt --mo-to-mogg --canonicalize --fuse-mutable-loads %s | FileCheck %s

// CHECK-LABEL: mo.graph @single_add_buffer_lhs(
// CHECK-SAME: [[buffer:%.*]]: !mo.buffer<[10, 10], f32>, [[rhs:%.*]]: !mo.tensor<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create([[buffer]])
// CHECK-NOT: mo.mutable.store
// CHECK: [[out:%.*]]:2 = mogg.kernel ([[buffer]], [[rhs]], [[ch0]])
// CHECK: [[cr:%.*]] = "mogg.create_ready_chain"() : () -> !mo.chain
// CHECK: [[tr:%.*]] = mogg.output_placeholders
// CHECK: mogg.output [[tr]], [[cr]]
// CHECK: inputFusionInterface = ["", "{{.*}}", ""]{{.*}}outputFusionInterface = ["{{.*}}", ""]
// CHECK: mogg.kernel ([[buffer]], [[out]]#0, [[out]]#1)
// CHECK: summary = "mo.mutable.store
mo.graph @single_add_buffer_lhs(%buffer: !mo.buffer<[10, 10], f32>, %rhs: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create(%buffer) : !mo.buffer<[10, 10], f32>
  %outChain, %outTensor = mo.mutable.load(%0, %buffer) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %1 = mo.add(%outTensor, %rhs) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%outChain, %buffer, %1) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain
  mo.output
}

// CHECK-LABEL: mo.graph @single_add_buffer_rhs(
// CHECK-SAME: [[buffer:%.*]]: !mo.buffer<[10, 10], f32>, [[rhs:%.*]]: !mo.tensor<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create([[buffer]])
// CHECK-NOT: mo.mutable.store
// CHECK: [[out:%.*]]:2 = mogg.kernel ([[rhs]], [[buffer]], [[ch0]])
// CHECK: [[cr:%.*]] = "mogg.create_ready_chain"() : () -> !mo.chain
// CHECK: [[tr:%.*]] = mogg.output_placeholders
// CHECK: mogg.output [[tr]], [[cr]]
// CHECK: inputFusionInterface = ["{{.*}}", "", ""]{{.*}}outputFusionInterface = ["{{.*}}", ""]
// CHECK: mogg.kernel ([[buffer]], [[out]]#0, [[out]]#1)
// CHECK: summary = "mo.mutable.store
mo.graph @single_add_buffer_rhs(%buffer: !mo.buffer<[10, 10], f32>, %rhs: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create(%buffer) : !mo.buffer<[10, 10], f32>
  %outChain, %outTensor = mo.mutable.load(%0, %buffer) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %1 = mo.add(%rhs, %outTensor) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%outChain, %buffer, %1) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain
  mo.output
}

// CHECK-LABEL: mo.graph @multiple_chain_users(
// CHECK-SAME: [[buffer:%.*]]: !mo.buffer<[10, 10], f32>, [[rhs:%.*]]: !mo.tensor<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create([[buffer]])
// CHECK: [[out:%.*]]:2 = mogg.kernel ([[rhs]], [[buffer]], [[ch0]])
// CHECK: mogg.kernel ([[buffer]], [[out]]#0, [[out]]#1)
// CHECK: summary = "mo.mutable.store
mo.graph @multiple_chain_users(%buffer: !mo.buffer<[10, 10], f32>, %rhs: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create(%buffer) : !mo.buffer<[10, 10], f32>
  mo.chain.create(%0) : !mo.chain
  %outChain, %outTensor = mo.mutable.load(%0, %buffer) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %1 = mo.add(%rhs, %outTensor) : !mo.tensor<[10, 10], f32>
  mo.chain.create(%outChain) : !mo.chain
  %2 = mo.mutable.store(%outChain, %buffer, %1) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain
  mo.output
}

// CHECK-LABEL: mo.graph @same_kernel_multiple_fusions(
// CHECK-SAME: [[buf1:%.*]]: !mo.buffer<[10, 10], f32>, [[buf2:%.*]]: !mo.buffer<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: mogg.kernel ([[buf1]], [[buf2]], [[ch0]])
mo.graph @same_kernel_multiple_fusions(%buf1: !mo.buffer<[10, 10], f32>, %buf2: !mo.buffer<[10, 10], f32>) {

  %ch0 = mo.chain.create()

  %ch1, %vbuf = mo.mutable.load(%ch0, %buf1) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %ch2, %vbuf2 = mo.mutable.load(%ch0, %buf2) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %res = mo.add(%vbuf, %vbuf2) : !mo.tensor<[10, 10], f32>
  %chfinal = mo.mutable.store(%ch1, %buf2, %res) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> (!mo.chain)

  mo.output
}

// CHECK-LABEL: multiple_load_fusion
// CHECK-SAME: [[buf:%.*]]: !mo.buffer
// [[ch0:%.*]] = mo.chain.create()
// CHECK-NOT: mo.mutable.load
// CHECK-NOT: mo.mutable.load
// CHECK-NOT: mo.mutable.load
// CHECK: mogg.kernel ([[buf]], [[ch0]])
// CHECK: mogg.kernel ([[buf]], [[ch0]])
// CHECK: mogg.kernel ([[buf]], [[buf]], [[ch0]])
mo.graph @multiple_load_fusion(%buf: !mo.buffer<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
{
  %0 = mo.chain.create()
  %outChain, %outTensor = mo.mutable.load(%0, %buf) : (!mo.chain, !mo.buffer<[2, 2], f32>) -> (!mo.chain, !mo.tensor<[2, 2], f32>)
  %outChain_0, %outTensor_1 = mo.mutable.load(%outChain, %buf) : (!mo.chain, !mo.buffer<[2, 2], f32>) -> (!mo.chain, !mo.tensor<[2, 2], f32>)
  %outChain_2, %outTensor_3 = mo.mutable.load(%outChain_0, %buf) : (!mo.chain, !mo.buffer<[2, 2], f32>) -> (!mo.chain, !mo.tensor<[2, 2], f32>)
  %outChain_3, %outTensor_4 = mo.mutable.load(%outChain_2, %buf) : (!mo.chain, !mo.buffer<[2, 2], f32>) -> (!mo.chain, !mo.tensor<[2, 2], f32>)

  %1 = mo.negative(%outTensor) : !mo.tensor<[2, 2], f32>
  %2 = mo.abs(%outTensor_1) : !mo.tensor<[2, 2], f32>
  %3 = mo.add(%outTensor_3, %outTensor_3) : !mo.tensor<[2, 2], f32>

  mo.output %1, %2, %3 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: fusion_with_slice
// CHECK-SAME: [[buf:%.*]]: !mo.buffer
// [[ch0:%.*]] = mo.chain.create()
// CHECK-NOT: mo.mutable.load
// CHECK: mogg.kernel ([[buf]]
mo.graph @fusion_with_slice(%arg0: !mo.buffer<[10, 10], f32>) -> !mo.tensor<[3, 3], f32> {
  %0 = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %1 = mo.constant {value = #M.dense_array<8, 8> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %2 = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %3 = mo.chain.create()
  %outChain, %outTensor = mo.mutable.load(%3, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %4 = mo.slice(%outTensor, %2, %1, %0) : (!mo.tensor<[10, 10], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 3], f32>
  %5 = mo.negative(%4) : !mo.tensor<[3, 3], f32>
  mo.output %5 : !mo.tensor<[3, 3], f32>
}

// CHECK-LABEL: dont_merge_loads_across_blocks
// CHECK: mo.mutable.load
// CHECK: mo.if
mo.graph @dont_merge_loads_across_blocks(%arg0: !mo.buffer<[2, 2], f32>, %arg1: !mo.tensor<[], bool>) -> !mo.buffer<[2, 2], f32> {
  %0 = mo.chain.create()
  // COM: we expect the load below to be preserved as we don't merge descendants from another block.
  %outChain, %outTensor = mo.mutable.load(%0, %arg0) : (!mo.chain, !mo.buffer<[2, 2], f32>) -> (!mo.chain, !mo.tensor<[2, 2], f32>)
  mo.if(%arg1 : !mo.tensor<[], bool>) {
    %outChain_0, %outTensor_1 = mo.mutable.load(%outChain, %arg0) : (!mo.chain, !mo.buffer<[2, 2], f32>) -> (!mo.chain, !mo.tensor<[2, 2], f32>)
    %1 = mo.negative(%outTensor_1) : !mo.tensor<[2, 2], f32>
    %2 = mo.mutable.store(%outChain_0, %arg0, %1) : (!mo.chain, !mo.buffer<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.chain
  } else {
  }
  mo.output %arg0 : !mo.buffer<[2, 2], f32>
}


// CHECK-LABEL: dont_parallelize_loads_across_blocks
// CHECK: {{.*}}, [[tensor:%.*]] = mo.mutable.load
// CHECK: mo.if
// CHECK: mogg.kernel ([[tensor]], [[tensor]])
// CHECK: else
// CHECK: mogg.kernel ([[tensor]])
mo.graph @dont_parallelize_loads_across_blocks(%arg0: !mo.buffer<[2, 2], f32>, %arg1: !mo.tensor<[], bool>) -> !mo.buffer<[2, 2], f32> {
  %1 = mo.chain.create()
  %outChain, %outTensor = mo.mutable.load(%1, %arg0) : (!mo.chain, !mo.buffer<[2, 2], f32>) -> (!mo.chain, !mo.tensor<[2, 2], f32>)
  %2 = mo.if(%arg1 : !mo.tensor<[], bool>) -> !mo.chain {
    %3 = mo.add(%outTensor, %outTensor) : !mo.tensor<[2, 2], f32>
    %4 = mo.mutable.store(%outChain, %arg0, %3) : (!mo.chain, !mo.buffer<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.chain
    mo.yield %4 : !mo.chain
  } else {
    %3 = mo.negative(%outTensor) : !mo.tensor<[2, 2], f32>
    %4 = mo.mutable.store(%outChain, %arg0, %3) : (!mo.chain, !mo.buffer<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.chain
    mo.yield %4 : !mo.chain
  }
  mo.output %arg0 : !mo.buffer<[2, 2], f32>
}
