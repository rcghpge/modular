// RUN: EMIT_MOJO=1 mo-opt --mo-to-mogg="kernel-library-paths=%kernel_api_test_pkg" --canonicalize --fuse-mutable-loads %s | FileCheck %s

// CHECK-LABEL: mo.graph @single_add_buffer_lhs(
// CHECK-SAME: [[buffer:%.*]]: !mo.buffer<[10, 10], f32>, [[rhs:%.*]]: !mo.tensor<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create([[buffer]])
// CHECK-NOT: mo.mutable.store
// CHECK: [[out:%.*]]:2 = mogg.experimental.kernel([[buffer]]: !mo.buffer<[10, 10], f32>, [[rhs]]: !mo.tensor<[10, 10], f32>) chain([[ch0]])
// CHECK: [[tr:%.*]] = mogg.output_placeholders
// CHECK: mogg.experimental.kernel([[buffer]]: !mo.buffer<[10, 10], f32>, [[out]]#0: !mo.tensor<[10, 10], f32>) chain([[out]]#1)
mo.graph @single_add_buffer_lhs(%buffer: !mo.buffer<[10, 10], f32>, %rhs: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create(%buffer) : !mo.buffer<[10, 10], f32>
  %outTensor, %outChain = mo.mutable.load(%buffer, %0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %1 = mo.add(%outTensor, %rhs) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%buffer, %1, %outChain) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain
  mo.output
}

// CHECK-LABEL: mo.graph @single_add_buffer_rhs(
// CHECK-SAME: [[buffer:%.*]]: !mo.buffer<[10, 10], f32>, [[rhs:%.*]]: !mo.tensor<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create([[buffer]])
// CHECK-NOT: mo.mutable.store
// CHECK: [[out:%.*]]:2 = mogg.experimental.kernel([[rhs]]: !mo.tensor<[10, 10], f32>, [[buffer]]: !mo.buffer<[10, 10], f32>) chain([[ch0]])
// CHECK: [[tr:%.*]] = mogg.output_placeholders
// CHECK: mogg.experimental.kernel([[buffer]]: !mo.buffer<[10, 10], f32>, [[out]]#0: !mo.tensor<[10, 10], f32>) chain([[out]]#1)
// CHECK: summary = "mo.mutable.store
mo.graph @single_add_buffer_rhs(%buffer: !mo.buffer<[10, 10], f32>, %rhs: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create(%buffer) : !mo.buffer<[10, 10], f32>
  %outTensor, %outChain = mo.mutable.load(%buffer, %0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %1 = mo.add(%rhs, %outTensor) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%buffer, %1, %outChain) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain
  mo.output
}

// CHECK-LABEL: mo.graph @multiple_chain_users(
// CHECK-SAME: [[buffer:%.*]]: !mo.buffer<[10, 10], f32>, [[rhs:%.*]]: !mo.tensor<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create([[buffer]])
// CHECK: [[out:%.*]]:2 = mogg.experimental.kernel([[rhs]]: !mo.tensor<[10, 10], f32>, [[buffer]]: !mo.buffer<[10, 10], f32>) chain([[ch0]])
// CHECK: mogg.experimental.kernel([[buffer]]: !mo.buffer<[10, 10], f32>, [[out]]#0: !mo.tensor<[10, 10], f32>) chain([[out]]#1)
// CHECK: summary = "mo.mutable.store
mo.graph @multiple_chain_users(%buffer: !mo.buffer<[10, 10], f32>, %rhs: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create(%buffer) : !mo.buffer<[10, 10], f32>
  mo.chain.create(%0) : !mo.chain
  %outTensor, %outChain = mo.mutable.load(%buffer, %0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %1 = mo.add(%rhs, %outTensor) : !mo.tensor<[10, 10], f32>
  mo.chain.create(%outChain) : !mo.chain
  %2 = mo.mutable.store(%buffer, %1, %outChain) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain
  mo.output
}

// CHECK-LABEL: mo.graph @same_kernel_multiple_fusions(
// CHECK-SAME: [[buf1:%.*]]: !mo.buffer<[10, 10], f32>, [[buf2:%.*]]: !mo.buffer<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: mogg.experimental.kernel([[buf1]]: !mo.buffer<[10, 10], f32>, [[buf2]]: !mo.buffer<[10, 10], f32>) chain([[ch0]])
mo.graph @same_kernel_multiple_fusions(%buf1: !mo.buffer<[10, 10], f32>, %buf2: !mo.buffer<[10, 10], f32>) {

  %ch0 = mo.chain.create()

  %vbuf, %ch1 = mo.mutable.load(%buf1, %ch0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %vbuf2, %ch2 = mo.mutable.load(%buf2, %ch0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %res = mo.add(%vbuf, %vbuf2) : !mo.tensor<[10, 10], f32>
  %chfinal = mo.mutable.store(%buf2, %res, %ch1) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain

  mo.output
}

// CHECK-LABEL: multiple_load_fusion
// CHECK-SAME: [[buf:%.*]]: !mo.buffer
// [[ch0:%.*]] = mo.chain.create()
// CHECK-NOT: mo.mutable.load
// CHECK-NOT: mo.mutable.load
// CHECK-NOT: mo.mutable.load
// CHECK: mogg.experimental.kernel([[buf]]: !mo.buffer<[2, 2], f32>) chain([[ch0]])
// CHECK: mogg.experimental.kernel([[buf]]: !mo.buffer<[2, 2], f32>) chain([[ch0]])
// CHECK: mogg.experimental.kernel([[buf]] as {{.*}}: !mo.buffer<[2, 2], f32>, [[buf]] as {{.*}}: !mo.buffer<[2, 2], f32>) chain([[ch0]])
mo.graph @multiple_load_fusion(%buf: !mo.buffer<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
{
  %0 = mo.chain.create()
  %outTensor, %outChain = mo.mutable.load(%buf, %0) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %outTensor_1, %outChain_0 = mo.mutable.load(%buf, %outChain) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %outTensor_3, %outChain_2 = mo.mutable.load(%buf, %outChain_0) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %outTensor_4, %outChain_3 = mo.mutable.load(%buf, %outChain_2) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)

  %1 = mo.negative(%outTensor) : !mo.tensor<[2, 2], f32>
  %2 = mo.abs(%outTensor_1) : !mo.tensor<[2, 2], f32>
  %3 = mo.add(%outTensor_3, %outTensor_3) : !mo.tensor<[2, 2], f32>

  mo.output %1, %2, %3 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: fusion_with_slice
// CHECK-SAME: [[buf:%.*]]: !mo.buffer
// [[ch0:%.*]] = mo.chain.create()
// CHECK-NOT: mo.mutable.load
// CHECK: mogg.experimental.kernel([[buf]]
mo.graph @fusion_with_slice(%arg0: !mo.buffer<[10, 10], f32>) -> !mo.tensor<[3, 3], f32> {
  %0 = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %1 = mo.constant {value = #M.dense_array<8, 8> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %2 = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %3 = mo.chain.create()
  %outTensor, %outChain = mo.mutable.load(%arg0, %3) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
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
  %outTensor, %outChain = mo.mutable.load(%arg0, %0) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  mo.if(%arg1 : !mo.tensor<[], bool>) {
    %outTensor_1, %outChain_0 = mo.mutable.load(%arg0, %outChain) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
    %1 = mo.negative(%outTensor_1) : !mo.tensor<[2, 2], f32>
    %2 = mo.mutable.store(%arg0, %1, %outChain_0) : (!mo.buffer<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.chain) -> !mo.chain
  } else {
  }
  mo.output %arg0 : !mo.buffer<[2, 2], f32>
}


// CHECK-LABEL: dont_parallelize_loads_across_blocks
// CHECK: [[tensor:%.*]], {{.*}} = mo.mutable.load
// CHECK: mo.if
// CHECK: mogg.experimental.kernel([[tensor]] as {{.*}}: !mo.tensor<[2, 2], f32>, [[tensor]] as {{.*}}: !mo.tensor<[2, 2], f32>)
// CHECK: else
// CHECK: mogg.experimental.kernel([[tensor]]
mo.graph @dont_parallelize_loads_across_blocks(%arg0: !mo.buffer<[2, 2], f32>, %arg1: !mo.tensor<[], bool>) -> !mo.buffer<[2, 2], f32> {
  %1 = mo.chain.create()
  %outTensor, %outChain = mo.mutable.load(%arg0, %1) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %2 = mo.if(%arg1 : !mo.tensor<[], bool>) -> !mo.chain {
    %3 = mo.add(%outTensor, %outTensor) : !mo.tensor<[2, 2], f32>
    %4 = mo.mutable.store(%arg0, %3, %outChain) : (!mo.buffer<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.chain) -> !mo.chain
    mo.yield %4 : !mo.chain
  } else {
    %3 = mo.negative(%outTensor) : !mo.tensor<[2, 2], f32>
    %4 = mo.mutable.store(%arg0, %3, %outChain) : (!mo.buffer<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.chain) -> !mo.chain
    mo.yield %4 : !mo.chain
  }
  mo.output %arg0 : !mo.buffer<[2, 2], f32>
}

// CHECK-LABEL: mo.graph @block_fusion_non_elementwise_inplace_kernel
// CHECK-SAME: [[buffer:%.*]]: !mo.buffer<[2, 2], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: [[outTensor:%.*]], [[outChain:%.*]] = mo.mutable.load([[buffer]], [[ch0]])
// CHECK: mogg.experimental.kernel([[buffer]]: !mo.buffer<[2, 2], f32>, [[outTensor]]: !mo.tensor<[2, 2], f32>) chain([[outChain]])
mo.graph @block_fusion_non_elementwise_inplace_kernel(%buffer: !mo.buffer<[2, 2], f32>) {
  %0 = mo.chain.create()
  %outTensor, %outChain = mo.mutable.load(%buffer, %0) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %1 = mo.custom {symbol = "inplace_no_out_func", device = #M.device_ref<"cpu", 0>}(%buffer, %outTensor, %outChain) : (!mo.buffer<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.chain) -> !mo.chain
  mo.output
}
