// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels --map-fuse-mutable-loads | FileCheck %s

// CHECK-LABEL: mo.graph @single_add_buffer_lhs(
// CHECK-SAME:  %[[BUF:.*]]: !mo.buffer<[10, 10], f32>, %[[RHS:.*]]: !mo.tensor<[10, 10], f32>
// CHECK:       %[[CH0:.*]] = mo.chain.create(%[[BUF]])
// CHECK-NOT:   mo.mutable.load
// CHECK:       map.iter.foreach<[10, 10]>
// CHECK-NEXT:  inputs  (%[[BUF]] as %[[BUFARG:.*]]: !mo.buffer<[10, 10], f32>, %[[RHS]] as {{.*}}: !mo.tensor<[10, 10], f32>, %[[BUF]] as {{.*}}: !mo.buffer<[10, 10], f32>)
// CHECK:       chain(%[[CH0]])
// CHECK:       %{{.*}} = map.load %[[BUFARG]][%{{.*}}] : !mo.buffer<[10, 10], f32>
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_mutable_load<[#mogg.fusion_leaf<"mo.mutable.load"{{.*}}, #mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.mutable.store"{{.*}}]>]>>
mo.graph @single_add_buffer_lhs(%buffer: !mo.buffer<[10, 10], f32>, %rhs: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create(%buffer) : !mo.buffer<[10, 10], f32>
  %outTensor, %outChain = mo.mutable.load(%buffer, %0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %1 = mo.add(%outTensor, %rhs) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%buffer, %1, %outChain) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain
  mo.output
}

// CHECK-LABEL: mo.graph @single_add_buffer_rhs(
// CHECK-SAME:  %[[BUF2:.*]]: !mo.buffer<[10, 10], f32>, %[[RHS2:.*]]: !mo.tensor<[10, 10], f32>
// CHECK:       %[[CH20:.*]] = mo.chain.create(%[[BUF2]])
// CHECK-NOT:   mo.mutable.load
// CHECK:       map.iter.foreach<[10, 10]>
// CHECK-NEXT:  inputs  (%[[RHS2]] as {{.*}}: !mo.tensor<[10, 10], f32>, %[[BUF2]] as %[[BUFARG2:.*]]: !mo.buffer<[10, 10], f32>, %[[BUF2]] as {{.*}}: !mo.buffer<[10, 10], f32>)
// CHECK:       chain(%[[CH20]])
// CHECK:       %{{.*}} = map.load %[[BUFARG2]][%{{.*}}] : !mo.buffer<[10, 10], f32>
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_mutable_load<[#mogg.fusion_leaf<"mo.mutable.load"{{.*}}, #mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.mutable.store"{{.*}}]>]>>
mo.graph @single_add_buffer_rhs(%buffer: !mo.buffer<[10, 10], f32>, %rhs: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create(%buffer) : !mo.buffer<[10, 10], f32>
  %outTensor, %outChain = mo.mutable.load(%buffer, %0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %1 = mo.add(%rhs, %outTensor) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%buffer, %1, %outChain) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain
  mo.output
}

// A chain.create with more than one use (the extra mo.chain.create(%0)/(%outChain)
// below) keeps the add and the store from pre-fusing into one map-fuse-kernels
// kernel (mirrors MOGG's checkMutableKernelFusible); the load then fuses
// into the (still separate) add kernel via fuseLoadBufferAndChains, and the
// store remains its own kernel chained off the add kernel's output chain.
// CHECK-LABEL: mo.graph @multiple_chain_users(
// CHECK-SAME:  %[[BUF3:.*]]: !mo.buffer<[10, 10], f32>, %[[RHS3:.*]]: !mo.tensor<[10, 10], f32>
// CHECK:       %[[CH30:.*]] = mo.chain.create(%[[BUF3]])
// CHECK:       %[[OUT3:.*]]:2 = map.iter.foreach<[10, 10]>
// CHECK-NEXT:  inputs  (%[[RHS3]]: !mo.tensor<[10, 10], f32>, %[[BUF3]]: !mo.buffer<[10, 10], f32>)
// CHECK:       chain(%[[CH30]])
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       map.iter.foreach<[10, 10]>
// CHECK-NEXT:  inputs  (%[[OUT3]]#0: !mo.tensor<[10, 10], f32>, %[[BUF3]]: !mo.buffer<[10, 10], f32>)
// CHECK:       chain(%[[OUT3]]#1)
// CHECK:       map.call.elementwise @{{.*}}MutableStore
mo.graph @multiple_chain_users(%buffer: !mo.buffer<[10, 10], f32>, %rhs: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create(%buffer) : !mo.buffer<[10, 10], f32>
  mo.chain.create(%0) : !mo.chain
  %outTensor, %outChain = mo.mutable.load(%buffer, %0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %1 = mo.add(%rhs, %outTensor) : !mo.tensor<[10, 10], f32>
  mo.chain.create(%outChain) : !mo.chain
  %2 = mo.mutable.store(%buffer, %1, %outChain) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain
  mo.output
}

// Both mutable.loads fuse directly into the single already-fused
// (Add + MutableStore) kernel via fuseLoadBufferAndChains.
// CHECK-LABEL: mo.graph @same_kernel_multiple_fusions(
// CHECK-SAME:  %[[BUF4:.*]]: !mo.buffer<[10, 10], f32>, %[[BUF5:.*]]: !mo.buffer<[10, 10], f32>
// CHECK:       %[[CH40:.*]] = mo.chain.create()
// CHECK-NOT:   mo.mutable.load
// CHECK:       map.iter.foreach<[10, 10]>
// CHECK-NEXT:  inputs  (%[[BUF4]] as {{.*}}: !mo.buffer<[10, 10], f32>, %[[BUF5]] as {{.*}}: !mo.buffer<[10, 10], f32>, %[[BUF5]] as {{.*}}: !mo.buffer<[10, 10], f32>)
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_mutable_load<[#mogg.fusion_leaf<"mo.mutable.load"{{.*}}, #mogg.fusion_leaf<"mo.mutable.load"{{.*}}, #mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.mutable.store"{{.*}}]>]>>
mo.graph @same_kernel_multiple_fusions(%buf1: !mo.buffer<[10, 10], f32>, %buf2: !mo.buffer<[10, 10], f32>) {

  %ch0 = mo.chain.create()

  %vbuf, %ch1 = mo.mutable.load(%buf1, %ch0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %vbuf2, %ch2 = mo.mutable.load(%buf2, %ch0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %res = mo.add(%vbuf, %vbuf2) : !mo.tensor<[10, 10], f32>
  %chfinal = mo.mutable.store(%buf2, %res, %ch1) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain

  mo.output
}

// CHECK-LABEL: mo.graph @multiple_load_fusion(
// CHECK-SAME:  %[[BUF6:.*]]: !mo.buffer
// CHECK:       %[[CH60:.*]] = mo.chain.create()
// CHECK-NOT:   mo.mutable.load
// CHECK-NOT:   mo.mutable.load
// CHECK-NOT:   mo.mutable.load
// CHECK:       %[[K1:.*]]:2 = map.iter.foreach<[2, 2]>
// CHECK-NEXT:  inputs  (%[[BUF6]]: !mo.buffer<[2, 2], f32>)
// CHECK:       chain(%[[CH60]])
// CHECK:       map.call.elementwise @{{.*}}.Negative
// CHECK:       %[[K2:.*]]:2 = map.iter.foreach<[2, 2]>
// CHECK-NEXT:  inputs  (%[[BUF6]]: !mo.buffer<[2, 2], f32>)
// CHECK:       chain(%[[K1]]#1)
// CHECK:       map.call.elementwise @{{.*}}.Abs
// CHECK:       map.iter.foreach<[2, 2]>
// CHECK-NEXT:  inputs  (%[[BUF6]] as {{.*}}: !mo.buffer<[2, 2], f32>, %[[BUF6]] as {{.*}}: !mo.buffer<[2, 2], f32>)
// CHECK:       chain(%[[K2]]#1)
// CHECK:       map.call.elementwise @{{.*}}.Add
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

// The top-level load's tensor result has no uses of its own in this block (its
// sole use is the inner load's chain input, across a block boundary), so it is
// never a fusion candidate and survives untouched; the nested load fuses fully
// within its own if-block.
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

// The top-level load's tensor result has two uses, one per if/else branch (two
// different blocks), so ParallelizeLoadsWithMultipleUsers declines to split it
// (cross-block uses are unsupported, GEX-1203) and it is never a fusion
// candidate either; each branch instead reads the surviving tensor directly.
// CHECK-LABEL: dont_parallelize_loads_across_blocks
// CHECK: %[[TENSOR:.*]], {{.*}} = mo.mutable.load
// CHECK: mo.if
// CHECK: map.iter.foreach
// CHECK-NEXT: inputs  (%[[TENSOR]] as {{.*}}: !mo.tensor<[2, 2], f32>, %[[TENSOR]] as {{.*}}: !mo.tensor<[2, 2], f32>)
// CHECK: else
// CHECK: map.iter.foreach
// CHECK-NEXT: inputs  (%[[TENSOR]]: !mo.tensor<[2, 2], f32>)
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

// CHECK-LABEL: mo.graph @sequenced_add(
// CHECK:       %[[CH70:.*]] = mo.chain.create()
// CHECK:       map.iter.foreach<[10, 10]>
// CHECK:       chain(%[[CH70]])
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       map.call.elementwise @{{.*}}MutableStore
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_mutable_load<[#mogg.fusion_leaf<"mo.mutable.load"{{.*}}, #mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.mutable.store"{{.*}}]>]>>
// CHECK:       mo.mutable.load
mo.graph @sequenced_add(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[10, 10], f32> {
  %0 = mo.chain.create()

  %outTensor, %outChain = mo.mutable.load(%arg0, %0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %1 = mo.add(%outTensor, %arg1) : !mo.tensor<[10, 10], f32>
  %3 = mo.add(%1, %arg1) : !mo.tensor<[10, 10], f32>
  %4 = mo.mutable.store(%arg0, %3, %outChain) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain

  %outTensor_3, %outChain_2 = mo.mutable.load(%arg0, %4) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)

  mo.output %outTensor_3 : !mo.tensor<[10, 10], f32>
}

// Store fusion is not possible here as add's output has a use other than the
// store (the trailing mo.sub), so the load fuses into the add kernel only and
// the store remains a separate kernel chained off the add kernel's out-chain.
// CHECK-LABEL: mo.graph @no_valid_fusion(
// CHECK:       %[[CH80:.*]] = mo.chain.create()
// CHECK:       %[[K8:.*]]:2 = map.iter.foreach<[10, 10]>
// CHECK:       chain(%[[CH80]])
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_mutable_load<[#mogg.fusion_leaf<"mo.mutable.load"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>>
// CHECK:       map.iter.foreach<[10, 10]>
// CHECK:       chain(%[[K8]]#1)
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.mutable.store"{{.*}}>>
// CHECK:       map.iter.foreach<[10, 10]>
// CHECK-NEXT:  inputs  (%[[K8]]#0: !mo.tensor<[10, 10], f32>,
// CHECK:       map.call.elementwise @{{.*}}.Sub
mo.graph @no_valid_fusion(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[10, 10], f32> {
  %0 = mo.chain.create()

  %outTensor, %outChain = mo.mutable.load(%arg0, %0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %1 = mo.add(%outTensor, %arg1) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%arg0, %1, %outChain) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain

  %3 = mo.sub(%1, %arg1) : !mo.tensor<[10, 10], f32>

  mo.output %3 : !mo.tensor<[10, 10], f32>
}

// The store's inChain is %0 (the top-level chain.create), not the load's own
// out-chain -- the "wrong producer" -- so the load fuses into the add kernel
// but the store cannot fuse with it and remains its own kernel, also chained
// off %0.
// CHECK-LABEL: mo.graph @store_chain_comes_from_wrong_producer(
// CHECK:       %[[CH90:.*]] = mo.chain.create()
// CHECK:       %[[K9:.*]]:2 = map.iter.foreach<[10, 10]>
// CHECK:       chain(%[[CH90]])
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_mutable_load<[#mogg.fusion_leaf<"mo.mutable.load"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>>
// CHECK:       map.iter.foreach<[10, 10]>
// CHECK-NEXT:  inputs  (%[[K9]]#0: !mo.tensor<[10, 10], f32>,
// CHECK:       chain(%[[CH90]])
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.mutable.store"{{.*}}>>
mo.graph @store_chain_comes_from_wrong_producer(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create()

  %outTensor, %outChain = mo.mutable.load(%arg0, %0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %1 = mo.add(%arg1, %outTensor) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%arg0, %1, %0) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain

  mo.output
}

// The add kernel's out-chain has two uses (the two mutable.store siblings
// below), so checkMutableKernelFusible blocks fusion into either -- both
// stores remain separate kernels, each chained off the add kernel's out-chain.
// CHECK-LABEL: mo.graph @store_to_fuse_has_sibling(
// CHECK:       %[[CH100:.*]] = mo.chain.create()
// CHECK:       %[[K10:.*]]:2 = map.iter.foreach<[10, 10]>
// CHECK:       chain(%[[CH100]])
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_mutable_load<[#mogg.fusion_leaf<"mo.mutable.load"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>>
// CHECK:       map.iter.foreach<[10, 10]>
// CHECK:       chain(%[[K10]]#1)
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.mutable.store"{{.*}}>>
// CHECK:       map.iter.foreach<[10, 10]>
// CHECK:       chain(%[[K10]]#1)
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.mutable.store"{{.*}}>>
mo.graph @store_to_fuse_has_sibling(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create()

  %outTensor, %outChain = mo.mutable.load(%arg0, %0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %1 = mo.add(%arg1, %outTensor) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%arg0, %1, %outChain) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain

  %3 = mo.mutable.store(%arg0, %arg1, %outChain) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain
  mo.output
}

// The load's tensor has two uses: an elementwise consumer and an in-place
// opaque kernel that also takes the load's buffer directly. Splitting the load
// would let the elementwise kernel fuse the buffer and read it while the opaque
// kernel writes it, so parallelization -- and therefore fusion -- must not
// happen.
// CHECK-LABEL: mo.graph @no_parallelize_with_opaque_buffer_writer(
// CHECK-SAME:  %[[BUF11:.*]]: !mo.buffer<[3, 2], si32>
// CHECK:       %[[T11:.*]], %{{.*}} = mo.mutable.load(%[[BUF11]],
// CHECK-NOT:   mo.mutable.load
// CHECK:       map.iter.foreach<[3, 2]>
// CHECK-NEXT:  inputs  (%[[T11]]: !mo.tensor<[3, 2], si32>)
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.negative"{{.*}}>>
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs  (%[[BUF11]] as {{.*}}: !mo.buffer<[3, 2], si32>, %[[T11]] as {{.*}}: !mo.tensor<[3, 2], si32>,
mo.graph @no_parallelize_with_opaque_buffer_writer(%buf: !mo.buffer<[3, 2], si32>, %offsets: !mo.tensor<[11], ui32>, %penalty: !mo.tensor<[7], f32>) -> (!mo.tensor<[3, 2], si32>, !mo.chain) {
  %ch0 = mo.chain.create()
  %t, %oc = mo.mutable.load(%buf, %ch0) : (!mo.buffer<[3, 2], si32>, !mo.chain) -> (!mo.tensor<[3, 2], si32>, !mo.chain)
  %neg = mo.negative(%t) : !mo.tensor<[3, 2], si32>
  %c = mo.custom {device = #M.device_ref<"gpu", 0>, symbol = "sampler.apply_penalties"}(%buf, %t, %offsets, %penalty, %penalty, %penalty, %oc) : (!mo.buffer<[3, 2], si32>, !mo.tensor<[3, 2], si32>, !mo.tensor<[11], ui32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.chain) -> !mo.chain
  mo.output %neg, %c : !mo.tensor<[3, 2], si32>, !mo.chain
}

// The fused add+store kernel's in-chain (%m) reaches the load's out-chain
// through an exclusive path of single-use chain.creates, so
// detachExclusiveChainPath rewires %m onto the load's *in*-chain and the load
// fuses -- the buffer replaces the tensor as the kernel's first input.
// CHECK-LABEL: mo.graph @detach_exclusive_chain_path(
// CHECK-SAME:  %[[BUF12:.*]]: !mo.buffer<[10, 10], f32>, %[[BUF13:.*]]: !mo.buffer<[10, 10], f32>, %[[X12:.*]]: !mo.tensor<[10, 10], f32>
// CHECK-NOT:   mo.mutable.load
// CHECK:       %[[CH120:.*]] = mo.chain.create()
// CHECK:       %[[M12:.*]] = mo.chain.create(%[[CH120]]) : !mo.chain
// CHECK:       %[[CH121:.*]] = mo.chain.create(%[[M12]], %[[CH120]])
// CHECK:       map.iter.foreach<[10, 10]>
// CHECK-NEXT:  inputs  (%[[BUF12]]: !mo.buffer<[10, 10], f32>, %[[X12]]: !mo.tensor<[10, 10], f32>, %[[BUF13]]: !mo.buffer<[10, 10], f32>)
// CHECK:       chain(%[[CH121]])
// CHECK:       %{{.*}} = map.load %[[BUF12]][%{{.*}}] : !mo.buffer<[10, 10], f32>
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_mutable_load<[#mogg.fusion_leaf<"mo.mutable.load"{{.*}}, #mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.mutable.store"{{.*}}]>]>>
mo.graph @detach_exclusive_chain_path(%buf1: !mo.buffer<[10, 10], f32>, %buf2: !mo.buffer<[10, 10], f32>, %x: !mo.tensor<[10, 10], f32>) {
  %ch0 = mo.chain.create()
  %t1, %oc1 = mo.mutable.load(%buf1, %ch0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %m = mo.chain.create(%oc1) : !mo.chain
  %add = mo.add(%t1, %x) : !mo.tensor<[10, 10], f32>
  %st = mo.mutable.store(%buf2, %add, %m) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain
  mo.output
}

// Same shape as @detach_exclusive_chain_path, but the inner chain.create (%a)
// on the load -> kernel path has a second user, so detachExclusiveChainPath
// bails on its hasOneUse() check and fuseLoadBuffer returns early to avoid a
// chain cycle. The load survives and the kernel keeps reading its tensor.
// Two chain.creates are needed: putting the extra user on the kernel's own
// in-chain would also stop map-fuse-kernels from pre-fusing add+store, leaving
// a chainless kernel that takes the (cycle-check-free)
// fuseLoadBufferAndChains path instead.
// CHECK-LABEL: mo.graph @no_detach_when_chain_path_shared(
// CHECK-SAME:  %[[BUF14:.*]]: !mo.buffer<[10, 10], f32>, %[[BUF15:.*]]: !mo.buffer<[10, 10], f32>, %[[X14:.*]]: !mo.tensor<[10, 10], f32>
// CHECK:       %[[CH140:.*]] = mo.chain.create()
// CHECK:       %[[T14:.*]], %[[OC14:.*]] = mo.mutable.load(%[[BUF14]], %[[CH140]])
// CHECK:       %[[A14:.*]] = mo.chain.create(%[[OC14]]) : !mo.chain
// CHECK:       mo.chain.create(%[[A14]]) : !mo.chain
// CHECK-NEXT:  %[[B14:.*]] = mo.chain.create(%[[A14]]) : !mo.chain
// CHECK:       map.iter.foreach<[10, 10]>
// CHECK-NEXT:  inputs  (%[[T14]]: !mo.tensor<[10, 10], f32>, %[[X14]]: !mo.tensor<[10, 10], f32>, %[[BUF15]]: !mo.buffer<[10, 10], f32>)
// CHECK:       chain(%[[B14]])
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.mutable.store"{{.*}}]>>
mo.graph @no_detach_when_chain_path_shared(%buf1: !mo.buffer<[10, 10], f32>, %buf2: !mo.buffer<[10, 10], f32>, %x: !mo.tensor<[10, 10], f32>) {
  %ch0 = mo.chain.create()
  %t1, %oc1 = mo.mutable.load(%buf1, %ch0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %a = mo.chain.create(%oc1) : !mo.chain
  mo.chain.create(%a) : !mo.chain
  %b = mo.chain.create(%a) : !mo.chain
  %add = mo.add(%t1, %x) : !mo.tensor<[10, 10], f32>
  %st = mo.mutable.store(%buf2, %add, %b) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain
  mo.output
}

// Ported from @fusion_with_slice in
// GraphCompiler/test/mo-opt/MOGGDialect/Transforms/fusion-mutable-loads.mlir.
// The bounds differ from the MOGG original ([3:8:2], which yields 3 elements
// by numpy semantics): MAP::SliceSpecAttr requires the extent to be an exact
// multiple of the step, so [2:8:2] is used here to keep the strided-slice
// shape while satisfying that verifier.
//
// map-fuse-kernels merges the slice foreach with the negative foreach, and the
// load then fuses into the result via fuseLoadBufferAndChains: the buffer takes
// the tensor's operand slot, the kernel gains an in/out chain pair, and the
// slice's index math reads straight from the buffer.
// CHECK-LABEL: mo.graph @fusion_with_slice(
// CHECK-SAME:  %[[BUF16:.*]]: !mo.buffer<[10, 10], f32>
// CHECK-NOT:   mo.mutable.load
// CHECK:       %[[CH160:.*]] = mo.chain.create()
// CHECK:       %{{.*}}:2 = map.iter.foreach<[3, 3]>
// CHECK-NEXT:  inputs  (%[[BUF16]] as %[[BUFARG16:.*]]: !mo.buffer<[10, 10], f32>,
// CHECK:       chain(%[[CH160]])
// CHECK-SAME:  -> (!mo.tensor<[3, 3], f32>, !mo.chain)
// CHECK:       %[[IDX16:.*]] = map.index.slice<#map.index.slice.spec<[10, 10] -> [3, 3], starts = [2, 2], stops = [8, 8], steps = [2, 2]>>
// CHECK:       %{{.*}} = map.load %[[BUFARG16]][%[[IDX16]]] : !mo.buffer<[10, 10], f32>
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_mutable_load<[#mogg.fusion_leaf<"mo.mutable.load"{{.*}}, #mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.slice"{{.*}}, #mogg.fusion_leaf<"mo.negative"{{.*}}]>]>>
mo.graph @fusion_with_slice(%arg0: !mo.buffer<[10, 10], f32>) -> !mo.tensor<[3, 3], f32> {
  %0 = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %1 = mo.constant {value = #M.dense_array<8, 8> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %3 = mo.chain.create()
  %outTensor, %outChain = mo.mutable.load(%arg0, %3) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %4 = mo.slice(%outTensor, %0, %1, %0) : (!mo.tensor<[10, 10], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 3], f32>
  %5 = mo.negative(%4) : !mo.tensor<[3, 3], f32>
  mo.output %5 : !mo.tensor<[3, 3], f32>
}

// The minimal happy path for an opaque kernel: one load, one chained opaque
// consumer, nothing else. The buffer takes the tensor's slot in `inputs`, the
// kernel's map.call.opaque reads the now-buffer-typed block arg, and the
// kernel's chain absorbs the load's in-chain (a no-op merge here, since it
// already carries it).
// CHECK-LABEL: mo.graph @fuse_into_opaque(
// CHECK-SAME:  %[[BUF17:.*]]: !mo.buffer<[3, 2], si32>, %[[TARGET17:.*]]: !mo.buffer<[3, 2], si32>
// CHECK:       %[[CH170:.*]] = mo.chain.create()
// CHECK-NOT:   mo.mutable.load
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs (%[[TARGET17]] as {{.*}}: !mo.buffer<[3, 2], si32>, %[[BUF17]] as %[[BUFARG17:.*]]: !mo.buffer<[3, 2], si32>,
// CHECK-SAME:  %[[CH170]] as {{.*}}: !mo.chain)
// CHECK:       map.call.opaque @{{.*}}sampler_apply_penalties[] inputs({{.*}}%[[BUFARG17]],
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_mutable_load<[#mogg.fusion_leaf<"mo.mutable.load"{{.*}}, #mogg.fusion_leaf<"custom__sampler.apply_penalties"{{.*}}]>>
mo.graph @fuse_into_opaque(%buf: !mo.buffer<[3, 2], si32>, %target: !mo.buffer<[3, 2], si32>, %offsets: !mo.tensor<[11], ui32>, %penalty: !mo.tensor<[7], f32>) -> !mo.chain {
  %ch0 = mo.chain.create()
  %t, %oc = mo.mutable.load(%buf, %ch0) : (!mo.buffer<[3, 2], si32>, !mo.chain) -> (!mo.tensor<[3, 2], si32>, !mo.chain)
  %c = mo.custom {device = #M.device_ref<"gpu", 0>, symbol = "sampler.apply_penalties"}(%target, %t, %offsets, %penalty, %penalty, %penalty, %ch0) : (!mo.buffer<[3, 2], si32>, !mo.tensor<[3, 2], si32>, !mo.tensor<[11], ui32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.chain) -> !mo.chain
  mo.output %c : !mo.chain
}

// Same shape as @fuse_into_opaque, except the opaque kernel writes the load's
// own buffer. A hand-written kernel is free to write any element while reading
// another, so reading the buffer inline is not the same as reading the snapshot
// the load took: no fusion, and the load survives.
// CHECK-LABEL: mo.graph @no_fuse_into_in_place_opaque(
// CHECK-SAME:  %[[BUF18:.*]]: !mo.buffer<[3, 2], si32>
// CHECK:       %[[T18:.*]], %{{.*}} = mo.mutable.load(%[[BUF18]],
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs (%[[BUF18]] as {{.*}}: !mo.buffer<[3, 2], si32>, %[[T18]] as {{.*}}: !mo.tensor<[3, 2], si32>,
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"custom__sampler.apply_penalties"{{.*}}>>
mo.graph @no_fuse_into_in_place_opaque(%buf: !mo.buffer<[3, 2], si32>, %offsets: !mo.tensor<[11], ui32>, %penalty: !mo.tensor<[7], f32>) -> !mo.chain {
  %ch0 = mo.chain.create()
  %t, %oc = mo.mutable.load(%buf, %ch0) : (!mo.buffer<[3, 2], si32>, !mo.chain) -> (!mo.tensor<[3, 2], si32>, !mo.chain)
  %c = mo.custom {device = #M.device_ref<"gpu", 0>, symbol = "sampler.apply_penalties"}(%buf, %t, %offsets, %penalty, %penalty, %penalty, %ch0) : (!mo.buffer<[3, 2], si32>, !mo.tensor<[3, 2], si32>, !mo.tensor<[11], ui32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.chain) -> !mo.chain
  mo.output %c : !mo.chain
}
