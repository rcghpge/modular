// RUN: mo-opt %s --add-devices --mo-to-primitives --canonicalize | FileCheck %s

// CHECK: mgp.model @mogg_split : mof init {
// CHECK: } execute (
// CHECK-SAME:   %[[CH:.*]]: !mef.chain,
// CHECK-SAME:   %[[CTX:.*]]: !mgp.context,
// CHECK-SAME:   %[[INPUT:.*]]: !mgp.tensor<?x8xf32>,
// CHECK-SAME:   %[[DUMMY:.*]]: !mgp.tensor<?x?xsi64>
// CHECK-SAME: ) -> (!mef.chain, !mgp.tensor<?x8xf32>, !mgp.tensor<?x8xf32>) {
// CHECK-NEXT:   %[[DUMMY_SPEC:.*]] = mgp.tensor.extract.tensor_spec %[[CTX]][1], %[[DUMMY]] : !mgp.tensor<?x?xsi64>
// CHECK-NEXT:   %[[DIMS:.*]]:2 = mgp.tensor_spec.dims.dynamic %[[DUMMY_SPEC]] : !mgp.tensor_spec<?x?xsi64>
// CHECK-NEXT:   %[[INPUT_BUF:.*]] = mgp.tensor.extract.buffer %[[CTX]][1], %[[INPUT]] : !mgp.tensor<?x8xf32>
// CHECK-NEXT:   %[[SPLIT_SPEC1:.*]] = mgp.tensor_spec.create %[[DIMS]]#0 : !mgp.tensor_spec<?x8xf32>
// CHECK-NEXT:   %[[SPLIT_SPEC2:.*]] = mgp.tensor_spec.create %[[DIMS]]#1 : !mgp.tensor_spec<?x8xf32>
// CHECK-NEXT:   %[[SPLIT_SIZE1:.*]] = mgp.tensor_spec.size %[[SPLIT_SPEC1]] : !mgp.tensor_spec<?x8xf32>
// CHECK-NEXT:   %[[SPLIT_SIZE2:.*]] = mgp.tensor_spec.size %[[SPLIT_SPEC2]] : !mgp.tensor_spec<?x8xf32>
// CHECK-NEXT:   %[[CH1:.*]] = mgp.chain.host_to_device %[[CH]], %[[CTX]][1]
// CHECK-NEXT:   %[[SPLIT_BUFS:.*]]:2 = mgp.buffer.split.borrow %[[CH1]], %[[CTX]][1], %[[INPUT_BUF]], %[[SPLIT_SIZE1]], %[[SPLIT_SIZE2]]
// CHECK-SAME:   : !mgp.buffer<>  -> (!mgp.buffer<>, !mgp.buffer<>)
// CHECK-NEXT:   %[[CH2:.*]] = mgp.chain.host_to_device %[[CH]], %[[CTX]][1]
// CHECK-NEXT:   %[[SPLIT_TENSOR1:.*]] = mgp.tensor.create %[[CH2]], %[[CTX]][1], %[[SPLIT_BUFS]]#0, %[[SPLIT_SPEC1]]  : !mgp.tensor<?x8xf32>
// CHECK-NEXT:   %[[CH3:.*]] = mgp.chain.host_to_device %[[CH]], %[[CTX]][1]
// CHECK-NEXT:   %[[SPLIT_TENSOR2:.*]] = mgp.tensor.create %[[CH3]], %[[CTX]][1], %[[SPLIT_BUFS]]#1, %[[SPLIT_SPEC2]]  : !mgp.tensor<?x8xf32>
// CHECK-NEXT:   mef.output %[[CH]], %[[SPLIT_TENSOR1]], %[[SPLIT_TENSOR2]] : !mef.chain, !mgp.tensor<?x8xf32>, !mgp.tensor<?x8xf32>
// CHECK-NEXT: }

// We pass in the output shape parameters of split to separate how we derived
// those parameters from the mogg e2e testing.
mo.graph @mogg_split<D0, D1, D2>(
  %input: !mo.tensor<[D0, 8], f32>,
  %dummy: !mo.tensor<[D1, D2], si64>
) -> (!mo.tensor<[D1, 8], f32>, !mo.tensor<[D2, 8], f32>) {

  %res:2 = mogg.split(%input) : (!mo.tensor<[D0, 8], f32>
  ) -> (!mo.tensor<[D1, 8], f32>, !mo.tensor<[D2, 8], f32>)

  mo.output %res#0, %res#1 : !mo.tensor<[D1, 8], f32>, !mo.tensor<[D2, 8], f32>
}
