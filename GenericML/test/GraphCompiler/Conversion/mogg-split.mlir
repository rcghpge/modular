// RUN: mo-opt %s --mo-to-primitives --allow-unregistered-dialect --canonicalize | FileCheck %s

// CHECK: mgp.model @mogg_split : tf {{.*}} init {
// CHECK-NEXT:   %[[CH0:.*]] = grt.chain.create ()
// CHECK-NEXT:   %[[CTX:.*]] = mgp.context.create 1
// CHECK-NEXT:   %[[CH1:.*]] = "mop.create_jit_context"(%[[CH0]], %[[CTX]]) {slot = 0 : ui32}
// CHECK-SAME:   : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK-NEXT:   mef.output %[[CH1]], %[[CTX]] : !mef.chain, !mgp.context
// CHECK-NEXT: } execute (
// CHECK-SAME:   %[[CH:.*]]: !mef.chain,
// CHECK-SAME:   %[[CTX:.*]]: !mgp.context,
// CHECK-SAME:   %[[INPUT:.*]]: !mgp.cpu.tensor<?x8xf32>,
// CHECK-SAME:   %[[SPLIT_SIZES:.*]]: !mgp.cpu.tensor<2xsi64>
// CHECK-SAME:   %[[DUMMY:.*]]: !mgp.cpu.tensor<?x?xsi64>
// CHECK-SAME: ) -> (!mef.chain, !mgp.cpu.tensor<?x8xf32>, !mgp.cpu.tensor<?x8xf32>) {
// CHECK-NEXT:   %{{.*}}, %[[DUMMY_SPEC:.*]] = mgp.cpu.tensor.extract %[[DUMMY]] : !mgp.cpu.tensor<?x?xsi64>
// CHECK-NEXT:   %[[DIMS:.*]]:2 = mgp.cpu.tensor_spec.dims.dynamic %[[DUMMY_SPEC]] : !mgp.cpu.tensor_spec<?x?xsi64>
// CHECK-NEXT:   %[[INPUT_BUF:.*]], %{{.*}} = mgp.cpu.tensor.extract %[[INPUT]] : !mgp.cpu.tensor<?x8xf32>
// CHECK-NEXT:   %[[SPLIT_SPEC1:.*]] = mgp.cpu.tensor_spec.create %[[DIMS]]#0 : !mgp.cpu.tensor_spec<?x8xf32>
// CHECK-NEXT:   %[[SPLIT_SPEC2:.*]] = mgp.cpu.tensor_spec.create %[[DIMS]]#1 : !mgp.cpu.tensor_spec<?x8xf32>
// CHECK-NEXT:   %[[SPLIT_SIZE1:.*]] = mgp.cpu.tensor_spec.size %[[SPLIT_SPEC1]] : !mgp.cpu.tensor_spec<?x8xf32>
// CHECK-NEXT:   %[[SPLIT_SIZE2:.*]] = mgp.cpu.tensor_spec.size %[[SPLIT_SPEC2]] : !mgp.cpu.tensor_spec<?x8xf32>
// CHECK-NEXT:   %[[SPLIT_BUFS:.*]]:2 = mgp.cpu.buffer.split.borrow %[[CH]], %[[INPUT_BUF]], %[[SPLIT_SIZE1]], %[[SPLIT_SIZE2]]
// CHECK-SAME:   : !mgp.cpu.buffer  -> (!mgp.cpu.buffer, !mgp.cpu.buffer)
// CHECK-NEXT:   %[[SPLIT_TENSOR1:.*]] = mgp.cpu.tensor.create %[[CH]], %[[SPLIT_BUFS]]#0, %[[SPLIT_SPEC1]]  : !mgp.cpu.tensor<?x8xf32>
// CHECK-NEXT:   %[[SPLIT_TENSOR2:.*]] = mgp.cpu.tensor.create %[[CH]], %[[SPLIT_BUFS]]#1, %[[SPLIT_SPEC2]]  : !mgp.cpu.tensor<?x8xf32>
// CHECK-NEXT:   mef.output %[[CH]], %[[SPLIT_TENSOR1]], %[[SPLIT_TENSOR2]] : !mef.chain, !mgp.cpu.tensor<?x8xf32>, !mgp.cpu.tensor<?x8xf32>
// CHECK-NEXT: }

// We pass in the output shape parameters of split to separate how we derived
// those parameters from the mogg e2e testing.
mo.graph @mogg_split<D0, D1, D2>(
  %input: !mo.tensor<[D0, 8], f32>,
  %split_sizes: !mo.tensor<[2], si64>,
  %dummy: !mo.tensor<[D1, D2], si64>
) -> (!mo.tensor<[D1, 8], f32>, !mo.tensor<[D2, 8], f32>) {

  %res:2 = mogg.split(%input, %split_sizes) : (
    !mo.tensor<[D0, 8], f32>, !mo.tensor<[2], si64>
  ) -> (!mo.tensor<[D1, 8], f32>, !mo.tensor<[D2, 8], f32>)

  mo.output %res#0, %res#1 : !mo.tensor<[D1, 8], f32>, !mo.tensor<[D2, 8], f32>
}
