// RUN: mo-opt %s --mo-to-primitives --allow-unregistered-dialect | FileCheck %s

// CHECK: mgp.model @mogg_split : tf init {
// CHECK-NEXT:   %[[CH0:.*]] = grt.chain.create
// CHECK-NEXT:   %[[CTX:.*]] = mgp.context.create 1
// CHECK-NEXT:   %[[CH1:.*]] = "mop.create_jit_context"(%[[CH0]], %[[CTX]]) {slot = 0 : ui32}
// CHECK-SAME:   : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK-NEXT:   mef.output %[[CH1]], %[[CTX]] : !mef.chain, !mgp.context
// CHECK-NEXT: } execute (
// CHECK-SAME:   %[[CH:.*]]: !mef.chain,
// CHECK-SAME:   %[[CTX:.*]]: !mgp.context,
// CHECK-SAME:   %[[INPUT:.*]]: !mgp.cpu.tensor<?x8xf32>,
// CHECK-SAME:   %[[SPLIT_SIZES:.*]]: !mgp.cpu.tensor<2xsi64>
// CHECK-SAME: ) -> (!mef.chain, !mgp.cpu.tensor<?x8xf32>, !mgp.cpu.tensor<?x8xf32>) {
// CHECK-NEXT:   %[[INPUT_BUF:.*]], %[[INPUT_RES:.*]] = mgp.cpu.tensor.extract %[[INPUT]] : !mgp.cpu.tensor<?x8xf32>
// CHECK-NEXT:   %[[SPLIT_SPECS:.*]]:2 = "mop.shape.split_axis_0"(%[[INPUT_RES]], %[[SPLIT_SIZES]])
// CHECK-SAME:   : (!mgp.cpu.tensor_spec<?x8xf32>, !mgp.cpu.tensor<2xsi64>) -> (!mgp.cpu.tensor_spec<?x8xf32>, !mgp.cpu.tensor_spec<?x8xf32>)
// CHECK-NEXT:   %[[SPLIT_SPEC1:.*]] = mgp.cpu.tensor_spec.size %[[SPLIT_SPECS]]#0 : !mgp.cpu.tensor_spec<?x8xf32>
// CHECK-NEXT:   %[[SPLIT_SPEC2:.*]] = mgp.cpu.tensor_spec.size %[[SPLIT_SPECS]]#1 : !mgp.cpu.tensor_spec<?x8xf32>
// CHECK-NEXT:   %[[SPLIT_BUFS:.*]]:2 = mgp.cpu.buffer.split.borrow %[[INPUT_BUF]], %[[SPLIT_SPEC1]], %[[SPLIT_SPEC2]]
// CHECK-SAME:   : !mgp.cpu.buffer  -> (!mgp.cpu.buffer, !mgp.cpu.buffer)
// CHECK-NEXT:   %[[SPLIT_TENSOR1:.*]] = mgp.cpu.tensor.create %[[CH]], %[[SPLIT_BUFS]]#0, %[[SPLIT_SPECS]]#0  : !mgp.cpu.tensor<?x8xf32>
// CHECK-NEXT:   %[[SPLIT_TENSOR2:.*]] = mgp.cpu.tensor.create %[[CH]], %[[SPLIT_BUFS]]#1, %[[SPLIT_SPECS]]#1  : !mgp.cpu.tensor<?x8xf32>
// CHECK-NEXT:   mef.output %[[CH]], %[[SPLIT_TENSOR1]], %[[SPLIT_TENSOR2]] : !mef.chain, !mgp.cpu.tensor<?x8xf32>, !mgp.cpu.tensor<?x8xf32>
// CHECK-NEXT: }

mo.graph @mogg_split(
  %input: !mo.tensor<[?, 8], f32>,
  %split_sizes: !mo.tensor<[2], si64>
) -> (!mo.tensor<[?, 8], f32>, !mo.tensor<[?, 8], f32>) {
  %res:2 = mogg.split(%input, %split_sizes) : (
    !mo.tensor<[?, 8], f32>, !mo.tensor<[2], si64>
  ) -> (!mo.tensor<[?, 8], f32>, !mo.tensor<[?, 8], f32>)
  mo.output %res#0, %res#1 : !mo.tensor<[?, 8], f32>, !mo.tensor<[?, 8], f32>
}
