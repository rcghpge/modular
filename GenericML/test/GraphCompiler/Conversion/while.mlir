// RUN: mo-opt %s --mo-to-mogg --mo-to-primitives --allow-unregistered-dialect --strip-buffer-attributes | FileCheck %s

// CHECK: mgp.model @while_op
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<2xf32>) {
// CHECK: %[[X:.*]] = mgp.cpu.tensor.create %[[CHAIN]], %{{.*}}, %{{.*}}  : !mgp.cpu.tensor<2xf32>
// CHECK: %[[Y:.*]] = mgp.cpu.tensor.create %[[CHAIN]], %{{.*}}, %{{.*}}  : !mgp.cpu.tensor<f32>
// CHECK: %[[OUT:.*]]:3 = grt.while (%[[CHAIN]]: !mef.chain, %[[X]]: !mgp.cpu.tensor<2xf32>, %[[Y]]: !mgp.cpu.tensor<f32>) {
// CHECK: %[[SHAPE:.*]] = mgp.cpu.tensor.create %[[CHAIN]], %{{.*}}, %{{.*}}  : !mgp.cpu.tensor<0xsi64>
// CHECK: %[[ZERO:.*]] = mgp.cpu.tensor.create %[[CHAIN]], %{{.*}}, %{{.*}}  : !mgp.cpu.tensor<si64>
// CHECK: %{{.*}}, %{{.*}} = mgp.cpu.tensor.extract %[[X]] : !mgp.cpu.tensor<2xf32>
// CHECK: %{{.*}}, %{{.*}} = mgp.cpu.tensor.extract %[[ZERO]] : !mgp.cpu.tensor<si64>
// CHECK: %{{.*}} = "mop.shape.reshape"(%{{.*}}, %[[SHAPE]]) : (!mgp.cpu.tensor_spec<1xf32>, !mgp.cpu.tensor<0xsi64>) -> !mgp.cpu.tensor_spec<f32>
// CHECK: %{{.*}}, %{{.*}} = mgp.cpu.tensor.extract %[[Y]] : !mgp.cpu.tensor<f32>
// CHECK: %[[CH1:.*]] = "mop.kernels.greater"(%[[CHAIN]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[OUT_BUF:.*]], %[[OUT_SPEC:.*]]) : (!mef.chain, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<f32>, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<f32>, !mgp.cpu.buffer<1>, !mgp.cpu.tensor_spec<bool>) -> !mef.chain
// CHECK: %[[COND_TEN:.*]] = mgp.cpu.tensor.create %[[CH1]], %[[OUT_BUF]], %[[OUT_SPEC]]  : !mgp.cpu.tensor<bool>
// CHECK: %[[COND_BUF:.*]], {{.*}} = mgp.cpu.tensor.extract %[[COND_TEN]] : !mgp.cpu.tensor<bool>
// CHECK: %[[COND_SCALAR:.*]] = mgp.to_grt_bool(%[[CHAIN]], %[[COND_BUF]]) : (!mef.chain, !mgp.cpu.buffer<1>) -> i1
// CHECK: mef.output %[[CHAIN]], %[[X]], %[[Y]], %[[COND_SCALAR]] : !mef.chain, !mgp.cpu.tensor<2xf32>, !mgp.cpu.tensor<f32>, i1
// CHECK: } do {
// CHECK: %{{.*}}, %{{.*}} = mgp.cpu.tensor.extract %[[X]] : !mgp.cpu.tensor<2xf32>
// CHECK: %{{.*}}, %{{.*}} = mgp.cpu.tensor.extract %[[X]] : !mgp.cpu.tensor<2xf32>
// CHECK: %[[CH2:.*]] = "mop.kernels.add"(%[[CHAIN]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[OUT_BUF:.*]], %[[OUT_SPEC:.*]]) : (!mef.chain, !mgp.cpu.buffer<8>, !mgp.cpu.tensor_spec<2xf32>, !mgp.cpu.buffer<8>, !mgp.cpu.tensor_spec<2xf32>, !mgp.cpu.buffer<8>, !mgp.cpu.tensor_spec<2xf32>) -> !mef.chain
// CHECK: %[[NEW_X:.*]] = mgp.cpu.tensor.create %[[CH2]], %[[OUT_BUF]], %[[OUT_SPEC]]  : !mgp.cpu.tensor<2xf32>
// CHECK: mef.output %[[CHAIN]], %[[NEW_X]], %[[Y]] : !mef.chain, !mgp.cpu.tensor<2xf32>, !mgp.cpu.tensor<f32>
// CHECK: }
// CHECK: mef.output %[[OUT]]#0, %[[OUT]]#1 : !mef.chain, !mgp.cpu.tensor<2xf32>
// CHECK: }

mo.graph @while_op() -> (!mo.tensor<[2], f32>) {
   %x = mo.constant { value = #M.dense_array<1.0, 1.0> : tensor<2xf32> } : !mo.tensor<[2], f32>
   %y = mo.constant { value = #M.dense_array<4.0> : tensor<1xf32> } : !mo.tensor<[], f32>
   %res:2 = mo.while (%x: !mo.tensor<[2], f32>, %y : !mo.tensor<[], f32>) {
     %zero = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
     %mean0 = mo.mean (%x, %zero) : (!mo.tensor<[2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[1], f32>
     %shape = mo.constant {value = #M.dense_array<> : tensor<0xsi64>} : !mo.tensor<[0], si64>
     %mean1 = mo.reshape (%mean0, %shape) : (!mo.tensor<[1], f32>, !mo.tensor<[0], si64>) -> !mo.tensor<[], f32>
     %cond = mo.greater (%y, %mean1) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
     mo.yield %cond : !mo.tensor<[], bool>
   } do {
     %new_x = mo.add (%x, %x) : !mo.tensor<[2], f32>
     mo.yield %new_x, %y : !mo.tensor<[2], f32>, !mo.tensor<[], f32>
   }
   mo.output %res#0 : !mo.tensor<[2], f32>
}
