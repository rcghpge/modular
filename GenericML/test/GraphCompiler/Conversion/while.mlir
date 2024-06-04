// RUN: mo-opt %s --add-devices --mo-to-mogg --translate-mogg-to-kgen --jit-compile-kernels --mo-to-primitives | FileCheck %s

// CHECK: mgp.model @while_op
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> (!mef.chain, !mgp.tensor<2xf32>) {
// CHECK: %[[UNUSED_CH:.*]] = mgp.chain.host_to_device %[[CH0]], %[[CTX]][1]
// CHECK: %[[UNUSED:.*]] = mgp.tensor.create %[[UNUSED_CH]], %[[CTX]][1], {{.*}}, %{{.*}} : !mgp.tensor<0xsi64>
// CHECK: %[[CH1:.*]] = mgp.chain.host_to_device %[[CH0]], %[[CTX]][1]
// CHECK: %[[ZERO:.*]] = mgp.tensor.create %[[CH1]], %[[CTX]][1], {{.*}}, %{{.*}} : !mgp.tensor<si64>
// CHECK: %[[CH2:.*]] = mgp.chain.host_to_device %[[CH0]], %[[CTX]][1]
// CHECK: %[[X:.*]] = mgp.tensor.create %[[CH2]], %[[CTX]][1], %{{.*}}, %{{.*}}  : !mgp.tensor<2xf32>
// CHECK: %[[CH3:.*]] = mgp.chain.host_to_device %[[CH0]], %[[CTX]][1]
// CHECK: %[[Y:.*]] = mgp.tensor.create %[[CH3]], %[[CTX]][1], %{{.*}}, %{{.*}}  : !mgp.tensor<f32>
// CHECK: %[[OUT:.*]]:5 = grt.while (%[[CH0]]: !mef.chain, %[[X]]: !mgp.tensor<2xf32>, %[[CTX]]: !mgp.context, %[[ZERO]]: !mgp.tensor<si64>, %[[Y]]: !mgp.tensor<f32>) {
// CHECK:        {{.*}} = mgp.chain.host_to_device %[[CH0]], %[[CTX]][1]
// CHECK:        %{{.*}} = mgp.kernel.execute {{.*}}, %[[CTX]]{{.*}}, summary "mo.mean{{.*}}"
// CHECK:        %[[CH2:.*]] = mgp.kernel.execute {{.*}}, %[[CTX]]{{.*}}, summary "mo.greater{{.*}}"
// CHECK-NEXT:   %[[COND_TEN:.*]] = mgp.tensor.create %[[CH2]], %[[CTX]]{{.*}}
// CHECK-NEXT:   %[[COND_BUF:.*]]= mgp.tensor.extract.buffer %{{.*}}
// CHECK:        %[[COND_CH:.*]] = mgp.chain.host_to_device %[[CH0]], %[[CTX]][1]
// CHECK:        %[[COND_SCALAR:.*]] = mgp.buffer.to_bool %[[COND_CH]], %[[CTX]][1], %[[COND_BUF]]
// CHECK:        mef.output %[[CH0]], %[[X]], %[[CTX]], %[[ZERO]], %[[Y]], %[[COND_SCALAR]]
// CHECK: } do {
// CHECK:        %{{.*}} = mgp.tensor.extract.buffer %[[CTX]][1], %[[X]]
// CHECK:        %{{.*}} = mgp.tensor.extract.tensor_spec %[[CTX]][1], %[[X]]
// CHECK:        %{{.*}} = mgp.tensor.extract.buffer %[[CTX]][1], %[[X]]
// CHECK:        %{{.*}} = mgp.tensor.extract.tensor_spec %[[CTX]][1], %[[X]]
// CHECK:        %[[CH3:.*]] = mgp.kernel.execute {{.*}}, %[[CTX]]{{.*}}, summary "mo.add{{.*}}"
// CHECK:        %[[NEW_X:.*]] = mgp.tensor.create %[[CH3]], %[[CTX]][1], %{{.*}}, %{{.*}}
// CHECK:        mef.output %[[CH0]], %[[NEW_X]], %[[CTX]], %[[ZERO]], %[[Y]]
// CHECK: }
// CHECK: mef.output %[[OUT]]#0, %[[OUT]]#1 : !mef.chain, !mgp.tensor<2xf32>
// CHECK: }

mo.graph @while_op() -> (!mo.tensor<[2], f32>) {
   %x = mo.constant { value = #M.dense_array<1.0, 1.0> : tensor<2xf32> } : !mo.tensor<[2], f32>
   %y = mo.constant { value = #M.dense_array<4.0> : tensor<1xf32> } : !mo.tensor<[], f32>
   %res = mo.while (%x as %xi: !mo.tensor<[2], f32>) {
     %zero = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
     %mean0 = mo.mean (%xi, %zero) : (!mo.tensor<[2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[1], f32>
     %shape = mo.constant {value = #M.dense_array<> : tensor<0xsi64>} : !mo.tensor<[0], si64>
     %mean1 = mo.reshape (%mean0, %shape) : (!mo.tensor<[1], f32>, !mo.tensor<[0], si64>) -> !mo.tensor<[], f32>
     %cond = mo.greater (%y, %mean1) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
     mo.yield %cond : !mo.tensor<[], bool>
   } do {
     %new_x = mo.add (%xi, %xi) : !mo.tensor<[2], f32>
     mo.yield %new_x : !mo.tensor<[2], f32>
   }
   mo.output %res : !mo.tensor<[2], f32>
}
