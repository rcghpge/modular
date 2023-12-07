// RUN: mo-opt %s --add-devices --mo-to-mogg --jit-compile-kernels --mo-to-primitives | FileCheck %s

// CHECK: mgp.model @while_op
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<2xf32>) {
// CHECK: %[[ZERO:.*]] = mgp.cpu.tensor.create %[[CH0]], {{.*}}, %{{.*}} : !mgp.cpu.tensor<si64>
// CHECK: %[[X:.*]] = mgp.cpu.tensor.create %[[CH0]], %{{.*}}, %{{.*}}  : !mgp.cpu.tensor<2xf32>
// CHECK: %[[Y:.*]] = mgp.cpu.tensor.create %[[CH0]], %{{.*}}, %{{.*}}  : !mgp.cpu.tensor<f32>
// CHECK: %[[OUT:.*]]:5 = grt.while (%[[CH0]]: !mef.chain, %[[X]]: !mgp.cpu.tensor<2xf32>, %[[ZERO]]: !mgp.cpu.tensor<si64>, %[[CTX]]: !mgp.context, %[[Y]]: !mgp.cpu.tensor<f32>) {
// CHECK:        %{{.*}} = mgp.cpu.kernel.execute %[[CH0]], %[[CTX]]{{.*}}, summary "mo.mean{{.*}}"
// CHECK:        %[[CH2:.*]] = mgp.cpu.kernel.execute %[[CH0]], %[[CTX]]{{.*}}, summary "mo.greater{{.*}}"
// CHECK-NEXT:   %[[COND_TEN:.*]] = mgp.cpu.tensor.create %[[CH2]], %{{.*}}, %{{.*}}
// CHECK-NEXT:   %[[COND_BUF:.*]], {{.*}} = mgp.cpu.tensor.extract %{{.*}}
// CHECK-NEXT:   %[[COND_SCALAR:.*]] = mgp.to_grt_bool(%[[CH0]], %[[COND_BUF]])
// CHECK:        mef.output %[[CH0]], %[[X]], %[[ZERO]], %[[CTX]], %[[Y]], %[[COND_SCALAR]]
// CHECK: } do {
// CHECK:        %{{.*}} = mgp.cpu.tensor.extract %[[X]]
// CHECK:        %{{.*}} = mgp.cpu.tensor.extract %[[X]]
// CHECK:        %[[CH3:.*]] = mgp.cpu.kernel.execute %[[CH0]], %[[CTX]]{{.*}}, summary "mo.add{{.*}}"
// CHECK:        %[[NEW_X:.*]] = mgp.cpu.tensor.create %[[CH3]], %{{.*}}, %{{.*}}
// CHECK:        mef.output %[[CH0]], %[[NEW_X]], %[[ZERO]], %[[CTX]], %[[Y]]
// CHECK: }
// CHECK: mef.output %[[OUT]]#0, %[[OUT]]#1 : !mef.chain, !mgp.cpu.tensor<2xf32>
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
