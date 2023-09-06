// RUN: mo-opt %s --mo-to-mogg --jit-compile-kernels --mo-to-primitives --allow-unregistered-dialect | FileCheck %s

// CHECK: mgp.model @while_op
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<2xf32>) {
// CHECK: %[[X:.*]] = mgp.cpu.tensor.create %[[CHAIN]], %{{.*}}, %{{.*}}  : !mgp.cpu.tensor<2xf32>
// CHECK: %[[Y:.*]] = mgp.cpu.tensor.create %[[CHAIN]], %{{.*}}, %{{.*}}  : !mgp.cpu.tensor<f32>
// CHECK: %[[OUT:.*]]:4 = grt.while (%[[CHAIN]]: !mef.chain, %[[CTX]]: !mgp.context, %[[X]]: !mgp.cpu.tensor<2xf32>, %[[Y]]: !mgp.cpu.tensor<f32>) {
// CHECK: %[[MEAN:.*]] = mgp.cpu.kernel.execute {{.*}}, name "{{.*}}mo.mean{{.*}}"
// CHECK: %[[CH1:.*]] = mgp.cpu.kernel.execute {{.*}}, name "{{.*}}mo.greater{{.*}}"
// CHECK-NEXT: %[[COND_TEN:.*]] = mgp.cpu.tensor.create %[[CH1]]
// CHECK-NEXT: %[[COND_BUF:.*]], {{.*}} = mgp.cpu.tensor.extract %{{.*}} : !mgp.cpu.tensor<bool>
// CHECK-NEXT: %[[COND_SCALAR:.*]] = mgp.to_grt_bool(%[[CHAIN]], %[[COND_BUF]]) : (!mef.chain, !mgp.cpu.buffer<1>) -> i1
// CHECK: mef.output %[[CHAIN]], %[[CTX]], %[[X]], %[[Y]], %[[COND_SCALAR]] : !mef.chain, !mgp.context, !mgp.cpu.tensor<2xf32>, !mgp.cpu.tensor<f32>, i1
// CHECK: } do {
// CHECK: mgp.cpu.tensor.extract %[[X]] : !mgp.cpu.tensor<2xf32>
// CHECK: mgp.cpu.tensor.extract %[[X]] : !mgp.cpu.tensor<2xf32>
// CHECK: %[[CH2:.*]] = mgp.cpu.kernel.execute %[[CHAIN]], %[[CTX]]
// CHECK: %[[NEW_X:.*]] = mgp.cpu.tensor.create %[[CH2]], %{{.*}}, %{{.*}}  : !mgp.cpu.tensor<2xf32>
// CHECK: mef.output %[[CHAIN]], %[[CTX]], %[[NEW_X]], %[[Y]] : !mef.chain, !mgp.context, !mgp.cpu.tensor<2xf32>, !mgp.cpu.tensor<f32>
// CHECK: }
// CHECK: mef.output %[[OUT]]#0, %[[OUT]]#2 : !mef.chain, !mgp.cpu.tensor<2xf32>
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
