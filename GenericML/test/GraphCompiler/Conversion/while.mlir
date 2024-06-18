// RUN: mo-opt %s --add-devices --mo-to-mogg --translate-mogg-to-kgen --inline-lambdas --mo-to-primitives --lower-primitives --remove-non-lowerable-results --lower-abi --canonicalize -cse --jit-compile-kernels --lower-mogg-region-compile | FileCheck %s

// CHECK: mgp.model @while_op
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> (!mef.chain, !mgp.tensor<2xf32>) {
// CHECK: %[[UNUSED_CH:.*]] = mgp.chain.host_to_device %[[CH0]], %[[CTX]][1]
// CHECK: %[[UNUSED:.*]] = mgp.tensor.create %[[UNUSED_CH]], %[[CTX]][1], {{.*}}, %{{.*}} : !mgp.tensor<0xsi64>
// CHECK: %[[ZERO:.*]] = mgp.tensor.create %[[UNUSED_CH]], %[[CTX]][1], {{.*}}, %{{.*}} : !mgp.tensor<si64>
// CHECK: %[[X:.*]] = mgp.tensor.create %[[UNUSED_CH]], %[[CTX]][1], %{{.*}}, %{{.*}}  : !mgp.tensor<2xf32>
// CHECK: %[[Y:.*]] = mgp.tensor.create %[[UNUSED_CH]], %[[CTX]][1], %{{.*}}, %{{.*}}  : !mgp.tensor<f32>
// CHECK: %[[OUT:.*]]:5 = grt.while (%[[CH0]]: !mef.chain, %[[X]]: !mgp.tensor<2xf32>, %[[CTX]]: !mgp.context, %[[ZERO]]: !mgp.tensor<si64>, %[[Y]]: !mgp.tensor<f32>) {
// CHECK: %[[outChain_6:.*]], %[[outResults_7:.*]] = "mgp.generic.execute"(%[[CTX]], %{{.*}}){{.*}}dSummary = "mgp.tensor_spec.create"
// CHECK: %[[outChain_8:.*]], %[[outResults_9:.*]] = "mgp.generic.execute"(%[[CTX]], %{{.*}}, %{{.*}}, %[[CTX]]){{.*}}dSummary = "mgp.buffer.alloc.static"
// CHECK: %[[outChain_10:.*]] = "mgp.generic.execute"(%[[CTX]], %{{.*}}, %[[CH0]], %[[outResults_9]], %{{.*}}, %{{.*}}){{.*}}dSummary = "mo.mean
// CHECK: %[[C_TENSOR:.*]] = mgp.tensor.create %{{.*}}, %[[CTX]][1], %[[outResults_9]], %[[outResults_7]] : !mgp.tensor<1xf32>
// CHECK: %[[TENSOR:.*]] = mgp.tensor.extract.buffer %[[CTX]][1], %[[C_TENSOR]] : !mgp.tensor<1xf32>
// CHECK: %[[outChain_11:.*]], %[[outResults_12:.*]] = "mgp.generic.execute"(%[[CTX]], %{{.*}}){{.*}}dSummary = "mgp.tensor_spec.create"
// CHECK: %[[C1_TENSOR:.*]] = mgp.tensor.create %{{.*}}, %[[CTX]][1], %[[TENSOR]], %[[outResults_12]] : !mgp.tensor<f32>
// CHECK: %[[BUF1:.*]] = mgp.tensor.extract.buffer %[[CTX]][1], %[[Y]] : !mgp.tensor<f32>
// CHECK: %[[BUF2:.*]] = mgp.tensor.extract.buffer %[[CTX]][1], %[[C1_TENSOR]] : !mgp.tensor<f32>
// CHECK: %[[outChain_13:.*]], %[[outResults_14:.*]] = "mgp.generic.execute"{{.*}}dSummary = "mgp.tensor_spec.create"
// CHECK: %[[outChain_15:.*]], %[[outResults_16:.*]] = "mgp.generic.execute"{{.*}}dSummary = "mgp.buffer.alloc.static"
// CHECK: %[[outChain_17:.*]] = "mgp.generic.execute"(%[[CTX]], %{{.*}}, %[[CH0]], %[[outResults_16]], %[[BUF1]], %[[BUF2]]){{.*}}dSummary = "mo.greater
// CHECK: %[[VAL33:.*]] = mgp.tensor.create %{{.*}}, %[[CTX]][1], %[[outResults_16]], %[[outResults_14]] : !mgp.tensor<bool>
// CHECK: %[[VAL34:.*]] = mgp.tensor.extract.buffer %[[CTX]][1], %[[VAL33]] : !mgp.tensor<bool>
// CHECK: %[[VAL35:.*]] = mgp.buffer.to_bool %{{.*}}, %[[CTX]][1], %[[VAL34]] : !mgp.buffer<1>
// CHECK: mef.output %[[CH0]], %[[X]], %[[CTX]], %[[ZERO]], %[[Y]], %[[VAL35]] : !mef.chain, !mgp.tensor<2xf32>, !mgp.context, !mgp.tensor<si64>, !mgp.tensor<f32>, i1
// CHECK: } do {
// CHECK: %[[VAL14:.*]] = mgp.tensor.extract.buffer %[[CTX]][1], %[[X]] : !mgp.tensor<2xf32>
// CHECK: %[[outChain_6:.*]], %[[outResults_7:.*]] = "mgp.generic.execute"{{.*}}dSummary = "mgp.tensor_spec.create"
// CHECK: %[[outChain_8:.*]], %[[outResults_9:.*]] = "mgp.generic.execute"{{.*}}dSummary = "mgp.buffer.alloc.static"
// CHECK: %[[outChain_10:.*]] = "mgp.generic.execute"(%[[CTX]], %{{.*}}, %[[CH0]], %[[outResults_9]], %[[VAL14]], %[[VAL14]]){{.*}}dSummary = "mo.add
// CHECK: %[[VAL21:.*]] = mgp.tensor.create %{{.*}}, %[[CTX]][1], %[[outResults_9]], %[[outResults_7]] : !mgp.tensor<2xf32>
// CHECK: mef.output %[[CH0]], %[[VAL21]], %[[CTX]], %[[ZERO]], %[[Y]] : !mef.chain, !mgp.tensor<2xf32>, !mgp.context, !mgp.tensor<si64>, !mgp.tensor<f32>
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
