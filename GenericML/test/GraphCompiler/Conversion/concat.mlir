// RUN: mo-opt %s --mo-to-mogg --mo-to-primitives --canonicalize --cse | FileCheck %s

// CHECK-LABEL: mgp.model @concat_dynamic_output
// CHECK:  } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[OUT_CHAIN0:.*]] = mogg.generic.compiled.region
// CHECK: %[[OUT_CHAIN1:.*]] = mogg.generic.compiled.region
// CHECK: %[[OUT_CHAIN3:.*]] = grt.chain.create (%[[OUT_CHAIN0]], %[[OUT_CHAIN1]]
// CHECK: %[[CH2:.*]] = mgp.buffer.concat %[[OUT_CHAIN3]], %[[CTX]][0], %[[BUF:.*]], %{{.*}}, %{{.*}}
// CHECK: %[[TEN:.*]] = mgp.tensor.create %[[CH2]], %[[CTX]][0], %[[BUF]]
// CHECK: mef.output %[[CH2]], %[[TEN]] : !mef.chain, !mgp.tensor<?xf32>

mo.graph @concat_dynamic_output<D6, D7>(%arg0: !mo.tensor<[D6], f32>) -> !mo.tensor<[D7], f32> {
  %0 = mo.relu(%arg0) : !mo.tensor<[D6], f32>
  %1 = mo.tanh(%arg0) : !mo.tensor<[D6], f32>
  %2 = mogg.concat(%0, %1) : (!mo.tensor<[D6], f32>, !mo.tensor<[D6], f32>) -> !mo.tensor<[D7], f32>
  mo.output %2 : !mo.tensor<[D7], f32>
}
