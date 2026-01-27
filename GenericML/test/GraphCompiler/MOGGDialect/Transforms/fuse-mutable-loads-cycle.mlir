// RUN: mo-opt -mo-to-mogg --mogg-fuse-mutable-loads %s | FileCheck %s

// Check that the second mutable.load doesn't fuse when doing so would create a
// cycle.

// CHECK-LABEL: @cycle
// CHECK: {{summary = "mo\.mutable\.load:[0-9]+\+custom__sampler\.apply_penalties}}
// CHECK: mo.mutable.load
// CHECK: summary = "custom__sampler.apply_penalties
// CHECK: mo.mutable.load
mo.graph @cycle(%arg0: !mo.buffer<[7, 5], f32>, %arg1: !mo.buffer<[3, 2], si32>, %arg2: !mo.tensor<[11], ui32>, %arg3: !mo.tensor<[7], f32>, %arg4: !mo.tensor<[7], f32>) -> (!mo.tensor<[7, 5], f32>, !mo.chain, !mo.chain) {
  %2 = mo.chain.create()
  %outTensor, %outChain = mo.mutable.load(%arg1, %2) : (!mo.buffer<[3, 2], si32>, !mo.chain) -> (!mo.tensor<[3, 2], si32>, !mo.chain)
  %4 = mo.custom {device = #M.device_ref<"gpu", 0>, symbol = "sampler.apply_penalties"}(%arg0, %outTensor, %arg2, %arg4, %arg4, %arg4, %2) : (!mo.buffer<[7, 5], f32>, !mo.tensor<[3, 2], si32>, !mo.tensor<[11], ui32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.chain) -> !mo.chain
  %outTensor_0, %outChain_1 = mo.mutable.load(%arg1, %2) : (!mo.buffer<[3, 2], si32>, !mo.chain) -> (!mo.tensor<[3, 2], si32>, !mo.chain)
  %5 = mo.chain.create(%2, %4, %outChain, %outChain_1) : !mo.chain, !mo.chain, !mo.chain, !mo.chain
  %8 = mo.custom {device = #M.device_ref<"gpu", 0>, symbol = "sampler.apply_penalties"}(%arg0, %outTensor_0, %arg2, %arg3, %arg4, %arg4, %5) : (!mo.buffer<[7, 5], f32>, !mo.tensor<[3, 2], si32>, !mo.tensor<[11], ui32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.chain) -> !mo.chain
  // One other user to break fusion.
  %outTensor_2, %outChain_3 = mo.mutable.load(%arg1, %5) : (!mo.buffer<[3, 2], si32>, !mo.chain) -> (!mo.tensor<[3, 2], si32>, !mo.chain)
  %outTensor_3, %outChain_4 = mo.mutable.load(%arg0, %8) : (!mo.buffer<[7, 5], f32>, !mo.chain) -> (!mo.tensor<[7, 5], f32>, !mo.chain)
  mo.output %outTensor_3, %outChain_3, %outChain_4 : !mo.tensor<[7, 5], f32>, !mo.chain, !mo.chain
}
