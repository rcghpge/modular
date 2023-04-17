// RUN: mgp-opt %s --allow-unregistered-dialect --remove-redundant-tensor-extracts --defer-allocs | FileCheck %s

// Check mgc.cpu.tensor.allocs.* ops are deferred until the writing op is ready.

// Original graph:
// mo.graph @basicMoGraph() -> (!mo.tensor<[8], f32>) {
//  %const0 = "mo.constant"() {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<8xf32>} : () -> !mo.tensor<[8],f32>
//  %const1 = "mo.constant"() {value = #M.dense_array<10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0> : tensor<8xf32>} : () -> !mo.tensor<[8], f32>  %1 = mo.mul(%iota, %tens) : !mo.tensor<[8], f32>
//  %x = mo.mul(%const0, %const1) : !mo.tensor<[8], f32>
//  %y = mo.sub(%x, %const1) : !mo.tensor<[8], f32>
//  %r = mo.add(%y, %x) : !mo.tensor<[8], f32>
//  mo.output %r : !mo.tensor<[8], f32>
//}

// CHECK-LABEL: mgp.model @basicMoGraph : tf init {
// CHECK: execute (%[[CH0:.*]]: !mef.chain,
// CHECK: %[[IN_BUF_1:.*]] = mgp.cpu.buffer.constant
// CHECK-NEXT: %[[IN_SPEC_1:.*]] = mgp.cpu.tensor_spec.create
// CHECK-NEXT: %[[IN_BUF_2:.*]] = mgp.cpu.buffer.constant
// CHECK-NEXT: %[[IN_SPEC_2:.*]] = mgp.cpu.tensor_spec.create
// CHECK-NEXT: %[[OUT_SPEC_1:.*]] = mgp.cpu.tensor_spec.create
// CHECK-NEXT: %[[OUT_BUF_1:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 32
// CHECK-NEXT: %[[CH1:.*]] = "mop.kernels.mul"(%[[CH0]], %[[IN_BUF_1]], %[[IN_SPEC_1]], %[[IN_BUF_2]], %[[IN_SPEC_2]], %[[OUT_BUF_1]], %[[OUT_SPEC_1]])
// ** Don't alloc sub output till about to mrun it.
// CHECK-NEXT: %[[OUT_SPEC_2:.*]] = mgp.cpu.tensor_spec.create
// CHECK-NEXT: %[[OUT_BUF_2:.*]] = mgp.cpu.buffer.alloc.static %[[CH1]], 32
// ** Sub must wait for mul to finish
// CHECK-NEXT: %[[CH2:.*]] = "mop.kernels.sub"(%[[CH1]], %[[OUT_BUF_1]], %[[OUT_SPEC_1]], %[[IN_BUF_2]], %[[IN_SPEC_2]], %[[OUT_BUF_2]], %[[OUT_SPEC_2]])
// CHECK-NEXT: %[[OUT_SPEC_3:.*]] = mgp.cpu.tensor_spec.create
// ** Add must wait for mul and sub to finish.
// CHECK-NEXT: %[[CH3:.*]] = grt.chain.merge %[[CH1]], %[[CH2]]
// ** Don't alloc add output till about to run it.
// CHECK-NEXT: %[[OUT_BUF_3:.*]] = mgp.cpu.buffer.alloc.static %[[CH3]], 32
// CHECK-NEXT: %[[CH4:.*]] = "mop.kernels.add"(%[[CH3]], %[[OUT_BUF_2]], %[[OUT_SPEC_2]], %[[OUT_BUF_1]], %[[OUT_SPEC_1]], %[[OUT_BUF_3]], %[[OUT_SPEC_3]])
// CHECK-NEXT: %[[OUT_TENSOR:.*]] = mgp.cpu.tensor.create %[[CH4]], %[[OUT_BUF_3]], %[[OUT_SPEC_3]]
// CHECK-NEXT: mef.output %[[CH0]], %[[OUT_TENSOR]]

mgp.model @basicMoGraph : tf init {
  %0 = grt.chain.create
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %arg1: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<8xf32>) {
  %0 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
  %1 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %2 = mgp.cpu.tensor.create %ch0, %0, %1 : !mgp.cpu.tensor<8xf32>
  %3 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
  %4 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %5 = mgp.cpu.tensor.create %ch0, %3, %4 : !mgp.cpu.tensor<8xf32>
  %buf0, %spec0 = mgp.cpu.tensor.extract %2 : !mgp.cpu.tensor<8xf32>
  %buf1, %spec1 = mgp.cpu.tensor.extract %5 : !mgp.cpu.tensor<8xf32>
  %6 = mgp.cpu.buffer.alloc.static %ch0, 32
  %7 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %ch1 = "mop.kernels.mul"(%ch0, %buf0, %spec0, %buf1, %spec1, %6, %7) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %9 = mgp.cpu.tensor.create %ch1, %6, %7 : !mgp.cpu.tensor<8xf32>
  %buf2, %spec2 = mgp.cpu.tensor.extract %9 : !mgp.cpu.tensor<8xf32>
  %buf3, %spec3 = mgp.cpu.tensor.extract %5 : !mgp.cpu.tensor<8xf32>
  %10 = mgp.cpu.buffer.alloc.static %ch0, 32
  %11 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %ch2 = "mop.kernels.sub"(%ch0, %buf2, %spec2, %buf3, %spec3, %10, %11) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %13 = mgp.cpu.tensor.create %ch2, %10, %11 : !mgp.cpu.tensor<8xf32>
  %buf4, %spec4 = mgp.cpu.tensor.extract %13 : !mgp.cpu.tensor<8xf32>
  %buf5, %spec5 = mgp.cpu.tensor.extract %9 : !mgp.cpu.tensor<8xf32>
  %14 = mgp.cpu.buffer.alloc.static %ch0, 32
  %15 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %ch3 = "mop.kernels.add"(%ch0, %buf4, %spec4, %buf5, %spec5, %14, %15) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %17 = mgp.cpu.tensor.create %ch3, %14, %15 : !mgp.cpu.tensor<8xf32>
  mef.output %ch0, %17 : !mef.chain, !mgp.cpu.tensor<8xf32>
}

//
// Check any existing chain dependencies are preserved, both via tensor creates
// and via the op input chains.
//

// CHECK-LABEL: mgp.model @DivergentChains
// CHECK: } execute (%[[CH0:.*]]: !mef.chain
// CHECK-NEXT: %[[CH1:.*]] = grt.chain.create
// CHECK-NEXT: %[[CH2:.*]] = grt.chain.create
// CHECK-NEXT: %[[CH3:.*]] = grt.chain.create
// CHECK-NEXT: %[[CH4:.*]] = grt.chain.create

// CHECK: %[[CH5:.*]] = grt.chain.merge %[[CH3]], %[[CH4]]
// CHECK-NEXT: %[[OUT_BUF_1:.*]] = mgp.cpu.buffer.alloc.static %[[CH5]], 32
// CHECK-NEXT: %[[CH6:.*]] = "mop.kernels.mul"(%[[CH5]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[OUT_BUF_1]], %{{.*}})

// CHECK: %[[CH7:.*]] = grt.chain.merge %[[CH1]], %[[CH4]], %[[CH6]]
// CHECK-NEXT: %[[OUT_BUF_2:.*]] = mgp.cpu.buffer.alloc.static %[[CH7]], 32
// CHECK-NEXT: %[[CH8:.*]] = "mop.kernels.sub"(%[[CH7]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[OUT_BUF_2]], %{{.*}})

// CHECK: %[[CH9:.*]] = grt.chain.merge %[[CH1]], %[[CH2]], %[[CH4]], %[[CH8]]
// CHECK-NEXT: %[[OUT_BUF_3:.*]] = mgp.cpu.buffer.alloc.static %[[CH9]], 32
// CHECK-NEXT: %[[CH10:.*]] = "mop.kernels.add"(%[[CH9]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[OUT_BUF_3]], %{{.*}})
// CHECK-NEXT: %[[OUT_TENSOR:.*]] = mgp.cpu.tensor.create %[[CH10]], %18, %16 : !mgp.cpu.tensor<8xf32>
// CHECK-NEXT: mef.output %[[CH10]], %[[OUT_TENSOR]]

mgp.model @DivergentChains : tf init {
  %0 = grt.chain.create
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<8xf32>) {
  %ch1 = grt.chain.create
  %ch2 = grt.chain.create
  %ch3 = grt.chain.create
  %ch4 = grt.chain.create
  %0 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
  %1 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %2 = mgp.cpu.tensor.create %ch3, %0, %1 : !mgp.cpu.tensor<8xf32>
  %3 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
  %4 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %5 = mgp.cpu.tensor.create %ch4, %3, %4 : !mgp.cpu.tensor<8xf32>
  %buf1, %spec1 = mgp.cpu.tensor.extract %2 : !mgp.cpu.tensor<8xf32>
  %buf2, %spec2 = mgp.cpu.tensor.extract %5 : !mgp.cpu.tensor<8xf32>
  %6 = mgp.cpu.buffer.alloc.static %arg0, 32
  %7 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %8 = "mop.kernels.mul"(%arg0, %buf1, %spec1, %buf2, %spec2, %6, %7) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %9 = mgp.cpu.tensor.create %8, %6, %7 : !mgp.cpu.tensor<8xf32>
  %buf3, %spec3 = mgp.cpu.tensor.extract %9 : !mgp.cpu.tensor<8xf32>
  %buf4, %spec4 = mgp.cpu.tensor.extract %5 : !mgp.cpu.tensor<8xf32>
  %10 = mgp.cpu.buffer.alloc.static %arg0, 32
  %11 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %12 = "mop.kernels.sub"(%ch1, %buf3, %spec3, %buf4, %spec4, %10, %11) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %13 = mgp.cpu.tensor.create %12, %10, %11 : !mgp.cpu.tensor<8xf32>
  %buf5, %spec5 = mgp.cpu.tensor.extract %13 : !mgp.cpu.tensor<8xf32>
  %buf6, %spec6 = mgp.cpu.tensor.extract %5 : !mgp.cpu.tensor<8xf32>
  %14 = mgp.cpu.buffer.alloc.static %arg0, 32
  %15 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %merged = grt.chain.merge %ch1, %ch2
  %16 = "mop.kernels.add"(%merged, %buf5, %spec5, %buf6, %spec6, %14, %15) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %17 = mgp.cpu.tensor.create %16, %14, %15 : !mgp.cpu.tensor<8xf32>
  mef.output %16, %17 : !mef.chain, !mgp.cpu.tensor<8xf32>
}
