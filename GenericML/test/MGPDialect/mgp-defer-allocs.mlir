// RUN: mo-opt %s --allow-unregistered-dialect --remove-redundant-tensor-extracts --defer-allocs --simplify-chains -cse | FileCheck %s

// Check mgp.buffer.alloc.* ops are deferred until the writing op is ready.

// Original graph:
// mo.graph @basicMoGraph() -> (!mo.tensor<[8], f32>) {
//  %const0 = "mo.constant"() {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<8xf32>} : () -> !mo.tensor<[8],f32>
//  %const1 = "mo.constant"() {value = #M.dense_array<10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0> : tensor<8xf32>} : () -> !mo.tensor<[8], f32>  %1 = mo.mul(%iota, %tens) : !mo.tensor<[8], f32>
//  %x = mo.mul(%const0, %const1) : !mo.tensor<[8], f32>
//  %y = mo.sub(%x, %const1) : !mo.tensor<[8], f32>
//  %r = mo.add(%y, %x) : !mo.tensor<[8], f32>
//  mo.output %r : !mo.tensor<[8], f32>
//}

// CHECK-LABEL: mgp.model @basic : mof
// CHECK: execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context

// ** No change to mul output allocation.
// CHECK: %[[CH1:.*]] = mgp.chain.host_to_device %[[CH0]]
// CHECK: %[[OUT_BUF_1:.*]] = mgp.buffer.alloc.static %[[CH1]]
// CHECK-NEXT: %[[CH2:.*]] = "mul"(%[[CH1]], %{{.*}}, %{{.*}}, %[[OUT_BUF_1]])

// ** Don't alloc sub output till about to run it.
// CHECK: %[[OUT_BUF_2:.*]] = mgp.buffer.alloc.static %[[CH2]]
// ** Sub must wait for mul to finish
// CHECK-NEXT: %[[CH3:.*]] = "sub"(%[[CH2]], %[[OUT_BUF_1]], %{{.*}}, %[[OUT_BUF_2]])

// ** Don't alloc add output till about to run it.
// CHECK: %[[OUT_BUF_3:.*]] = mgp.buffer.alloc.static %[[CH3]]
// CHECK-NEXT: %[[CH4:.*]] = "add"(%[[CH3]], %[[OUT_BUF_2]], %[[OUT_BUF_1]], %[[OUT_BUF_3]])
// CHECK-NEXT: %[[OUT_TENSOR:.*]] = mgp.tensor.create %[[CH4]], %[[CTX]][1], %[[OUT_BUF_3]], %{{.*}}
// CHECK-NEXT: mef.output %[[CH0]], %[[OUT_TENSOR]]
// CHECK-NEXT: }

mgp.model @basic : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, !mgp.tensor<8xf32>) {
  %0 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
  %spec = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %2 = mgp.tensor.create %ch1, %ctx[1], %0, %spec : !mgp.tensor<8xf32>
  %3 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
  %5 = mgp.tensor.create %ch1, %ctx[1], %3, %spec : !mgp.tensor<8xf32>
  %buf0 = mgp.tensor.extract.buffer %ctx[1], %2 : !mgp.tensor<8xf32>
  %buf1 = mgp.tensor.extract.buffer %ctx[1], %5 : !mgp.tensor<8xf32>
  %6 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %ch2 = "mul"(%ch1, %buf0, %buf1, %6) { tempInOutSig=#M.inout_sig<"iiio"> } : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %9 = mgp.tensor.create %ch2, %ctx[1], %6, %spec : !mgp.tensor<8xf32>
  %buf2 = mgp.tensor.extract.buffer %ctx[1], %9 : !mgp.tensor<8xf32>
  %buf3 = mgp.tensor.extract.buffer %ctx[1], %5 : !mgp.tensor<8xf32>
  %10 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %ch3 = "sub"(%ch1, %buf2, %buf3, %10) { tempInOutSig=#M.inout_sig<"iiio"> } : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %13 = mgp.tensor.create %ch3, %ctx[1], %10, %spec : !mgp.tensor<8xf32>
  %buf4 = mgp.tensor.extract.buffer %ctx[1], %13 : !mgp.tensor<8xf32>
  %buf5 = mgp.tensor.extract.buffer %ctx[1], %9 : !mgp.tensor<8xf32>
  %14 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %ch4 = "add"(%ch1, %buf4, %buf5, %14) { tempInOutSig=#M.inout_sig<"iiio"> } : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %17 = mgp.tensor.create %ch4, %ctx[1], %14, %spec : !mgp.tensor<8xf32>
  mef.output %ch0, %17 : !mef.chain, !mgp.tensor<8xf32>
}

//
// Check any existing chain dependencies are preserved, both via tensor creates
// and via the op input chains.
//

// CHECK-LABEL: mgp.model @divergent
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// CHECK-NEXT: %[[CHX:.*]] = mgp.chain.host_to_device %[[CH0]]
// CHECK-NEXT: %[[CH1:.*]] = "some.op"()
// CHECK-NEXT: %[[CH2:.*]] = "some.op"()
// CHECK-NEXT: %[[CH3:.*]] = "some.op"()
// CHECK-NEXT: %[[CH4:.*]] = "some.op"()

// CHECK: %[[CH5:.*]] = mgp.chain.create %[[CTX]][1], (%[[CHX]], %[[CH3]], %[[CH4]])
// CHECK-NEXT: %[[OUT_BUF_1:.*]] = mgp.buffer.alloc.static %[[CH5]]
// CHECK-NEXT: %[[CH6:.*]] = "mul"(%[[CH5]], %{{.*}}, %{{.*}}, %[[OUT_BUF_1]])

// CHECK: %[[CH7:.*]] = mgp.chain.create %[[CTX]][1], (%[[CH1]], %[[CH6]])
// CHECK-NEXT: %[[OUT_BUF_2:.*]] = mgp.buffer.alloc.static %[[CH7]]
// CHECK-NEXT: %[[CH8:.*]] = "sub"(%[[CH7]], %[[OUT_BUF_1]], %{{.*}}, %[[OUT_BUF_2]])

// CHECK: %[[CH9:.*]] = mgp.chain.create %[[CTX]][1], (%[[CH2]], %[[CH8]])
// CHECK-NEXT: %[[OUT_BUF_3:.*]] = mgp.buffer.alloc.static %[[CH9]]
// CHECK-NEXT: %[[CH10:.*]] = "add"(%[[CH9]], %[[OUT_BUF_2]], %[[OUT_BUF_1]], %[[OUT_BUF_3]])
// CHECK-NEXT: %[[OUT_TENSOR:.*]] = mgp.tensor.create %[[CH10]], %[[CTX]][1], %[[OUT_BUF_3]], %{{.*}}
// CHECK-NEXT: %[[CH11:.*]] = mgp.chain.device_to_host %[[CH10]]
// CHECK-NEXT: mef.output %[[CH11]], %[[OUT_TENSOR]]
// CHECK-NEXT: }

mgp.model @divergent : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, !mgp.tensor<8xf32>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %ch2 = "some.op"() : () -> !mgp.chain
  %ch3 = "some.op"() : () -> !mgp.chain
  %ch4 = "some.op"() : () -> !mgp.chain
  %ch5 = "some.op"() : () -> !mgp.chain
  %0 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
  %spec = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %2 = mgp.tensor.create %ch4, %ctx[1], %0, %spec : !mgp.tensor<8xf32>
  %3 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
  %5 = mgp.tensor.create %ch5, %ctx[1], %3, %spec : !mgp.tensor<8xf32>
  %buf0 = mgp.tensor.extract.buffer %ctx[1], %2 : !mgp.tensor<8xf32>
  %buf1 = mgp.tensor.extract.buffer %ctx[1], %5 : !mgp.tensor<8xf32>
  %6 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %ch6 = "mul"(%ch1, %buf0, %buf1, %6) { tempInOutSig=#M.inout_sig<"iiio"> } : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %9 = mgp.tensor.create %ch6, %ctx[1], %6, %spec : !mgp.tensor<8xf32>
  %buf2 = mgp.tensor.extract.buffer %ctx[1], %9 : !mgp.tensor<8xf32>
  %buf3 = mgp.tensor.extract.buffer %ctx[1], %5 : !mgp.tensor<8xf32>
  %10 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %ch7 = "sub"(%ch2, %buf2, %buf3, %10) { tempInOutSig=#M.inout_sig<"iiio"> } : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %13 = mgp.tensor.create %ch7, %ctx[1], %10, %spec : !mgp.tensor<8xf32>
  %buf4 = mgp.tensor.extract.buffer %ctx[1], %13 : !mgp.tensor<8xf32>
  %buf5 = mgp.tensor.extract.buffer %ctx[1], %9 : !mgp.tensor<8xf32>
  %14 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %ch8 = mgp.chain.create %ctx[1], (%ch2, %ch3)
  %ch9 = "add"(%ch8, %buf4, %buf5, %14) { tempInOutSig=#M.inout_sig<"iiio"> } : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %17 = mgp.tensor.create %ch9, %ctx[1], %14, %spec : !mgp.tensor<8xf32>
  %ch10 = mgp.chain.device_to_host %ch9, %ctx[1]
  mef.output %ch10, %17 : !mef.chain, !mgp.tensor<8xf32>
}

// Simply testing for algorithm convergence (in presence of alloc reordering)
// CHECK-LABEL: mgp.model @multiple_writers_convergence
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// CHECK-NEXT: %[[CHX:.*]] = mgp.chain.host_to_device %[[CH0]]
// CHECK-NEXT: %[[BUF0:.*]] = mgp.buffer.alloc.static %[[CHX]], %[[CTX]][1], align 8
// CHECK-NEXT: %[[BUF1:.*]] = mgp.buffer.alloc.static %[[CHX]], %[[CTX]][1], align 16
// CHECK-NEXT: %[[CH1:.*]] = "test.write_bufs"(%[[CHX]], %[[BUF0]], %[[BUF1]])
// CHECK-NEXT: %[[CHO:.*]] = mgp.chain.device_to_host %[[CH1]]
// CHECK-NEXT: mef.output %[[CHO]], %[[BUF0]], %[[BUF1]]
// CHECK-NEXT: }

mgp.model @multiple_writers_convergence : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, !mgp.buffer<8>, !mgp.buffer<8>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %buf0 = mgp.buffer.alloc.static %ch1, %ctx[1], align 8 : !mgp.buffer<8>
  %buf1 = mgp.buffer.alloc.static %ch1, %ctx[1], align 16 : !mgp.buffer<8>
  %ch2 = "test.write_bufs"(%ch1, %buf0, %buf1) { tempInOutSig=#M.inout_sig<"ioo"> }
           : (!mgp.chain, !mgp.buffer<8>, !mgp.buffer<8>) -> !mgp.chain
  %ch3 = mgp.chain.device_to_host %ch2, %ctx[1]
  mef.output %ch3, %buf0, %buf1 : !mef.chain, !mgp.buffer<8>, !mgp.buffer<8>
}
