// RUN: mo-opt %s --mlir-disable-threading --allow-unregistered-dialect --async-dataflow-dependencies-test --check-buffer-read-write | FileCheck %s

//
// Simple graph with no control flow
//

// For reference here's the model with vars matching the checks:
// CHECK-LABEL: "mgp.model"() ({
// ### Block 0 ###
// CHECK:   %[[SSA48:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA49:.*]] = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %[[SSA51:.*]] = "mgp.runtime.create"(%[[SSA48]], %[[SSA49]])
// CHECK:   "mef.output"(%[[SSA51]], %[[SSA49]]
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%[[arg4:[a-z0-9]*]]: !mef.chain, %[[arg5:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA42:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA43:.*]] = "some.op"(%[[arg4]]) : (!mef.chain) -> !mef.chain
// CHECK:   %[[SSA44:.*]] = "some.op"(%[[SSA43]], %[[SSA42]]) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %[[SSA45:.*]] = "some.op"(%[[arg4]]) : (!mef.chain) -> !mef.chain
// CHECK:   %[[SSA46:.*]] = "some.op"(%[[SSA45]], %[[SSA42]]) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %[[SSA47:.*]] = "some.op"(%[[SSA44]], %[[SSA46]]) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   "mef.output"(%[[SSA47]]) : (!mef.chain) -> ()
// CHECK: })

// sameAs only ties the 'init' results to the 'execute args
// CHECK: sameAs begin
// CHECK:   %[[arg4]] |-> {%[[SSA51]][0]}
// CHECK:   %[[arg5]] |-> {%[[SSA49]][0]}
// CHECK: sameAs end

// maybeSameAs == sameAs
// CHECK: maybeSameAs begin
// CHECK:   %[[arg4]] |-> {%[[SSA51]][0]}
// CHECK:   %[[arg5]] |-> {%[[SSA49]][0]}
// CHECK: maybeSameAs end

// The two always ready chains (this is up to sameAs equivalence)
// CHECK: isAlwaysReady({%[[SSA48]][0], %[[SSA42]]})

// impliesReady captures transitive operand dependencies (plus identity)
// CHECK: impliesReady begin
// CHECK:   %[[SSA51]][0] |-> {%[[SSA48]][0], %[[SSA49]][0]}
// CHECK:   %[[arg4]] |-> {%[[SSA48]][0], %[[SSA49]][0], %[[SSA51]][0]}
// CHECK:   %[[arg5]] |-> {%[[SSA49]][0]}
// CHECK:   %[[SSA43]] |-> {%[[SSA48]][0], %[[SSA49]][0], %[[SSA51]][0], %[[arg4]]}
// CHECK:   %[[SSA44]] |-> {%[[SSA48]][0], %[[SSA49]][0], %[[SSA51]][0], %[[arg4]], %[[SSA42]], %[[SSA43]]}
// CHECK:   %[[SSA45]] |-> {%[[SSA48]][0], %[[SSA49]][0], %[[SSA51]][0], %[[arg4]]}
// CHECK:   %[[SSA46]] |-> {%[[SSA48]][0], %[[SSA49]][0], %[[SSA51]][0], %[[arg4]], %[[SSA42]], %[[SSA45]]}
// CHECK:   %[[SSA47]] |-> {%[[SSA48]][0], %[[SSA49]][0], %[[SSA51]][0], %[[arg4]], %[[SSA42]], %[[SSA43]], %[[SSA44]], %[[SSA45]], %[[SSA46]]}
// CHECK: impliesReady end

// No control flow so nothing in these relations.
// CHECK: structurallyAtMostOne begin
// CHECK: structurallyAtMostOne end
// CHECK: structurallyCannotOverlapRelation begin
// CHECK: structurallyCannotOverlapRelation end

// runsLater for which ops (identified by their result values) run after which
// CHECK: runsLater begin
// CHECK:   %[[SSA48]][0] |-> {%[[SSA51]][0], %[[SSA43]], %[[SSA44]], %[[SSA45]], %[[SSA46]], %[[SSA47]]}
// CHECK:   %[[SSA49]][0] |-> {%[[SSA51]][0], %[[SSA43]], %[[SSA44]], %[[SSA45]], %[[SSA46]], %[[SSA47]]}
// CHECK:   %[[SSA51]][0] |-> {%[[SSA43]], %[[SSA44]], %[[SSA45]], %[[SSA46]], %[[SSA47]]}
// CHECK:   %[[SSA42]] |-> {%[[SSA44]], %[[SSA46]], %[[SSA47]]}
// CHECK:   %[[SSA43]] |-> {%[[SSA44]], %[[SSA47]]}
// CHECK:   %[[SSA44]] |-> {%[[SSA47]]}
// CHECK:   %[[SSA45]] |-> {%[[SSA46]], %[[SSA47]]}
// CHECK:   %[[SSA46]] |-> {%[[SSA47]]}
// CHECK: runsLater end

// exclusive tells us which values can't have overlapping lifetimes.
// CHECK: exclusive begin
// CHECK:   %[[SSA48]][0] |-> {%[[SSA43]], %[[SSA44]], %[[SSA45]], %[[SSA46]], %[[SSA47]]}
// CHECK:   %[[SSA44]] |-> {%[[SSA48]][0]}
// CHECK:   %[[SSA45]] |-> {%[[SSA48]][0], %[[SSA47]]}
// CHECK:   %[[SSA46]] |-> {%[[SSA48]][0]}
// CHECK:   %[[SSA47]] |-> {%[[SSA48]][0], %[[SSA42]], %[[SSA43]], %[[SSA45]]}
// CHECK: exclusive end

mgp.model @simple : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">
} execute (%ch0: !mef.chain, %cxt: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch1 = grt.chain.create ()
  %ch2 = "some.op"(%ch0) : (!mef.chain) -> !mef.chain
  %ch3 = "some.op"(%ch2, %ch1) : (!mef.chain, !mef.chain) -> !mef.chain
  %ch4 = "some.op"(%ch0) : (!mef.chain) -> !mef.chain
  %ch5 = "some.op"(%ch4, %ch1) : (!mef.chain, !mef.chain) -> !mef.chain
  %ch6 = "some.op"(%ch3, %ch5) : (!mef.chain, !mef.chain) -> !mef.chain
  mef.output %ch6 : !mef.chain
}

//
// Realistic graph with no control flow.
//

// For reference here's the model with vars matching the checks:
// CHECK-LABEL: "mgp.model"() ({
// ### Block 0 ###
// CHECK:   %[[SSA38:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA39:.*]] = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %[[SSA41:.*]] = "mgp.runtime.create"(%[[SSA38]], %[[SSA39]])
// CHECK:   "mef.output"(%[[SSA41]], %[[SSA39]]
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%[[arg2:[a-z0-9]*]]: !mef.chain, %[[arg3:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA13:.*]] = "mgp.chain.host_to_device"(%[[arg2]], %[[arg3]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mgp.chain
// CHECK:   %[[SSA14:.*]] = "mgp.buffer.constant"(%[[arg2]], %[[arg3]]) {aRuntimeSlot = 1 : ui64, bValue = #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32> : !M.array<32xui8>, cRawAlign = 32 : ui64, dDevice = #M.device_ref<"cpu", 0>, eIsTrivial = true} : (!mef.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %[[SSA15:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 66 : ui8} : () -> !mgp.tensor_spec<8xf32>
// CHECK:   %[[SSA16:.*]] = "mgp.tensor.create"(%[[SSA13]], %[[arg3]], %[[SSA14]], %[[SSA15]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mgp.tensor<8xf32>
// CHECK:   %[[SSA17:.*]] = "mgp.buffer.constant"(%[[arg2]], %[[arg3]]) {aRuntimeSlot = 1 : ui64, bValue = #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32> : !M.array<32xui8>, cRawAlign = 32 : ui64, dDevice = #M.device_ref<"cpu", 0>, eIsTrivial = true} : (!mef.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %[[SSA18:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 66 : ui8} : () -> !mgp.tensor_spec<8xf32>
// CHECK:   %[[SSA19:.*]] = "mgp.tensor.create"(%[[SSA13]], %[[arg3]], %[[SSA17]], %[[SSA18]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mgp.tensor<8xf32>
// CHECK:   %[[SSA20:.*]] = "mgp.tensor.extract.buffer"(%[[arg3]], %[[SSA16]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<8xf32>) -> !mgp.buffer<32>
// CHECK:   %[[SSA21:.*]] = "mgp.tensor.extract.buffer"(%[[arg3]], %[[SSA19]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<8xf32>) -> !mgp.buffer<32>
// CHECK:   %[[SSA22:.*]] = "mgp.buffer.alloc.static"(%[[SSA13]], %[[arg3]]) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %[[SSA23:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 66 : ui8} : () -> !mgp.tensor_spec<8xf32>
// CHECK:   %[[SSA24:.*]] = "mul"(%[[SSA13]], %[[SSA20]], %[[SSA21]], %[[SSA22]]) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
// CHECK:   %[[SSA25:.*]] = "mgp.tensor.create"(%[[SSA24]], %[[arg3]], %[[SSA22]], %[[SSA23]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mgp.tensor<8xf32>
// CHECK:   %[[SSA26:.*]] = "mgp.tensor.extract.buffer"(%[[arg3]], %[[SSA25]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<8xf32>) -> !mgp.buffer<32>
// CHECK:   %[[SSA27:.*]] = "mgp.tensor.extract.buffer"(%[[arg3]], %[[SSA19]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<8xf32>) -> !mgp.buffer<32>
// CHECK:   %[[SSA28:.*]] = "mgp.buffer.alloc.static"(%[[SSA13]], %[[arg3]]) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %[[SSA29:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 66 : ui8} : () -> !mgp.tensor_spec<8xf32>
// CHECK:   %[[SSA30:.*]] = "sub"(%[[SSA13]], %[[SSA26]], %[[SSA27]], %[[SSA28]]) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
// CHECK:   %[[SSA31:.*]] = "mgp.tensor.create"(%[[SSA30]], %[[arg3]], %[[SSA28]], %[[SSA29]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mgp.tensor<8xf32>
// CHECK:   %[[SSA32:.*]] = "mgp.tensor.extract.buffer"(%[[arg3]], %[[SSA31]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<8xf32>) -> !mgp.buffer<32>
// CHECK:   %[[SSA33:.*]] = "mgp.tensor.extract.buffer"(%[[arg3]], %[[SSA25]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<8xf32>) -> !mgp.buffer<32>
// CHECK:   %[[SSA34:.*]] = "mgp.buffer.alloc.static"(%[[SSA13]], %[[arg3]]) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %[[SSA35:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 66 : ui8} : () -> !mgp.tensor_spec<8xf32>
// CHECK:   %[[SSA36:.*]] = "add"(%[[SSA13]], %[[SSA32]], %[[SSA33]], %[[SSA34]]) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
// CHECK:   %[[SSA37:.*]] = "mgp.tensor.create"(%[[SSA36]], %[[arg3]], %[[SSA34]], %[[SSA35]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mgp.tensor<8xf32>
// CHECK:   "mef.output"(%[[arg2]], %[[SSA37]]) : (!mef.chain, !mgp.tensor<8xf32>) -> ()
// CHECK: })

// Spot check impliesReady
// CHECK: impliesReady begin
// CHECK:   %[[SSA24]] |-> {%[[SSA38]][0], %[[SSA39]][0], %[[SSA41]][0], %[[arg2]], %[[arg3]], %[[SSA13]], %[[SSA14]], %[[SSA15]], %[[SSA16]], %[[SSA17]], %[[SSA18]], %[[SSA19]], %[[SSA20]], %[[SSA21]], %[[SSA22]]}
// CHECK: impliesReady end

// Spot check runsLater
// CHECK: runsLater begin
// CHECK:   %[[SSA26]] |-> {%[[SSA30]], %[[SSA31]], %[[SSA32]], %[[SSA36]], %[[SSA37]]}
// CHECK:   %[[SSA27]] |-> {%[[SSA30]], %[[SSA31]], %[[SSA32]], %[[SSA36]], %[[SSA37]]}
// CHECK:   %[[SSA28]] |-> {%[[SSA30]], %[[SSA31]], %[[SSA32]], %[[SSA36]], %[[SSA37]]}
// CHECK: runsLater end

// CHECK: exclusive begin
// CHECK:   %[[SSA17]] |-> {%[[SSA38]][0], %[[SSA31]], %[[SSA32]], %[[SSA36]], %[[SSA37]]}
// CHECK: exclusive end

mgp.model @straightline : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, !mgp.tensor<8xf32>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %0 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32> : !M.array<32xui8>
  %1 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %2 = mgp.tensor.create %ch1, %ctx[1],  %0, %1  : !mgp.tensor<8xf32>
  %3 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32> : !M.array<32xui8>
  %4 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %5 = mgp.tensor.create %ch1, %ctx[1],  %3, %4  : !mgp.tensor<8xf32>
  %bufferResult = mgp.tensor.extract.buffer %ctx[1], %2 : !mgp.tensor<8xf32>
  %bufferResult_0 = mgp.tensor.extract.buffer %ctx[1], %5 : !mgp.tensor<8xf32>
  %6 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %7 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %8 = "mul"(%ch1, %bufferResult, %bufferResult_0, %6) { tempInOutSig=#M.inout_sig<"iiio"> } : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %9 = mgp.tensor.create %8, %ctx[1],  %6, %7  : !mgp.tensor<8xf32>
  %bufferResult_2 = mgp.tensor.extract.buffer %ctx[1], %9 : !mgp.tensor<8xf32>
  %bufferResult_4 = mgp.tensor.extract.buffer %ctx[1], %5 : !mgp.tensor<8xf32>
  %10 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %11 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %12 = "sub"(%ch1, %bufferResult_2, %bufferResult_4, %10) { tempInOutSig=#M.inout_sig<"iiio"> } : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %13 = mgp.tensor.create %12, %ctx[1],  %10, %11  : !mgp.tensor<8xf32>
  %bufferResult_6 = mgp.tensor.extract.buffer %ctx[1], %13 : !mgp.tensor<8xf32>
  %bufferResult_8 = mgp.tensor.extract.buffer %ctx[1], %9 : !mgp.tensor<8xf32>
  %14 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %15 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %16 = "add"(%ch1, %bufferResult_6, %bufferResult_8, %14) { tempInOutSig=#M.inout_sig<"iiio"> } : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %17 = mgp.tensor.create %16, %ctx[1],  %14, %15  : !mgp.tensor<8xf32>
  mef.output %ch0, %17 : !mef.chain, !mgp.tensor<8xf32>
}

//
// mgp.buffer.split.borrow implicitly takes reference to argument
//

// For reference here's the model with vars matching the checks:
// CHECK-LABEL: "mgp.model"() ({
// CHECK:   %[[SSA9:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA10:.*]] = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %[[SSA12:.*]] = "mgp.runtime.create"(%[[SSA9]], %[[SSA10]])
// CHECK:   "mef.output"(%[[SSA12]], %[[SSA10]]
// CHECK: }, {
// CHECK: ^bb0(%[[arg0:[a-z0-9]*]]: !mef.chain, %[[arg1:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA0:.*]] = "mgp.chain.host_to_device"(%[[arg0]], %[[arg1]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mgp.chain
// CHECK:   %[[SSA1:.*]] = "mgp.buffer.alloc.static"(%[[SSA0]], %[[arg1]]) {aRuntimeSlot = 1 : ui64, bSize = 6 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<6>
// CHECK:   %[[SSA2:.*]] = "write_to_buffer"(%[[SSA1]]) {tempInOutSig = #M.inout_sig<"o">} : (!mgp.buffer<6>) -> !mgp.chain
// CHECK:   %[[SSA3:.*]]:2 = "mgp.buffer.split.borrow"(%[[SSA2]], %[[arg1]], %[[SSA1]]) {aRuntimeSlot = 1 : ui64, bOutBufferRawSizes = #M.dense_array<2, 4> : !M.array<2xui64>, cDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<6>) -> (!mgp.buffer<2>, !mgp.buffer<4>)
// CHECK:   %[[SSA4:.*]] = "mgp.buffer.alloc.static"(%[[SSA2]], %[[arg1]]) {aRuntimeSlot = 1 : ui64, bSize = 6 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<6>
// CHECK:   %[[SSA5:.*]] = "concat_buffers"(%[[SSA2]], %[[SSA4]], %[[SSA3]]#0, %[[SSA3]]#1) {tempInOutSig = #M.inout_sig<"ioii">} : (!mgp.chain, !mgp.buffer<6>, !mgp.buffer<2>, !mgp.buffer<4>) -> !mgp.chain
// CHECK:   %[[SSA6:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<6> : !M.array<1xui64>, bRawDType = 135 : ui8} : () -> !mgp.tensor_spec<6xsi8>
// CHECK:   %[[SSA7:.*]] = "mgp.tensor.create"(%[[SSA5]], %[[arg1]], %[[SSA4]], %[[SSA6]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<6>, !mgp.tensor_spec<6xsi8>) -> !mgp.tensor<6xsi8>
// CHECK:   %[[SSA8:.*]] = "mgp.chain.device_to_host"(%[[SSA5]], %[[arg1]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%[[SSA8]], %[[SSA7]]) : (!mef.chain, !mgp.tensor<6xsi8>) -> ()
// CHECK: })

// CHECK: maybeSameAs begin
// Note %3#1 is considered maybe the same as %1.
// CHECK:   %[[SSA3]]#0 |-> {%[[SSA1]]}
// CHECK:   %[[SSA3]]#1 |-> {%[[SSA1]]}
// CHECK: maybeSameAs end

// CHECK: exclusive begin
// Note %1 is only exclusive with %7 because it is implicitly lifetime
// extended by %3#{0,1}.
// CHECK:   %[[SSA1]] |-> {%[[SSA9]][0], %[[SSA7]], %[[SSA8]]}
// CHECK: exclusive end

mgp.model @safe_split_borrow : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, !mgp.tensor<6xsi8>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %in_buf = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<6>
  %ch2 = "write_to_buffer"(%in_buf) {tempInOutSig = #M.inout_sig<"o">} : (!mgp.buffer<6>) -> !mgp.chain
  %in_buf_borrows:2 = mgp.buffer.split.borrow %ch2, %ctx[1], %in_buf : !mgp.buffer<6> -> (!mgp.buffer<2>, !mgp.buffer<4>)
  %out_buf = mgp.buffer.alloc.static %ch2, %ctx[1] : !mgp.buffer<6>
  %ch3 = "concat_buffers"(%ch2, %out_buf, %in_buf_borrows#0, %in_buf_borrows#1) {tempInOutSig = #M.inout_sig<"ioii">} : (!mgp.chain, !mgp.buffer<6>, !mgp.buffer<2>, !mgp.buffer<4>) -> !mgp.chain
  %out_spec = mgp.tensor_spec.create : !mgp.tensor_spec<6xsi8>
  %ten = mgp.tensor.create %ch3, %ctx[1], %out_buf, %out_spec : !mgp.tensor<6xsi8>
  %ch4 = mgp.chain.device_to_host %ch3, %ctx[1]
  mef.output %ch4, %ten : !mef.chain, !mgp.tensor<6xsi8>
}
