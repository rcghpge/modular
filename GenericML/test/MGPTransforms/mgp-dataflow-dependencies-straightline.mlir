// RUN: mo-opt %s --mlir-disable-threading --allow-unregistered-dialect --async-dataflow-dependencies-test | FileCheck %s

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

mgp.model @simple : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
} execute (%ch0: !mef.chain, %cxt: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
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
// CHECK:   %[[SSA0:.*]] = "mgp.buffer.constant"(%[[arg2]], %[[arg3]]) {aRuntimeSlot = 1 : ui64, bValue = #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32> : !M.array<32xui8>, cRawAlign = 32 : ui64, dDevice = #M.device_ref<"cpu", 0>, eIsTrivial = true} : (!mef.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %[[SSA1:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 69 : ui8} : () -> !mgp.tensor_spec<8xf32>
// CHECK:   %[[SSA2:.*]] = "mgp.tensor.create"(%[[arg2]], %[[arg3]], %[[SSA0]], %[[SSA1]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mgp.tensor<8xf32>
// CHECK:   %[[SSA3:.*]] = "mgp.buffer.constant"(%[[arg2]], %[[arg3]]) {aRuntimeSlot = 1 : ui64, bValue = #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32> : !M.array<32xui8>, cRawAlign = 32 : ui64, dDevice = #M.device_ref<"cpu", 0>, eIsTrivial = true} : (!mef.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %[[SSA4:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 69 : ui8} : () -> !mgp.tensor_spec<8xf32>
// CHECK:   %[[SSA5:.*]] = "mgp.tensor.create"(%[[arg2]], %[[arg3]], %[[SSA3]], %[[SSA4]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mgp.tensor<8xf32>
// CHECK:   %[[SSA6:.*]] = "mgp.tensor.extract.buffer"
// CHECK:   %[[SSA7:.*]] = "mgp.tensor.extract.buffer"
// CHECK:   %[[SSA8:.*]] = "index.constant"() <{value = 32 : index}> : () -> index
// CHECK:   %[[SSA9:.*]] = "mgp.buffer.alloc"(%[[arg2]], %[[arg3]], %[[SSA8]], {{.*}}) {aRuntimeSlot = 1 : ui64, bRawAlign = 18446744073709551615 : ui64, cDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context, index, !mgp.opaque<"cpu0_context">) -> !mgp.buffer<32>
// CHECK:   %[[SSA10:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 69 : ui8} : () -> !mgp.tensor_spec<8xf32>
// CHECK:   %[[SSA11:.*]] = "mul"(%[[arg2]], %[[SSA6]], %[[SSA7]], %[[SSA9]]) {tempInOutSig = #M.inout_sig<"iiio">} : (!mef.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mef.chain
// CHECK:   %[[SSA12:.*]] = "mgp.tensor.create"(%[[SSA11]], %[[arg3]], %[[SSA9]], %[[SSA10]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mgp.tensor<8xf32>
// CHECK:   %[[SSA13:.*]] = "mgp.tensor.extract.buffer"
// CHECK:   %[[SSA14:.*]] = "mgp.tensor.extract.buffer"
// CHECK:   %[[SSA15:.*]] = "mgp.buffer.alloc"(%[[arg2]], %[[arg3]], %[[SSA8]], {{.*}}) {aRuntimeSlot = 1 : ui64, bRawAlign = 18446744073709551615 : ui64, cDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context, index, !mgp.opaque<"cpu0_context">) -> !mgp.buffer<32>
// CHECK:   %[[SSA16:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 69 : ui8} : () -> !mgp.tensor_spec<8xf32>
// CHECK:   %[[SSA17:.*]] = "sub"(%[[arg2]], %[[SSA13]], %[[SSA14]], %[[SSA15]]) {tempInOutSig = #M.inout_sig<"iiio">} : (!mef.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mef.chain
// CHECK:   %[[SSA18:.*]] = "mgp.tensor.create"(%[[SSA17]], %[[arg3]], %[[SSA15]], %[[SSA16]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mgp.tensor<8xf32>
// CHECK:   %[[SSA19:.*]] = "mgp.tensor.extract.buffer"
// CHECK:   %[[SSA20:.*]] = "mgp.tensor.extract.buffer"
// CHECK:   %[[SSA21:.*]] = "mgp.buffer.alloc"(%[[arg2]], %[[arg3]], %[[SSA8]], {{.*}}) {aRuntimeSlot = 1 : ui64, bRawAlign = 18446744073709551615 : ui64, cDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context, index, !mgp.opaque<"cpu0_context">) -> !mgp.buffer<32>
// CHECK:   %[[SSA22:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 69 : ui8} : () -> !mgp.tensor_spec<8xf32>
// CHECK:   %[[SSA23:.*]] = "add"(%[[arg2]], %[[SSA19]], %[[SSA20]], %[[SSA21]]) {tempInOutSig = #M.inout_sig<"iiio">} : (!mef.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mef.chain
// CHECK:   %[[SSA24:.*]] = "mgp.tensor.create"(%[[SSA23]], %[[arg3]], %[[SSA21]], %[[SSA22]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mgp.tensor<8xf32>
// CHECK:   "mef.output"(%[[arg2]], %[[SSA24]]) : (!mef.chain, !mgp.tensor<8xf32>) -> ()
// CHECK: })

// Spot check impliesReady
// CHECK: impliesReady begin
// CHECK:   %[[SSA11]] |-> {{{.*}}[0], %[[SSA38]][0], %[[SSA39]][0], %[[SSA41]][0], %[[arg2]], %[[arg3]], {{.*}}, %[[SSA0]], %[[SSA1]], %[[SSA2]], %[[SSA3]], %[[SSA4]], %[[SSA5]], %[[SSA6]], %[[SSA7]], %[[SSA8]], %[[SSA9]]}
// CHECK: impliesReady end

// Spot check runsLater
// CHECK: runsLater begin
// CHECK:   %[[SSA13]] |-> {%[[SSA17]], %[[SSA18]], %[[SSA19]], %[[SSA23]], %[[SSA24]]}
// CHECK:   %[[SSA14]] |-> {%[[SSA17]], %[[SSA18]], %[[SSA19]], %[[SSA23]], %[[SSA24]]}
// CHECK:   %[[SSA15]] |-> {%[[SSA17]], %[[SSA18]], %[[SSA19]], %[[SSA23]], %[[SSA24]]}
// CHECK: runsLater end

// CHECK: exclusive begin
// CHECK:   %[[SSA3]] |-> {%[[SSA38]][0], %[[SSA18]], %[[SSA19]], %[[SSA23]], %[[SSA24]]}
// CHECK: exclusive end

mgp.model @straightline : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, !mgp.tensor<8xf32>) {
  %0 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32> : !M.array<32xui8>
  %1 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %2 = mgp.tensor.create %ch0, %ctx[1],  %0, %1  : !mgp.tensor<8xf32>
  %3 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32> : !M.array<32xui8>
  %4 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %5 = mgp.tensor.create %ch0, %ctx[1],  %3, %4  : !mgp.tensor<8xf32>
  %bufferResult = mgp.tensor.extract.buffer %2 : <8xf32>
  %bufferResult_0 = mgp.tensor.extract.buffer %5 : <8xf32>
  %size = index.constant 32
  %6 = mgp.buffer.alloc %ch0, %ctx[1], %size, %cpu : !mgp.buffer<32>, !mgp.opaque<"cpu0_context">
  %7 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %8 = "mul"(%ch0, %bufferResult, %bufferResult_0, %6) { tempInOutSig=#M.inout_sig<"iiio"> } : (!mef.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mef.chain
  %9 = mgp.tensor.create %8, %ctx[1],  %6, %7  : !mgp.tensor<8xf32>
  %bufferResult_2 = mgp.tensor.extract.buffer %9 : <8xf32>
  %bufferResult_4 = mgp.tensor.extract.buffer %5 : <8xf32>
  %10 = mgp.buffer.alloc %ch0, %ctx[1], %size, %cpu : !mgp.buffer<32>, !mgp.opaque<"cpu0_context">
  %11 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %12 = "sub"(%ch0, %bufferResult_2, %bufferResult_4, %10) { tempInOutSig=#M.inout_sig<"iiio"> } : (!mef.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mef.chain
  %13 = mgp.tensor.create %12, %ctx[1],  %10, %11  : !mgp.tensor<8xf32>
  %bufferResult_6 = mgp.tensor.extract.buffer %13 : <8xf32>
  %bufferResult_8 = mgp.tensor.extract.buffer %9 : <8xf32>
  %14 = mgp.buffer.alloc %ch0, %ctx[1], %size, %cpu : !mgp.buffer<32>, !mgp.opaque<"cpu0_context">
  %15 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %16 = "add"(%ch0, %bufferResult_6, %bufferResult_8, %14) { tempInOutSig=#M.inout_sig<"iiio"> } : (!mef.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mef.chain
  %17 = mgp.tensor.create %16, %ctx[1],  %14, %15  : !mgp.tensor<8xf32>
  mef.output %ch0, %17 : !mef.chain, !mgp.tensor<8xf32>
}
