// RUN: mo-opt %s --mlir-disable-threading --allow-unregistered-dialect --async-dataflow-dependencies-test | FileCheck %s

//
// Simple grt.if
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK-LABEL: "mgp.model"() ({
// ### Block 0 ###
// CHECK:   %[[SSA76:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA77:.*]] = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %[[SSA78:.*]] = "mgp.check_runtime"(%[[SSA76]], %[[SSA77]]) {aSlot = 0 : ui64, bDevices = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "unknown", arch = "">>]>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   %[[SSA79:.*]] = "mgp.runtime.create"(%[[SSA78]], %[[SSA77]])
// CHECK:   "mef.output"(%[[SSA79]], %[[SSA77]]) : (!mef.chain, !mgp.context) -> ()
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%arg26: !mef.chain, %arg27: !mgp.context):
// CHECK:   %[[SSA68:.*]] = "some.op"(%arg26) : (!mef.chain) -> i1
// CHECK:   %[[SSA69:.*]] = "some.op"(%arg26) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA70:.*]] = "some.op"(%arg26) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA71:.*]]:3 = "grt.if"(%[[SSA68]], %arg26, %[[SSA69]], %[[SSA70]], %[[SSA70]]) ({
// ### Block 2 ###
// CHECK:   ^bb0(%arg32: !mef.chain, %arg33: !mgp.tensor<f32>, %arg34: !mgp.tensor<f32>, %arg35: !mgp.tensor<f32>):
// CHECK:     %[[SSA74:.*]] = "some.op"(%arg32) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:     %[[SSA75:.*]] = "some.op"(%arg34) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:     "mef.output"(%[[SSA74]], %[[SSA75]], %arg35) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%arg28: !mef.chain, %arg29: !mgp.tensor<f32>, %arg30: !mgp.tensor<f32>, %arg31: !mgp.tensor<f32>):
// CHECK:     %[[SSA73:.*]] = "some.op"(%arg30) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:     "mef.output"(%arg29, %[[SSA73]], %arg31) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> ()
// CHECK:   }) : (i1, !mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>)
// CHECK:   %[[SSA72:.*]] = "some.op"(%[[SSA71]]#0, %[[SSA71]]#1, %[[SSA71]]#2) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%arg26, %[[SSA72]]) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: }) {frameworkName = "mof", inputOutputTypes = () -> !mgp.tensor<f32>, name = "simple"} : () -> ()

// Notice if results now capture both arms
// CHECK: maybeSameAs begin
// CHECK:   %arg26 |-> {%[[SSA79]][0]}
// CHECK:   %arg27 |-> {%[[SSA77]][0]}
// CHECK:   %arg32[2] |-> {%[[SSA79]][0], %arg26}
// CHECK:   %arg33[2] |-> {%[[SSA69]]}
// CHECK:   %arg34[2] |-> {%[[SSA70]]}
// CHECK:   %arg35[2] |-> {%[[SSA70]]}
// CHECK:   %arg28[3] |-> {%[[SSA79]][0], %arg26}
// CHECK:   %arg29[3] |-> {%[[SSA69]]}
// CHECK:   %arg30[3] |-> {%[[SSA70]]}
// CHECK:   %arg31[3] |-> {%[[SSA70]]}
// CHECK:   %[[SSA71]]#0 |-> {%[[SSA69]], %[[SSA74]][2], %arg29[3]}
// CHECK:   %[[SSA71]]#1 |-> {%[[SSA75]][2], %[[SSA73]][3]}
// CHECK:   %[[SSA71]]#2 |-> {%[[SSA70]], %arg35[2], %arg31[3]}
// CHECK: maybeSameAs end

// CHECK: isAlwaysReady({%[[SSA76]][0]})

// impliesReady is actually 'implies ready or never bound', so
// %[[SSA71]]#1 captures that either one or other of the if branches will
// be executed.
// We know the if condition is ready if any of its results are ready.
// CHECK: impliesReady begin
// CHECK:   %[[SSA71]]#0 |-> {%[[SSA76]][0], %[[SSA77]][0], %[[SSA78]][0], %[[SSA79]][0], %arg26, %[[SSA68]], %[[SSA74]][2]}
// CHECK:   %[[SSA71]]#1 |-> {%[[SSA76]][0], %[[SSA77]][0], %[[SSA78]][0], %[[SSA79]][0], %arg26, %[[SSA68]], %[[SSA70]], %[[SSA75]][2], %[[SSA73]][3]}
// CHECK:   %[[SSA71]]#2 |-> {%[[SSA76]][0], %[[SSA77]][0], %[[SSA78]][0], %[[SSA79]][0], %arg26, %[[SSA68]], %[[SSA70]]}
// CHECK: impliesReady end

// Ops in the if branches are structurally disjoint
// CHECK: structurallyAtMostOne begin
// CHECK:   %[[SSA74]][2] |-> {%[[SSA73]][3]}
// CHECK:   %[[SSA75]][2] |-> {%[[SSA73]][3]}
// CHECK:   %[[SSA73]][3] |-> {%[[SSA74]][2], %[[SSA75]][2]}
// CHECK: structurallyAtMostOne end

// CHECK: structurallyCannotOverlapRelation begin
// CHECK:   %[[SSA74]][2] |-> {%[[SSA73]][3]}
// CHECK:   %[[SSA75]][2] |-> {%[[SSA73]][3]}
// CHECK:   %[[SSA73]][3] |-> {%[[SSA74]][2], %[[SSA75]][2]}
// CHECK: structurallyCannotOverlapRelation end

// runsLater is actually 'runs later or never runs'.
// CHECK: runsLater begin
// CHECK:   %[[SSA76]][0] |-> {%[[SSA78]][0], %[[SSA79]][0], %[[SSA68]], %[[SSA69]], %[[SSA70]], %[[SSA71]]#0, %[[SSA74]][2], %[[SSA75]][2], %[[SSA73]][3], %[[SSA72]]}
// CHECK:   %[[SSA77]][0] |-> {%[[SSA78]][0], %[[SSA79]][0], %[[SSA68]], %[[SSA69]], %[[SSA70]], %[[SSA71]]#0, %[[SSA74]][2], %[[SSA75]][2], %[[SSA73]][3], %[[SSA72]]}
// CHECK:   %[[SSA78]][0] |-> {%[[SSA79]][0], %[[SSA68]], %[[SSA69]], %[[SSA70]], %[[SSA71]]#0, %[[SSA74]][2], %[[SSA75]][2], %[[SSA73]][3], %[[SSA72]]}
// CHECK:   %[[SSA79]][0] |-> {%[[SSA68]], %[[SSA69]], %[[SSA70]], %[[SSA71]]#0, %[[SSA74]][2], %[[SSA75]][2], %[[SSA73]][3], %[[SSA72]]}
// CHECK:   %[[SSA68]] |-> {%[[SSA71]]#0, %[[SSA72]]}
// CHECK:   %[[SSA70]] |-> {%[[SSA75]][2], %[[SSA73]][3], %[[SSA72]]}
// CHECK:   %[[SSA71]]#0 |-> {%[[SSA72]]}
// CHECK:   %[[SSA74]][2] |-> {%[[SSA72]]}
// CHECK:   %[[SSA75]][2] |-> {%[[SSA72]]}
// CHECK:   %[[SSA73]][3] |-> {%[[SSA72]]}
// CHECK: runsLater end

// exclusive now accounts for nesting and the disjointness of the if branches
// CHECK: exclusive begin
// CHECK:   %[[SSA76]][0] |-> {%[[SSA79]][0], %arg26, %[[SSA68]], %[[SSA69]], %[[SSA70]], %arg32[2], %arg33[2], %arg34[2], %arg35[2], %[[SSA74]][2], %[[SSA75]][2], %arg28[3], %arg29[3], %arg30[3], %arg31[3], %[[SSA73]][3], %[[SSA71]]#0, %[[SSA71]]#1, %[[SSA71]]#2, %[[SSA72]]}
// CHECK:   %[[SSA78]][0] |-> {%[[SSA68]], %[[SSA69]], %[[SSA70]], %arg33[2], %arg34[2], %arg35[2], %[[SSA74]][2], %[[SSA75]][2], %arg29[3], %arg30[3], %arg31[3], %[[SSA73]][3], %[[SSA71]]#0, %[[SSA71]]#1, %[[SSA71]]#2, %[[SSA72]]}
// CHECK:   %[[SSA79]][0] |-> {%[[SSA76]][0]}
// CHECK:   %arg26 |-> {%[[SSA76]][0]}
// CHECK:   %[[SSA68]] |-> {%[[SSA76]][0], %[[SSA78]][0], %[[SSA72]]}
// CHECK:   %[[SSA69]] |-> {%[[SSA76]][0], %[[SSA78]][0]}
// CHECK:   %[[SSA70]] |-> {%[[SSA76]][0], %[[SSA78]][0]}
// CHECK:   %arg32[2] |-> {%[[SSA76]][0]}
// CHECK:   %arg33[2] |-> {%[[SSA76]][0], %[[SSA78]][0]}
// CHECK:   %arg34[2] |-> {%[[SSA76]][0], %[[SSA78]][0]}
// CHECK:   %arg35[2] |-> {%[[SSA76]][0], %[[SSA78]][0]}
// CHECK:   %[[SSA74]][2] |-> {%[[SSA76]][0], %[[SSA78]][0], %[[SSA73]][3]}
// CHECK:   %[[SSA75]][2] |-> {%[[SSA76]][0], %[[SSA78]][0], %[[SSA73]][3]}
// CHECK:   %arg28[3] |-> {%[[SSA76]][0]}
// CHECK:   %arg29[3] |-> {%[[SSA76]][0], %[[SSA78]][0]}
// CHECK:   %arg30[3] |-> {%[[SSA76]][0], %[[SSA78]][0]}
// CHECK:   %arg31[3] |-> {%[[SSA76]][0], %[[SSA78]][0]}
// CHECK:   %[[SSA73]][3] |-> {%[[SSA76]][0], %[[SSA78]][0], %[[SSA74]][2], %[[SSA75]][2]}
// CHECK:   %[[SSA71]]#0 |-> {%[[SSA76]][0], %[[SSA78]][0]}
// CHECK:   %[[SSA71]]#1 |-> {%[[SSA76]][0], %[[SSA78]][0]}
// CHECK:   %[[SSA71]]#2 |-> {%[[SSA76]][0], %[[SSA78]][0]}
// CHECK:   %[[SSA72]] |-> {%[[SSA76]][0], %[[SSA78]][0], %[[SSA68]]}
// CHECK: exclusive end

mgp.model @simple : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %cxt: !mgp.context) -> (!mef.chain, !mgp.tensor<f32>) {
  %cond = "some.op"(%ch0) : (!mef.chain) -> i1
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %y = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %i, %j, %k = grt.if %cond (%ch0: !mef.chain, %x: !mgp.tensor<f32>, %y: !mgp.tensor<f32>, %y as %z: !mgp.tensor<f32>)
    -> (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) {
    %c = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
    %d = "some.op"(%y) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
    mef.output %c, %d, %z : !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>
  } else {
    %e = "some.op"(%y) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
    mef.output %x, %e, %z : !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>
  }
  %l = "some.op"(%i, %j, %k) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
  mef.output %ch0, %l : !mef.chain, !mgp.tensor<f32>
}

//
// Lots of redundant grt.chain.create operands.
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// ### Block 0 ###
// CHECK-LABEL: "mgp.model"() ({
// CHECK:   %[[SSA64:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA65:.*]] = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %[[SSA66:.*]] = "mgp.check_runtime"(%[[SSA64]], %[[SSA65]]) {aSlot = 0 : ui64, bDevices = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "unknown", arch = "">>]>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   %[[SSA67:.*]] = "mgp.runtime.create"(%[[SSA66]], %[[SSA65]])
// CHECK:   "mef.output"(%[[SSA67]], %[[SSA65]]) : (!mef.chain, !mgp.context) -> ()
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%arg22: !mef.chain, %arg23: !mgp.context):
// CHECK:   %[[SSA49:.*]] = "grt.chain.create"(%arg22) : (!mef.chain) -> !mef.chain
// CHECK:   %[[SSA50:.*]] = "grt.chain.create"(%[[SSA49]], %[[SSA49]]) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %[[SSA51:.*]] = "some.op"(%[[SSA50]]) : (!mef.chain) -> !mef.chain
// CHECK:   %[[SSA52:.*]] = "grt.chain.create"(%arg22, %[[SSA51]]) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %[[SSA53:.*]] = "some.op"(%[[SSA52]]) : (!mef.chain) -> !mef.chain
// CHECK:   %[[SSA54:.*]] = "some.op"(%[[SSA52]]) : (!mef.chain) -> !mef.chain
// CHECK:   %[[SSA55:.*]] = "some.op"(%[[SSA52]]) : (!mef.chain) -> !mef.chain
// CHECK:   %[[SSA56:.*]] = "grt.chain.create"(%[[SSA53]], %[[SSA54]]) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %[[SSA57:.*]] = "grt.chain.create"(%[[SSA52]], %[[SSA56]], %[[SSA55]]) : (!mef.chain, !mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %[[SSA58:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA59:.*]] = "grt.chain.create"(%[[SSA58]], %[[SSA57]]) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %[[SSA60:.*]] = "some.op"() : () -> i1
// CHECK:   %[[SSA61:.*]] = "grt.if"(%[[SSA60]], %[[SSA58]]) ({
// ### Block 2 ###
// CHECK:   ^bb0(%arg25: !mef.chain):
// CHECK:     "mef.output"(%arg25) : (!mef.chain) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%arg24: !mef.chain):
// CHECK:     %[[SSA63:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:     "mef.output"(%[[SSA63]]) : (!mef.chain) -> ()
// CHECK:   }) : (i1, !mef.chain) -> !mef.chain
// CHECK:   %[[SSA62:.*]] = "grt.chain.create"(%[[SSA58]], %[[SSA59]], %[[SSA61]]) : (!mef.chain, !mef.chain, !mef.chain) -> !mef.chain
// CHECK:   "mef.output"(%[[SSA62]]) : (!mef.chain) -> ()
// CHECK: }) {frameworkName = "mof", inputOutputTypes = () -> (), name = "redundant"} : () -> ()

// CHECK: sameAs begin
// CHECK:   %arg25[2] |-> {%[[SSA58]]}
// CHECK:   %arg24[3] |-> {%[[SSA58]]}
// CHECK: sameAs end

// CHECK: maybeSameAs begin
// CHECK:   %[[SSA61]] |-> {%[[SSA58]], %arg25[2], %[[SSA63]][3]}
// CHECK: maybeSameAs end

// CHECK: isAlwaysReady({%[[SSA64]][0], %[[SSA58]], %arg25[2], %arg24[3], %[[SSA63]][3], %[[SSA61]]})

// CHECK: impliesReady begin
// CHECK:   %[[SSA51]] |-> {%[[SSA64]][0], %[[SSA65]][0], %[[SSA66]][0], %[[SSA67]][0], %arg22, %[[SSA49]], %[[SSA50]]}
// CHECK: impliesReady end

// CHECK: runsLater begin
// CHECK:   %[[SSA56]] |-> {%[[SSA57]], %[[SSA59]], %[[SSA62]]}
// CHECK: runsLater end

// CHECK: exclusive begin
// CHECK:   %[[SSA56]] |-> {%[[SSA64]][0], %[[SSA66]][0], %[[SSA49]], %[[SSA50]], %[[SSA51]], %[[SSA59]], %[[SSA62]]}
// CHECK: exclusive end

mgp.model @redundant : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> !mef.chain {
  %ch1 = grt.chain.create (%ch0)
  %ch2 = grt.chain.create (%ch1, %ch1)
  %ch3 = "some.op"(%ch2) : (!mef.chain) -> !mef.chain
  %ch4 = grt.chain.create (%ch0, %ch3)
  %ch5 = "some.op"(%ch4) : (!mef.chain) -> !mef.chain
  %ch6 = "some.op"(%ch4) : (!mef.chain) -> !mef.chain
  %ch7 = "some.op"(%ch4) : (!mef.chain) -> !mef.chain
  %ch8 = grt.chain.create (%ch5, %ch6)
  %ch9 = grt.chain.create (%ch4, %ch8, %ch7)
  %ch10 = grt.chain.create ()
  %ch11 = grt.chain.create (%ch10, %ch9)
  %c = "some.op"() : () -> i1
  %ch12 = grt.if %c (%ch10 : !mef.chain) -> !mef.chain {
    mef.output %ch10 : !mef.chain
  } else {
    %ch13 = grt.chain.create ()
    mef.output %ch13 : !mef.chain
  }
  %ch14 = grt.chain.create (%ch10, %ch11, %ch12)
  mef.output %ch14 : !mef.chain
}

//
// Realistic grt.if with tensors in and out
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK-LABEL: "mgp.model"() ({
// ### Block 0 ###
// CHECK:   %[[SSA45:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA46:.*]] = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %[[SSA47:.*]] = "mgp.check_runtime"(%[[SSA45]], %[[SSA46]]) {aSlot = 0 : ui64, bDevices = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "unknown", arch = "">>]>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   %[[SSA48:.*]] = "mgp.runtime.create"(%[[SSA47]], %[[SSA46]])
// CHECK:   "mef.output"(%[[SSA48]], %[[SSA46]]) : (!mef.chain, !mgp.context) -> ()
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%arg12: !mef.chain, %arg13: !mgp.context):
// CHECK:   %[[SSA21:.*]] = "mgp.chain.host_to_device"(%arg12, %arg13) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mgp.chain
// CHECK:   %[[SSA22:.*]] = "mgp.buffer.constant"(%arg12, %arg13) {aRuntimeSlot = 1 : ui64, bValue = #M.aligned_bytes<"0x01", align 8> : !M.array<1xui8>, cRawAlign = 8 : ui64, dDevice = #M.device_ref<"cpu", 0>, eIsTrivial = true} : (!mef.chain, !mgp.context) -> !mgp.buffer<1>
// CHECK:   %[[SSA23:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<> : !M.array<0xui64>, bRawDType = 1 : ui8} : () -> !mgp.tensor_spec<bool>
// CHECK:   %[[SSA24:.*]] = "mgp.tensor.create"(%[[SSA21]], %arg13, %[[SSA22]], %[[SSA23]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<1>, !mgp.tensor_spec<bool>) -> !mgp.tensor<bool>
// CHECK:   %[[SSA25:.*]] = "mgp.buffer.alloc.static"(%[[SSA21]], %arg13) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %[[SSA26:.*]] = "initialize"(%[[SSA25]]) {tempInOutSig = #M.inout_sig<"o">} : (!mgp.buffer<32>) -> !mgp.chain
// CHECK:   %[[SSA27:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<32> : !M.array<1xui64>, bRawDType = 134 : ui8} : () -> !mgp.tensor_spec<32xui8>
// CHECK:   %[[SSA28:.*]] = "mgp.tensor.create"(%[[SSA26]], %arg13, %[[SSA25]], %[[SSA27]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mgp.tensor<32xui8>
// CHECK:   %[[SSA29:.*]] = "mgp.buffer.alloc.static"(%[[SSA21]], %arg13) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %[[SSA30:.*]] = "initialize"(%[[SSA29]]) {tempInOutSig = #M.inout_sig<"o">} : (!mgp.buffer<32>) -> !mgp.chain
// CHECK:   %[[SSA31:.*]] = "mgp.tensor.create"(%[[SSA30]], %arg13, %[[SSA29]], %[[SSA27]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mgp.tensor<32xui8>
// CHECK:   %[[SSA32:.*]] = "mgp.tensor.extract.buffer"(%arg13, %[[SSA24]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<bool>) -> !mgp.buffer<1>
// CHECK:   %[[SSA33:.*]] = "mgp.buffer.to_bool"(%[[SSA21]], %arg13, %[[SSA32]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<1>) -> i1
// CHECK:   %[[SSA34:.*]]:2 = "grt.if"(%[[SSA33]], %[[SSA21]], %[[SSA28]], %[[SSA31]], %arg13) ({
// ### Block 2 ###
// CHECK:   ^bb0(%arg18: !mgp.chain, %arg19: !mgp.tensor<32xui8>, %arg20: !mgp.tensor<32xui8>, %arg21: !mgp.context):
// CHECK:     %[[SSA42:.*]] = "mgp.tensor.extract.buffer"(%arg21, %arg20) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<32xui8>) -> !mgp.buffer<32>
// CHECK:     %[[SSA43:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<32> : !M.array<1xui64>, bRawDType = 134 : ui8} : () -> !mgp.tensor_spec<32xui8>
// CHECK:     %[[SSA44:.*]] = "mgp.tensor.create"(%arg18, %arg21, %[[SSA42]], %[[SSA43]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mgp.tensor<32xui8>
// CHECK:     "mef.output"(%arg18, %[[SSA44]]) : (!mgp.chain, !mgp.tensor<32xui8>) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%arg14: !mgp.chain, %arg15: !mgp.tensor<32xui8>, %arg16: !mgp.tensor<32xui8>, %arg17: !mgp.context):
// CHECK:     %[[SSA39:.*]] = "mgp.tensor.extract.buffer"(%arg17, %arg15) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<32xui8>) -> !mgp.buffer<32>
// CHECK:     %[[SSA40:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<32> : !M.array<1xui64>, bRawDType = 134 : ui8} : () -> !mgp.tensor_spec<32xui8>
// CHECK:     %[[SSA41:.*]] = "mgp.tensor.create"(%arg14, %arg17, %[[SSA39]], %[[SSA40]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mgp.tensor<32xui8>
// CHECK:     "mef.output"(%arg14, %[[SSA41]]) : (!mgp.chain, !mgp.tensor<32xui8>) -> ()
// CHECK:   }) : (i1, !mgp.chain, !mgp.tensor<32xui8>, !mgp.tensor<32xui8>, !mgp.context) -> (!mgp.chain, !mgp.tensor<32xui8>)
// CHECK:   %[[SSA35:.*]] = "mgp.tensor.extract.buffer"(%arg13, %[[SSA34]]#1) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<32xui8>) -> !mgp.buffer<32>
// CHECK:   %[[SSA36:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<32> : !M.array<1xui64>, bRawDType = 134 : ui8} : () -> !mgp.tensor_spec<32xui8>
// CHECK:   %[[SSA37:.*]] = "mgp.tensor.create"(%[[SSA34]]#0, %arg13, %[[SSA35]], %[[SSA36]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mgp.tensor<32xui8>
// CHECK:   %[[SSA38:.*]] = "mgp.chain.device_to_host"(%[[SSA34]]#0, %arg13) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%[[SSA38]], %[[SSA37]]) : (!mef.chain, !mgp.tensor<32xui8>) -> ()
// CHECK: }) {frameworkName = "mof", inputOutputTypes = () -> !mgp.tensor<32xui8>, name = "if"} : () -> ()


// CHECK: sameAs begin
// CHECK:   %arg19[2] |-> {%[[SSA28]]}
// CHECK:   %arg20[2] |-> {%[[SSA31]]}
// CHECK: sameAs end

// CHECK: maybeSameAs begin
// CHECK:   %[[SSA34]]#0 |-> {%[[SSA21]], %arg18[2], %arg14[3]}
// The second if result may alias via the tensor.{create,extract}s.
// CHECK:   %[[SSA34]]#1 |-> {%[[SSA25]], %[[SSA28]], %[[SSA29]], %[[SSA31]], %arg20[2], %[[SSA42]][2], %[[SSA44]][2], %arg15[3], %[[SSA39]][3], %[[SSA41]][3]}
// CHECK: maybeSameAs end

// CHECK: impliesReady begin
// CHECK:   %[[SSA44]][2] |-> {%[[SSA45]][0], %[[SSA46]][0], %[[SSA47]][0], %[[SSA48]][0], %arg12, %arg13, %[[SSA21]], %[[SSA27]], %[[SSA29]], %[[SSA30]], %[[SSA31]], %arg18[2], %arg20[2], %arg21[2], %[[SSA42]][2], %[[SSA43]][2]}
// CHECK:   %[[SSA39]][3] |-> {%[[SSA45]][0], %[[SSA46]][0], %[[SSA47]][0], %[[SSA48]][0], %arg12, %arg13, %[[SSA21]], %[[SSA25]], %[[SSA26]], %[[SSA27]], %[[SSA28]], %arg15[3], %arg17[3]}
// CHECK: impliesReady end

// CHECK: runsLater begin
// CHECK:   %[[SSA27]] |-> {%[[SSA28]], %[[SSA31]], %[[SSA42]][2], %[[SSA44]][2], %[[SSA39]][3], %[[SSA41]][3], %[[SSA35]], %[[SSA37]]}
// CHECK: runsLater end

// CHECK: exclusive begin
// The tensor.{create,extract} aliasing excludes many exclusive entries.
// CHECK:   %[[SSA44]][2] |-> {%[[SSA45]][0], %[[SSA47]][0], %[[SSA30]], %[[SSA39]][3], %[[SSA40]][3], %[[SSA41]][3]}
// CHECK:   %[[SSA34]]#1 |-> {%[[SSA45]][0], %[[SSA47]][0], %[[SSA22]], %[[SSA23]], %[[SSA24]], %[[SSA26]], %[[SSA30]], %[[SSA32]]}
// CHECK: exclusive end

mgp.model @if : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain, !mgp.tensor<32xui8>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %0 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x01", align 8> : !M.array<1xui8>
  %1 = mgp.tensor_spec.create : !mgp.tensor_spec<bool>
  %2 = mgp.tensor.create %ch1, %ctx[1],  %0, %1  : !mgp.tensor<bool>
  %3 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  // NOTE: Since %11#1 maybe the same as %6 and %9, and those are considered
  //       to be maybe the same as %3 and %7, the CheckBufferReadWritePass will
  //       complain that %3 and %7 may be concurrently mutated. Fixing that
  //       would require a more discriminating CheckBufferReadWritePass.
  %4 = "initialize"(%3) {tempInOutSig = #M.inout_sig<"o">} : (!mgp.buffer<32>) -> !mgp.chain
  %5 = mgp.tensor_spec.create  : !mgp.tensor_spec<32xui8>
  %6 = mgp.tensor.create %4, %ctx[1],  %3, %5  : !mgp.tensor<32xui8>
  %7 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %8 = "initialize"(%7) {tempInOutSig = #M.inout_sig<"o">} : (!mgp.buffer<32>) -> !mgp.chain
  %9 = mgp.tensor.create %8, %ctx[1],  %7, %5  : !mgp.tensor<32xui8>
  %bufferResult = mgp.tensor.extract.buffer %ctx[1], %2 : !mgp.tensor<bool>
  %10 = mgp.buffer.to_bool %ch1, %ctx[1], %bufferResult : !mgp.buffer<1>
  %11:2 = grt.if %10 (%ch1: !mgp.chain, %6: !mgp.tensor<32xui8>, %9: !mgp.tensor<32xui8>, %ctx: !mgp.context)  -> (!mgp.chain, !mgp.tensor<32xui8>) {
    %bufferResult_2 = mgp.tensor.extract.buffer %ctx[1], %9 : !mgp.tensor<32xui8>
    %tensorSpecResult_3 = mgp.tensor_spec.create : !mgp.tensor_spec<32xui8>
    %13 = mgp.tensor.create %ch1, %ctx[1],  %bufferResult_2, %tensorSpecResult_3 : !mgp.tensor<32xui8>
    mef.output %ch1, %13 : !mgp.chain, !mgp.tensor<32xui8>
  } else {
    %bufferResult_2 = mgp.tensor.extract.buffer %ctx[1], %6 : !mgp.tensor<32xui8>
    %tensorSpecResult_3 = mgp.tensor_spec.create : !mgp.tensor_spec<32xui8>
    %13 = mgp.tensor.create %ch1, %ctx[1],  %bufferResult_2, %tensorSpecResult_3 : !mgp.tensor<32xui8>
    mef.output %ch1, %13 : !mgp.chain, !mgp.tensor<32xui8>
  }
  %bufferResult_0 = mgp.tensor.extract.buffer %ctx[1], %11#1 : !mgp.tensor<32xui8>
  %tensorSpecResult_1 = mgp.tensor_spec.create : !mgp.tensor_spec<32xui8>
  %12 = mgp.tensor.create %11#0, %ctx[1], %bufferResult_0, %tensorSpecResult_1 : !mgp.tensor<32xui8>
  %ch2 = mgp.chain.device_to_host %11#0, %ctx[1]
  mef.output %ch2, %12 : !mef.chain, !mgp.tensor<32xui8>
}

//
// Use of if result
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK-LABEL: "mgp.model"() ({
// ### Block 0 ###
// CHECK:   %[[SSA17:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA18:.*]] = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %[[SSA19:.*]] = "mgp.check_runtime"(%[[SSA17]], %[[SSA18]]) {aSlot = 0 : ui64, bDevices = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "unknown", arch = "">>]>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   %[[SSA20:.*]] = "mgp.runtime.create"(%[[SSA19]], %[[SSA18]])
// CHECK:   "mef.output"(%[[SSA20]], %[[SSA18]]) : (!mef.chain, !mgp.context) -> ()
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK:   %[[SSA0:.*]] = "mgp.chain.host_to_device"(%arg0, %arg1) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mgp.chain
// CHECK:   %[[SSA1:.*]] = "mgp.buffer.constant"(%arg0, %arg1) {aRuntimeSlot = 1 : ui64, bValue = #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32> : !M.array<32xui8>, cRawAlign = 32 : ui64, dDevice = #M.device_ref<"cpu", 0>, eIsTrivial = true} : (!mef.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %[[SSA2:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 66 : ui8} : () -> !mgp.tensor_spec<8xf32>
// CHECK:   %[[SSA3:.*]] = "mgp.buffer.constant"(%arg0, %arg1) {aRuntimeSlot = 1 : ui64, bValue = #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32> : !M.array<32xui8>, cRawAlign = 32 : ui64, dDevice = #M.device_ref<"cpu", 0>, eIsTrivial = true} : (!mef.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %[[SSA4:.*]] = "mgp.buffer.constant"(%arg0, %arg1) {aRuntimeSlot = 1 : ui64, bValue = #M.aligned_bytes<"0x00", align 32> : !M.array<1xui8>, cRawAlign = 32 : ui64, dDevice = #M.device_ref<"cpu", 0>, eIsTrivial = true} : (!mef.chain, !mgp.context) -> !mgp.buffer<1>
// CHECK:   %[[SSA5:.*]] = "mgp.buffer.to_bool"(%[[SSA0]], %arg1, %[[SSA4]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<1>) -> i1
// CHECK:   %[[SSA6:.*]]:2 = "grt.if"(%[[SSA5]], %[[SSA0]], %[[SSA1]], %[[SSA2]], %[[SSA3]], %arg1) ({
// ### Block 2 ###
// CHECK:   ^bb0(%arg7: !mgp.chain, %arg8: !mgp.buffer<32>, %arg9: !mgp.tensor_spec<8xf32>, %arg10: !mgp.buffer<32>, %arg11: !mgp.context):
// CHECK:     %[[SSA15:.*]] = "mgp.buffer.alloc.static"(%arg7, %arg11) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:     %[[SSA16:.*]] = "mul"(%arg7, %arg8, %arg10, %[[SSA15]]) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
// CHECK:     "mef.output"(%[[SSA16]], %[[SSA15]]) : (!mgp.chain, !mgp.buffer<32>) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%arg2: !mgp.chain, %arg3: !mgp.buffer<32>, %arg4: !mgp.tensor_spec<8xf32>, %arg5: !mgp.buffer<32>, %arg6: !mgp.context):
// CHECK:     %[[SSA13:.*]] = "mgp.buffer.alloc.static"(%arg2, %arg6) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:     %[[SSA14:.*]] = "sub"(%arg2, %arg3, %arg5, %[[SSA13]]) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
// CHECK:     "mef.output"(%[[SSA14]], %[[SSA13]]) : (!mgp.chain, !mgp.buffer<32>) -> ()
// CHECK:   }) : (i1, !mgp.chain, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>, !mgp.buffer<32>, !mgp.context) -> (!mgp.chain, !mgp.buffer<32>)
// CHECK:   %[[SSA7:.*]] = "mgp.buffer.alloc.static"(%[[SSA6]]#0, %arg1) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %[[SSA8:.*]] = "sub"(%[[SSA6]]#0, %[[SSA6]]#1, %[[SSA3]], %[[SSA7]]) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
// CHECK:   %[[SSA9:.*]] = "mgp.buffer.alloc.static"(%[[SSA8]], %arg1) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %[[SSA10:.*]] = "add"(%[[SSA8]], %[[SSA7]], %[[SSA3]], %[[SSA9]]) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
// CHECK:   %[[SSA11:.*]] = "mgp.tensor.create"(%[[SSA10]], %arg1, %[[SSA9]], %[[SSA2]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mgp.tensor<8xf32>
// CHECK:   %[[SSA12:.*]] = "mgp.chain.device_to_host"(%[[SSA10]], %arg1) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%[[SSA12]], %[[SSA11]]) : (!mef.chain, !mgp.tensor<8xf32>) -> ()
// CHECK: }) {frameworkName = "mof", inputOutputTypes = () -> !mgp.tensor<8xf32>, name = "conditional_c"} : () -> ()

// CHECK: exclusive begin
// CHECK:   %[[SSA7]] |-> {%[[SSA17]][0], %[[SSA19]][0], %[[SSA0]], %[[SSA1]], %[[SSA4]], %[[SSA5]], %arg7[2], %arg8[2], %arg2[3], %arg3[3], %[[SSA11]], %[[SSA12]]}
// CHECK: exclusive end

mgp.model @conditional_c : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain, !mgp.tensor<8xf32>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %a = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32> : !M.array<32xui8>
  %spec = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %b = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32> : !M.array<32xui8>
  %zero = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x00", align 32> : !M.array<1xui8>
  %cond = mgp.buffer.to_bool %ch1, %ctx[1], %zero : !mgp.buffer<1>
  %ch3, %c = grt.if %cond (%ch1 : !mgp.chain, %a : !mgp.buffer<32>, %spec : !mgp.tensor_spec<8xf32>, %b : !mgp.buffer<32>, %ctx : !mgp.context) -> (!mgp.chain, !mgp.buffer<32>) {
    %c = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
    %ch2 = "mul"(%ch1, %a, %b, %c) {tempInOutSig=#M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
    mef.output %ch2, %c : !mgp.chain, !mgp.buffer<32>
  } else {
    %c = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
    %ch2 = "sub"(%ch1, %a, %b, %c) {tempInOutSig=#M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
    mef.output %ch2, %c : !mgp.chain, !mgp.buffer<32>
  }
  %d = mgp.buffer.alloc.static %ch3, %ctx[1] : !mgp.buffer<32>
  %ch4 = "sub"(%ch3, %c, %b, %d) {tempInOutSig=#M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %e = mgp.buffer.alloc.static %ch4, %ctx[1] : !mgp.buffer<32>
  %ch5 = "add"(%ch4, %d, %b, %e) {tempInOutSig=#M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %out = mgp.tensor.create %ch5, %ctx[1],  %e, %spec : !mgp.tensor<8xf32>
  %ch6 = mgp.chain.device_to_host %ch5, %ctx[1]
  mef.output %ch6, %out : !mef.chain, !mgp.tensor<8xf32>
}
