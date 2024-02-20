// RUN: mgp-opt %s --mlir-disable-threading --allow-unregistered-dialect --async-dataflow-dependencies-test --check-buffer-read-write | FileCheck %s

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() ({
// ### Block 0 ###
// CHECK:   %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %1 = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %2 = "mgp.check_runtime"(%0, %1) {aSlot = 0 : ui64, bDevices = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "unknown", arch = "">>]>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   %3 = "mgp.runtime.create"(%2, %1) {aRuntimeSlot = 1 : ui64, bDeviceSpecMapSlot = 0 : ui64, cDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%3, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK:   %0 = "some.op"(%arg0) : (!mef.chain) -> si32
// CHECK:   %1 = "some.op"(%arg0) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %2 = "grt.repeat.i32"(%0, %1) ({
// ### Block 2 ###
// CHECK:   ^bb0(%arg2: !mgp.tensor<f32>):
// CHECK:     %4 = "some.op"(%arg2) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:     "mef.output"(%4) : (!mgp.tensor<f32>) -> ()
// CHECK:   }) : (si32, !mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   %3 = "some.op"(%2) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%arg0, %3) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: }) {frameworkName = "mof", inputOutputTypes = () -> !mgp.tensor<f32>, sym_name = "repeat1"} : () -> ()


// Note the repeat input, result, sub-block arg and sub-block output are all
// in the same equivalence class.
// CHECK: maybeSameAs begin
// CHECK: %arg2[2] |-> {%1, %4[2]}
// CHECK: %2 |-> {%1, %arg2[2], %4[2]}
// CHECK: maybeSameAs end

mgp.model @repeat1 : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %cxt: !mgp.context) -> (!mef.chain, !mgp.tensor<f32>) {
  %c = "some.op"(%ch0) : (!mef.chain) -> si32
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %z = grt.repeat.i32 %c (%x : !mgp.tensor<f32>) {
    %y = "some.op"(%x) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
    mef.output %y : !mgp.tensor<f32>
  }
  %out = "some.op"(%z) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<f32>
}

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() ({
// ### Block ###
// CHECK:   %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %1 = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %2 = "mgp.check_runtime"(%0, %1) {aSlot = 0 : ui64, bDevices = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "unknown", arch = "">>]>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   %3 = "mgp.runtime.create"(%2, %1) {aRuntimeSlot = 1 : ui64, bDeviceSpecMapSlot = 0 : ui64, cDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%3, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK:   %0 = "some.op"(%arg0) : (!mef.chain) -> si32
// CHECK:   %1 = "some.op"(%arg0) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %2 = "grt.repeat.i32"(%0, %1) ({
// ### Block 2 ###
// CHECK:   ^bb0(%arg2: !mgp.tensor<f32>):
// CHECK:     "mef.output"(%arg2) : (!mgp.tensor<f32>) -> ()
// CHECK:   }) : (si32, !mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   %3 = "some.op"(%2) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%arg0, %3) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: }) {frameworkName = "mof", inputOutputTypes = () -> !mgp.tensor<f32>, sym_name = "repeat2"} : () -> ()

// CHECK: maybeSameAs begin
// CHECK: %arg2[2] |-> {%1, %arg2[2]}
// CHECK: %2 |-> {%1, %arg2[2]}
// CHECK: maybeSameAs end

mgp.model @repeat2 : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %cxt: !mgp.context) -> (!mef.chain, !mgp.tensor<f32>) {
  %c = "some.op"(%ch0) : (!mef.chain) -> si32
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %z = grt.repeat.i32 %c (%x : !mgp.tensor<f32>) {
    mef.output %x : !mgp.tensor<f32>
  }
  %out = "some.op"(%z) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<f32>
}

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() ({
// ### Block 0 ###
// CHECK:   %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %1 = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %2 = "mgp.check_runtime"(%0, %1) {aSlot = 0 : ui64, bDevices = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "unknown", arch = "">>]>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   %3 = "mgp.runtime.create"(%2, %1) {aRuntimeSlot = 1 : ui64, bDeviceSpecMapSlot = 0 : ui64, cDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%3, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK:   %0 = "some.op"(%arg0) : (!mef.chain) -> si32
// CHECK:   %1 = "some.op"(%arg0) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %2:2 = "grt.repeat.i32"(%0, %1, %1) ({
// ### Block 2 ###
// CHECK:   ^bb0(%arg2: !mgp.tensor<f32>, %arg3: !mgp.tensor<f32>):
// CHECK:     %4 = "some.op"(%arg2) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:     "mef.output"(%4, %arg3) : (!mgp.tensor<f32>, !mgp.tensor<f32>) -> ()
// CHECK:   }) : (si32, !mgp.tensor<f32>, !mgp.tensor<f32>) -> (!mgp.tensor<f32>, !mgp.tensor<f32>)
// CHECK:   %3 = "some.op"(%2#0) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%arg0, %3) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: }) {frameworkName = "mof", inputOutputTypes = () -> !mgp.tensor<f32>, sym_name = "repeat3"} : () -> ()

// CHECK: maybeSameAs begin
// CHECK: %arg2[2] |-> {%1, %4[2]}
// CHECK: %arg3[2] |-> {%1, %arg3[2]}
// CHECK: %2#0 |-> {%1, %arg2[2], %4[2]}
// CHECK: %2#1 |-> {%1, %arg3[2]}
// CHECK: maybeSameAs end

mgp.model @repeat3 : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %cxt: !mgp.context) -> (!mef.chain, !mgp.tensor<f32>) {
  %c = "some.op"(%ch0) : (!mef.chain) -> si32
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %z, %dummy = grt.repeat.i32 %c (%x as %x0: !mgp.tensor<f32>, %x as %x1: !mgp.tensor<f32>)  {
    %y = "some.op"(%x0) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
    mef.output %y, %x1 : !mgp.tensor<f32>, !mgp.tensor<f32>
  }
  %out = "some.op"(%z) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<f32>
}

//
// Simple grt.while
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// ### Block 0 ###
// CHECK: "mgp.model"() ({
// CHECK:   %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %1 = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %2 = "mgp.check_runtime"(%0, %1) {aSlot = 0 : ui64, bDevices = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "unknown", arch = "">>]>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   %3 = "mgp.runtime.create"(%2, %1) {aRuntimeSlot = 1 : ui64, bDeviceSpecMapSlot = 0 : ui64, cDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%3, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK:   %0 = "some.op"(%arg0) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %1 = "some.op"(%arg0) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %2:4 = "grt.while"(%arg0, %0, %1, %1) ({
// ### Block 2 ###
// CHECK:   ^bb0(%arg2: !mef.chain, %arg3: !mgp.tensor<f32>, %arg4: !mgp.tensor<f32>, %arg5: !mgp.tensor<f32>):
// CHECK:     %4 = "some.op"(%arg2) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:     %5 = "some.op"(%arg4) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:     %6 = "some.op"(%arg2) : (!mef.chain) -> i1
// CHECK:     "mef.output"(%arg2, %4, %5, %arg5, %6) : (!mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>, i1) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%arg2: !mef.chain, %arg3: !mgp.tensor<f32>, %arg4: !mgp.tensor<f32>, %arg5: !mgp.tensor<f32>):
// CHECK:     %4 = "some.op"(%arg4) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:     "mef.output"(%arg2, %arg3, %4, %arg5) : (!mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> ()
// CHECK:   }) : (!mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> (!mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>)
// CHECK:   %3 = "some.op"(%2#1, %2#2, %2#3) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%2#0, %3) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: }) {frameworkName = "mof", inputOutputTypes = () -> !mgp.tensor<f32>, sym_name = "simple"} : () -> ()

// CHECK: sameAs begin
// CHECK:   %2#0 |-> {%3[0], %arg0, %arg2[2], %arg2[3]}
// CHECK:   %2#1 |-> {%4[2], %arg3[3]}
// CHECK:   %2#2 |-> {%5[2], %arg4[3]}
// CHECK:   %2#3 |-> {%1, %arg5[2], %arg5[3]}
// CHECK: sameAs end

// CHECK: maybeSameAs begin
// CHECK:   %2#0 |-> {%3[0], %arg0, %arg2[2], %arg2[3]}
// CHECK:   %2#1 |-> {%4[2], %arg3[3]}
// CHECK:   %2#2 |-> {%5[2], %arg4[3]}
// CHECK:   %2#3 |-> {%1, %arg5[2], %arg5[3]}
// CHECK: maybeSameAs end

// CHECK: impliesReady begin
// CHECK:   %4[3] |-> {%0[0], %1[0], %2[0], %3[0], %arg0, %1, %arg4[2], %5[2], %arg4[3]}
// CHECK:   %2#0 |-> {%0[0], %1[0], %2[0], %3[0], %arg0, %1, %arg2[2], %arg4[2], %arg5[2], %4[2], %5[2], %arg2[3], %arg3[3], %arg4[3], %arg5[3], %4[3]}
// CHECK: impliesReady end

// No branches, so this relation is empty.
// CHECK: structurallyAtMostOne begin
// CHECK-NEXT: structurallyAtMostOne end

// But the loop cond and body blocks are known not to overlap.
// CHECK: structurallyCannotOverlapRelation begin
// CHECK-NEXT: %4[2] |-> {%4[3]}
// CHECK-NEXT: %5[2] |-> {%4[3]}
// CHECK-NEXT: %6[2] |-> {%4[3]}
// CHECK-NEXT: %4[3] |-> {%4[2], %5[2], %6[2]}
// CHECK-NEXT: structurallyCannotOverlapRelation end

// CHECK: runsLater begin
// CHECK: %4[2] |-> {%3}
// CHECK: runsLater end

mgp.model @simple : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %cxt: !mgp.context) -> (!mef.chain, !mgp.tensor<f32>) {
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %y = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %ch1, %i, %j, %k = grt.while (%ch0: !mef.chain, %x: !mgp.tensor<f32>, %y: !mgp.tensor<f32>, %y as %z: !mgp.tensor<f32>) {
    %c = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
    %d = "some.op"(%y) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
    %cond = "some.op"(%ch0) : (!mef.chain) -> i1
    mef.output %ch0, %c, %d, %z, %cond : !mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>, i1
  } do {
    %e = "some.op"(%y) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
    mef.output %ch0, %x, %e, %z : !mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>
  }
  %l = "some.op"(%i, %j, %k) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
  mef.output %ch1, %l : !mef.chain, !mgp.tensor<f32>
}

//
// Unstable loop
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// ### Block 0 ###
// CHECK: "mgp.model"() ({
// CHECK:   %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %1 = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %2 = "mgp.check_runtime"(%0, %1) {aSlot = 0 : ui64, bDevices = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "unknown", arch = "">>]>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   %3 = "mgp.runtime.create"(%2, %1) {aRuntimeSlot = 1 : ui64, bDeviceSpecMapSlot = 0 : ui64, cDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%3, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK:   %0 = "some.op"(%arg0) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %1 = "some.op"(%arg0) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %2:2 = "grt.while"(%0, %1) ({
// ### Block 2 ###
// CHECK:   ^bb0(%arg2: !mgp.tensor<f32>, %arg3: !mgp.tensor<f32>):
// CHECK:     %4 = "some.op"() : () -> i1
// CHECK:     "mef.output"(%arg2, %arg3, %4) : (!mgp.tensor<f32>, !mgp.tensor<f32>, i1) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%arg2: !mgp.tensor<f32>, %arg3: !mgp.tensor<f32>):
// CHECK:     "mef.output"(%arg3, %arg2) : (!mgp.tensor<f32>, !mgp.tensor<f32>) -> ()
// CHECK:   }) : (!mgp.tensor<f32>, !mgp.tensor<f32>) -> (!mgp.tensor<f32>, !mgp.tensor<f32>)
// CHECK:   %3 = "some.op"(%2#0, %2#1) : (!mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%arg0, %3) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: }) {frameworkName = "mof", inputOutputTypes = () -> !mgp.tensor<f32>, sym_name = "crissCross"} : () -> ()

// CHECK: sameAs begin
// CHECK: %2#0 |-> {%arg2[2], %arg2[3]}
// CHECK: %2#1 |-> {%arg3[2], %arg3[3]}
// CHECK: sameAs end

// CHECK: maybeSameAs begin
// CHECK: %2#0 |-> {%0, %1, %arg2[2], %arg3[2], %arg2[3], %arg3[3]}
// CHECK: %2#1 |-> {%0, %1, %arg2[2], %arg3[2], %arg2[3], %arg3[3]}
// CHECK: maybeSameAs end

mgp.model @crissCross : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %cxt: !mgp.context) -> (!mef.chain, !mgp.tensor<f32>) {
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %y = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %i, %j = grt.while (%x: !mgp.tensor<f32>, %y: !mgp.tensor<f32>) {
    %c = "some.op"() : () -> i1
    mef.output %x, %y, %c : !mgp.tensor<f32>, !mgp.tensor<f32>, i1
  } do {
    mef.output %y, %x: !mgp.tensor<f32>, !mgp.tensor<f32>
  }
  %l = "some.op"(%i, %j) : (!mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
  mef.output %ch0, %l : !mef.chain, !mgp.tensor<f32>
}

//
// Loop requiring more than one iteration to reach fixed point.
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// ### Block 0 ###
// CHECK: "mgp.model"() ({
// CHECK:   %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %1 = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %2 = "mgp.check_runtime"(%0, %1) {aSlot = 0 : ui64, bDevices = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "unknown", arch = "">>]>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   %3 = "mgp.runtime.create"(%2, %1) {aRuntimeSlot = 1 : ui64, bDeviceSpecMapSlot = 0 : ui64, cDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%3, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK:   %0 = "some.op"(%arg0) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %1:3 = "grt.while"(%0, %0, %0) ({
// ### Block 2 ###
// CHECK:   ^bb0(%arg2: !mgp.tensor<f32>, %arg3: !mgp.tensor<f32>, %arg4: !mgp.tensor<f32>):
// CHECK:     %3 = "some.op"() : () -> i1
// CHECK:     "mef.output"(%arg2, %arg3, %arg4, %3) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>, i1) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%arg2: !mgp.tensor<f32>, %arg3: !mgp.tensor<f32>, %arg4: !mgp.tensor<f32>):
// CHECK:     %3 = "some.op"() : () -> !mgp.tensor<f32>
// CHECK:     "mef.output"(%arg3, %arg4, %3) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> ()
// CHECK:   }) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>)
// CHECK:   %2 = "some.op"(%1#0, %1#1, %1#2) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%arg0, %2) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: }) {frameworkName = "mof", inputOutputTypes = () -> !mgp.tensor<f32>, sym_name = "shift"} : () -> ()

// CHECK: sameAs begin
// CHECK: %1#0 |-> {%arg2[2], %arg2[3]}
// CHECK: %1#1 |-> {%arg3[2], %arg3[3]}
// CHECK: %1#2 |-> {%arg4[2], %arg4[3]}
// CHECK: sameAs end

// CHECK: maybeSameAs begin
// CHECK: %1#0 |-> {%0, %arg2[2], %arg3[2], %arg4[2], %arg2[3], %arg3[3], %arg4[3], %3[3]}
// CHECK: %1#1 |-> {%0, %arg3[2], %arg4[2], %arg3[3], %arg4[3], %3[3]}
// CHECK: %1#2 |-> {%0, %arg4[2], %arg4[3], %3[3]}
// CHECK: maybeSameAs end

// CHECK: impliesReady begin
// CHECK: %2 |-> {%arg2[2], %arg3[2], %arg4[2], %arg2[3], %arg3[3], %arg4[3], %3[3], %1#0, %1#1, %1#2}
// CHECK: impliesReady end

mgp.model @shift : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %cxt: !mgp.context) -> (!mef.chain, !mgp.tensor<f32>) {
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %i, %j, %k = grt.while (%x: !mgp.tensor<f32>, %x as %y: !mgp.tensor<f32>, %x as %z: !mgp.tensor<f32>) {
    %c = "some.op"() : () -> i1
    mef.output %x, %y, %z, %c : !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>, i1
  } do {
    %n = "some.op"() : () -> !mgp.tensor<f32>
    mef.output %y, %z, %n: !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>
  }
  %l = "some.op"(%i, %j, %k) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
  mef.output %ch0, %l : !mef.chain, !mgp.tensor<f32>
}

//
// Realistic grt.while with tensors in and out
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// ### Block 0 ###
// CHECK: "mgp.model"() ({
// CHECK:   %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %1 = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %2 = "mgp.check_runtime"(%0, %1) {aSlot = 0 : ui64, bDevices = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "unknown", arch = "">>]>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   %3 = "mgp.runtime.create"(%2, %1) {aRuntimeSlot = 1 : ui64, bDeviceSpecMapSlot = 0 : ui64, cDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%3, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK:   %0 = "mgp.chain.host_to_device"(%arg0, %arg1) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mgp.chain
// CHECK:   %1 = "mgp.buffer.constant"(%arg1) {aRuntimeSlot = 1 : ui64, bValue = #M.aligned_bytes<"0x00000004", align 8> : !M.array<4xui8>, cRawAlign = 8 : ui64, dDevice = #M.device_ref<"cpu", 0>, eIsTrivial = true} : (!mgp.context) -> !mgp.buffer<4>
// CHECK:   %2 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<> : !M.array<0xui64>, bRawDType = 138 : ui8} : () -> !mgp.tensor_spec<ui32>
// CHECK:   %3 = "mgp.tensor.create"(%0, %arg1, %1, %2) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<4>, !mgp.tensor_spec<ui32>) -> !mgp.tensor<ui32>
// CHECK:   %4 = "mgp.buffer.alloc.static"(%0, %arg1) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %5 = "initialize"(%4) {tempInOutSig = #M.inout_sig<"o">} : (!mgp.buffer<32>) -> !mgp.chain
// CHECK:   %6 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<32> : !M.array<1xui64>, bRawDType = 134 : ui8} : () -> !mgp.tensor_spec<32xui8>
// CHECK:   %7 = "mgp.tensor.create"(%5, %arg1, %4, %6) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mgp.tensor<32xui8>
// CHECK:   %8 = "mgp.buffer.alloc.static"(%0, %arg1) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %9 = "initialize"(%8) {tempInOutSig = #M.inout_sig<"o">} : (!mgp.buffer<32>) -> !mgp.chain
// CHECK:   %10 = "mgp.tensor.create"(%9, %arg1, %8, %6) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mgp.tensor<32xui8>
// CHECK:   %11:5 = "grt.while"(%0, %3, %7, %10, %arg1) ({
// ### Block 2 ###
// CHECK:   ^bb0(%arg2: !mgp.chain, %arg3: !mgp.tensor<ui32>, %arg4: !mgp.tensor<32xui8>, %arg5: !mgp.tensor<32xui8>, %arg6: !mgp.context):
// CHECK:     %16 = "mgp.buffer.constant"(%arg6) {aRuntimeSlot = 1 : ui64, bValue = #M.aligned_bytes<"0x00000000", align 8> : !M.array<4xui8>, cRawAlign = 8 : ui64, dDevice = #M.device_ref<"cpu", 0>, eIsTrivial = true} : (!mgp.context) -> !mgp.buffer<4>
// CHECK:     %17 = "mgp.tensor.extract.buffer"(%arg6, %arg3) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<ui32>) -> !mgp.buffer<4>
// CHECK:     %18 = "mgp.buffer.alloc.static"(%arg2, %arg6) {aRuntimeSlot = 1 : ui64, bSize = 1 : ui64, cRawAlign = 32 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<1>
// CHECK:     %19 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<> : !M.array<0xui64>, bRawDType = 1 : ui8} : () -> !mgp.tensor_spec<bool>
// CHECK:     %20 = "greater"(%arg2, %17, %16, %18) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<4>, !mgp.buffer<4>, !mgp.buffer<1>) -> !mgp.chain
// CHECK:     %21 = "mgp.tensor.create"(%20, %arg6, %18, %19) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<1>, !mgp.tensor_spec<bool>) -> !mgp.tensor<bool>
// CHECK:     %22 = "mgp.tensor.extract.buffer"(%arg6, %21) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<bool>) -> !mgp.buffer<1>
// CHECK:     %23 = "mgp.buffer.to_bool"(%arg2, %arg6, %22) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<1>) -> i1
// CHECK:     "mef.output"(%arg2, %arg3, %arg4, %arg5, %arg6, %23) : (!mgp.chain, !mgp.tensor<ui32>, !mgp.tensor<32xui8>, !mgp.tensor<32xui8>, !mgp.context, i1) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%arg2: !mgp.chain, %arg3: !mgp.tensor<ui32>, %arg4: !mgp.tensor<32xui8>, %arg5: !mgp.tensor<32xui8>, %arg6: !mgp.context):
// CHECK:     %16 = "mgp.tensor.extract.buffer"(%arg6, %arg4) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<32xui8>) -> !mgp.buffer<32>
// CHECK:     %17 = "mgp.tensor.extract.buffer"(%arg6, %arg5) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<32xui8>) -> !mgp.buffer<32>
// CHECK:     %18 = "mgp.buffer.alloc.static"(%arg2, %arg6) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:     %19 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<32> : !M.array<1xui64>, bRawDType = 134 : ui8} : () -> !mgp.tensor_spec<32xui8>
// CHECK:     %20 = "add"(%arg2, %16, %17, %18) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
// CHECK:     %21 = "mgp.tensor.create"(%20, %arg6, %18, %19) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mgp.tensor<32xui8>
// CHECK:     %22 = "mgp.buffer.constant"(%arg6) {aRuntimeSlot = 1 : ui64, bValue = #M.aligned_bytes<"0x00000001", align 8> : !M.array<4xui8>, cRawAlign = 8 : ui64, dDevice = #M.device_ref<"cpu", 0>, eIsTrivial = true} : (!mgp.context) -> !mgp.buffer<4>
// CHECK:     %23 = "mgp.tensor.extract.buffer"(%arg6, %arg3) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<ui32>) -> !mgp.buffer<4>
// CHECK:     %24 = "mgp.buffer.alloc.static"(%arg2, %arg6) {aRuntimeSlot = 1 : ui64, bSize = 4 : ui64, cRawAlign = 32 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<4>
// CHECK:     %25 = "sub"(%arg2, %23, %22, %24) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<4>, !mgp.buffer<4>, !mgp.buffer<4>) -> !mgp.chain
// CHECK:     %26 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<> : !M.array<0xui64>, bRawDType = 138 : ui8} : () -> !mgp.tensor_spec<ui32>
// CHECK:     %27 = "mgp.tensor.create"(%25, %arg6, %24, %26) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<4>, !mgp.tensor_spec<ui32>) -> !mgp.tensor<ui32>
// CHECK:     "mef.output"(%arg2, %27, %21, %arg5, %arg6) : (!mgp.chain, !mgp.tensor<ui32>, !mgp.tensor<32xui8>, !mgp.tensor<32xui8>, !mgp.context) -> ()
// CHECK:   }) : (!mgp.chain, !mgp.tensor<ui32>, !mgp.tensor<32xui8>, !mgp.tensor<32xui8>, !mgp.context) -> (!mgp.chain, !mgp.tensor<ui32>, !mgp.tensor<32xui8>, !mgp.tensor<32xui8>, !mgp.context)
// CHECK:   %12 = "mgp.tensor.extract.buffer"(%arg1, %11#2) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<32xui8>) -> !mgp.buffer<32>
// CHECK:   %13 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<32> : !M.array<1xui64>, bRawDType = 134 : ui8} : () -> !mgp.tensor_spec<32xui8>
// CHECK:   %14 = "mgp.tensor.create"(%0, %arg1, %12, %13) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mgp.tensor<32xui8>
// CHECK:   %15 = "mgp.chain.device_to_host"(%11#0, %arg1) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%15, %14) : (!mef.chain, !mgp.tensor<32xui8>) -> ()
// CHECK: }) {frameworkName = "mof", inputOutputTypes = () -> !mgp.tensor<32xui8>, sym_name = "while"} : () -> ()

// CHECK: sameAs begin
// CHECK:   %11#0 |-> {%0, %arg2[2], %arg2[3]}
// CHECK:   %11#1 |-> {%arg3[2], %arg3[3]}
// CHECK:   %11#2 |-> {%arg4[2], %arg4[3]}
// CHECK:   %11#3 |-> {%10, %arg5[2], %arg5[3]}
// CHECK:   %11#4 |-> {%1[0], %arg1, %arg6[2], %arg6[3]}
// CHECK: sameAs end

// CHECK: impliesReady begin
// CHECK:   %10 |-> {%0[0], %1[0], %2[0], %3[0], %arg0, %arg1, %0, %6, %8, %9}
// CHECK:   %arg3[2] |-> {%0[0], %1[0], %2[0], %3[0], %arg0, %arg1, %0, %1, %2, %3}
// CHECK:   %arg4[2] |-> {%0[0], %1[0], %2[0], %3[0], %arg0, %arg1, %0, %4, %5, %6, %7}
// CHECK:   %11#0 |-> {%0[0], %1[0], %2[0], %3[0], %arg0, %arg1, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %arg2[2], %arg3[2], %arg4[2], %arg5[2], %arg6[2], %arg2[3], %arg3[3], %arg4[3], %arg5[3], %arg6[3], %16[3], %17[3], %18[3], %19[3], %20[3], %21[3], %22[3], %23[3], %24[3], %25[3], %26[3], %27[3]}
// CHECK:   %14 |-> {%0[0], %1[0], %2[0], %3[0], %arg0, %arg1, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %arg2[2], %arg3[2], %arg4[2], %arg5[2], %arg6[2], %arg2[3], %arg3[3], %arg4[3], %arg5[3], %arg6[3], %16[3], %17[3], %18[3], %19[3], %20[3], %21[3], %22[3], %23[3], %24[3], %25[3], %26[3], %27[3], %11#2, %12, %13}
// CHECK: impliesReady end

// The tensor.{create,extract} aliasing excludes entries.
// CHECK: exclusive begin
// CHECK: %21[3] |-> {%0[0], %2[0], %5, %6, %9}
// CHECK: exclusive end

mgp.model @while : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain, !mgp.tensor<32xui8>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %0 = mgp.buffer.constant %ctx[1], #M.aligned_bytes<"0x00000004", align 8> : !M.array<4xui8>
  %1 = mgp.cpu.tensor_spec.create  : !mgp.tensor_spec<ui32>
  %2 = mgp.tensor.create %ch1, %ctx[1], %0, %1  : !mgp.tensor<ui32>
  %3 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %4 = "initialize"(%3) {tempInOutSig=#M.inout_sig<"o">} : (!mgp.buffer<32>) -> !mgp.chain
  %5 = mgp.cpu.tensor_spec.create  : !mgp.tensor_spec<32xui8>
  %6 = mgp.tensor.create %4, %ctx[1], %3, %5  : !mgp.tensor<32xui8>
  %7 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %8 = "initialize"(%7) {tempInOutSig=#M.inout_sig<"o">} : (!mgp.buffer<32>) -> !mgp.chain
  %9 = mgp.tensor.create %8, %ctx[1],  %7, %5  : !mgp.tensor<32xui8>
  %10:5 = grt.while (%ch1: !mgp.chain, %2: !mgp.tensor<ui32>, %6: !mgp.tensor<32xui8>, %9: !mgp.tensor<32xui8>, %ctx: !mgp.context) {
    %12 = mgp.buffer.constant %ctx[1], #M.aligned_bytes<"0x00000000", align 8> : !M.array<4xui8>
    %bufferResult_0 = mgp.tensor.extract.buffer %ctx[1], %2 : !mgp.tensor<ui32>
    %13 = mgp.buffer.alloc.static %ch1, %ctx[1], align 32 : !mgp.buffer<1>
    %14 = mgp.cpu.tensor_spec.create  : !mgp.tensor_spec<bool>
    %15 = "greater"(%ch1, %bufferResult_0, %12, %13) {tempInOutSig=#M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<4>, !mgp.buffer<4>, !mgp.buffer<1>) -> !mgp.chain
    %16 = mgp.tensor.create %15, %ctx[1],  %13, %14  : !mgp.tensor<bool>
    %bufferResult_2 = mgp.tensor.extract.buffer %ctx[1], %16 : !mgp.tensor<bool>
    %17 = mgp.buffer.to_bool %ch1, %ctx[1], %bufferResult_2 : !mgp.buffer<1>
    mef.output %ch1, %2, %6, %9, %ctx, %17 : !mgp.chain, !mgp.tensor<ui32>, !mgp.tensor<32xui8>, !mgp.tensor<32xui8>, !mgp.context, i1
  } do {
    %bufferResult_0 = mgp.tensor.extract.buffer %ctx[1], %6 : !mgp.tensor<32xui8>
    %bufferResult_2 = mgp.tensor.extract.buffer %ctx[1], %9 : !mgp.tensor<32xui8>
    %12 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
    %13 = mgp.cpu.tensor_spec.create  : !mgp.tensor_spec<32xui8>
    %14 = "add"(%ch1, %bufferResult_0, %bufferResult_2, %12) {tempInOutSig=#M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
    %15 = mgp.tensor.create %14, %ctx[1],  %12, %13  : !mgp.tensor<32xui8>
    %16 = mgp.buffer.constant %ctx[1], #M.aligned_bytes<"0x00000001", align 8> : !M.array<4xui8>
    %bufferResult_4 = mgp.tensor.extract.buffer %ctx[1], %2 : !mgp.tensor<ui32>
    %17 = mgp.buffer.alloc.static %ch1, %ctx[1], align 32 : !mgp.buffer<4>
    %18 = "sub"(%ch1, %bufferResult_4, %16, %17) {tempInOutSig=#M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<4>, !mgp.buffer<4>, !mgp.buffer<4>) -> !mgp.chain
    %tensorSpecResult_5 = mgp.cpu.tensor_spec.create : !mgp.tensor_spec<ui32>
    %19 = mgp.tensor.create %18, %ctx[1],  %17, %tensorSpecResult_5  : !mgp.tensor<ui32>
    mef.output %ch1, %19, %15, %9, %ctx : !mgp.chain, !mgp.tensor<ui32>, !mgp.tensor<32xui8>, !mgp.tensor<32xui8>, !mgp.context
  }
  %bufferResult = mgp.tensor.extract.buffer %ctx[1], %10#2 : !mgp.tensor<32xui8>
  %tensorSpecResult = mgp.cpu.tensor_spec.create : !mgp.tensor_spec<32xui8>
  %11 = mgp.tensor.create %ch1, %ctx[1],  %bufferResult, %tensorSpecResult : !mgp.tensor<32xui8>
  %ch2 = mgp.chain.device_to_host %10#0, %ctx[1]
  mef.output %ch2, %11 : !mef.chain, !mgp.tensor<32xui8>
}
