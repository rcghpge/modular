// RUN: mgp-opt %s --mlir-disable-threading --allow-unregistered-dialect --async-dataflow-dependencies-test --check-buffer-read-write | FileCheck %s

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() ({
//              ### Block 0 ###
// CHECK-NEXT:  %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK-NEXT:  %1 = "mgp.context.create"() {numSlots = 0 : ui64} : () -> !mgp.context
// CHECK-NEXT:  "mef.output"(%0, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK-NEXT:}, {
//            ### Block 1 ###
// CHECK-NEXT:^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK-NEXT:  %0 = "some.op"(%arg0) : (!mef.chain) -> si32
// CHECK-NEXT:  %1 = "some.op"(%arg0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  %2 = "grt.repeat.i32"(%0, %1) ({
//              ### Block 2 ###
// CHECK-NEXT:  ^bb0(%arg2: !mgp.cpu.tensor<f32>):
// CHECK-NEXT:    %4 = "some.op"(%arg2) : (!mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:    "mef.output"(%4) : (!mgp.cpu.tensor<f32>) -> ()
// CHECK-NEXT:  }) : (si32, !mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  %3 = "some.op"(%2) : (!mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  "mef.output"(%arg0, %3) : (!mef.chain, !mgp.cpu.tensor<f32>) -> ()
// CHECK-NEXT:}) {frameworkName = "tf", inputOutputTypes = () -> !mgp.cpu.tensor<f32>, sym_name = "repeat1"} : () -> ()

// CHECK: maybeSameAs begin
// CHECK: %arg2[2] |-> {%1, %4[2]}
// CHECK: %2 |-> {%1, %arg2[2], %4[2]}
// CHECK: maybeSameAs end

mgp.model @repeat1 : tf init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %cxt: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<f32>) {
  %c = "some.op"(%ch0) : (!mef.chain) -> si32
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
  %z = grt.repeat.i32 %c (%x : !mgp.cpu.tensor<f32>) {
    %y = "some.op"(%x) : (!mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
    mef.output %y : !mgp.cpu.tensor<f32>
  }
  %out = "some.op"(%z) : (!mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
  mef.output %ch0, %out : !mef.chain, !mgp.cpu.tensor<f32>
}

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() ({
//              ### Block 0 ###
// CHECK-NEXT:  %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK-NEXT:  %1 = "mgp.context.create"() {numSlots = 0 : ui64} : () -> !mgp.context
// CHECK-NEXT:  "mef.output"(%0, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK-NEXT:}, {
//            ### Block 1 ###
// CHECK-NEXT:^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK-NEXT:  %0 = "some.op"(%arg0) : (!mef.chain) -> si32
// CHECK-NEXT:  %1 = "some.op"(%arg0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  %2 = "grt.repeat.i32"(%0, %1) ({
//              ### Block 2 ###
// CHECK-NEXT:  ^bb0(%arg2: !mgp.cpu.tensor<f32>):
// CHECK-NEXT:    "mef.output"(%arg2) : (!mgp.cpu.tensor<f32>) -> ()
// CHECK-NEXT:  }) : (si32, !mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  %3 = "some.op"(%2) : (!mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  "mef.output"(%arg0, %3) : (!mef.chain, !mgp.cpu.tensor<f32>) -> ()
// CHECK-NEXT:}) {frameworkName = "tf", inputOutputTypes = () -> !mgp.cpu.tensor<f32>, sym_name = "repeat2"} : () -> ()

// CHECK: maybeSameAs begin
// CHECK: %arg2[2] |-> {%1, %arg2[2]}
// CHECK: %2 |-> {%1, %arg2[2]}
// CHECK: maybeSameAs end

mgp.model @repeat2 : tf init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %cxt: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<f32>) {
  %c = "some.op"(%ch0) : (!mef.chain) -> si32
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
  %z = grt.repeat.i32 %c (%x : !mgp.cpu.tensor<f32>) {
    mef.output %x : !mgp.cpu.tensor<f32>
  }
  %out = "some.op"(%z) : (!mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
  mef.output %ch0, %out : !mef.chain, !mgp.cpu.tensor<f32>
}

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() ({
//              ### Block 0 ###
// CHECK-NEXT:  %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK-NEXT:  %1 = "mgp.context.create"() {numSlots = 0 : ui64} : () -> !mgp.context
// CHECK-NEXT:  "mef.output"(%0, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK-NEXT:}, {
//            ### Block 1 ###
// CHECK-NEXT:^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK-NEXT:  %0 = "some.op"(%arg0) : (!mef.chain) -> si32
// CHECK-NEXT:  %1 = "some.op"(%arg0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  %2:2 = "grt.repeat.i32"(%0, %1, %1) ({
//              ### Block 2 ###
// CHECK-NEXT:  ^bb0(%arg2: !mgp.cpu.tensor<f32>, %arg3: !mgp.cpu.tensor<f32>):
// CHECK-NEXT:    %4 = "some.op"(%arg2) : (!mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:    "mef.output"(%4, %arg3) : (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> ()
// CHECK-NEXT:  }) : (si32, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>)
// CHECK-NEXT:  %3 = "some.op"(%2#0) : (!mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  "mef.output"(%arg0, %3) : (!mef.chain, !mgp.cpu.tensor<f32>) -> ()
// CHECK-NEXT:}) {frameworkName = "tf", inputOutputTypes = () -> !mgp.cpu.tensor<f32>, sym_name = "repeat3"} : () -> ()

// CHECK: maybeSameAs begin
// CHECK: %arg2[2] |-> {%1, %4[2]}
// CHECK: %arg3[2] |-> {%1, %arg3[2]}
// CHECK: %2#0 |-> {%1, %arg2[2], %4[2]}
// CHECK: %2#1 |-> {%1, %arg3[2]}
// CHECK: maybeSameAs end

mgp.model @repeat3 : tf init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %cxt: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<f32>) {
  %c = "some.op"(%ch0) : (!mef.chain) -> si32
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
  %z, %dummy = grt.repeat.i32 %c (%x as %x0: !mgp.cpu.tensor<f32>, %x as %x1: !mgp.cpu.tensor<f32>)  {
    %y = "some.op"(%x0) : (!mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
    mef.output %y, %x1 : !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>
  }
  %out = "some.op"(%z) : (!mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
  mef.output %ch0, %out : !mef.chain, !mgp.cpu.tensor<f32>
}


//
// Simple grt.while
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() ({
//             ### Block 0 ###
// CHECK-NEXT:  %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK-NEXT:  %1 = "mgp.context.create"() {numSlots = 0 : ui64} : () -> !mgp.context
// CHECK-NEXT:  "mef.output"(%0, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK-NEXT:}, {
//            ### Block 1 ###
// CHECK-NEXT:^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK-NEXT:  %0 = "some.op"(%arg0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  %1 = "some.op"(%arg0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  %2:4 = "grt.while"(%arg0, %0, %1, %1) ({
//              ### Block 2 ###
// CHECK-NEXT:  ^bb0(%arg2: !mef.chain, %arg3: !mgp.cpu.tensor<f32>, %arg4: !mgp.cpu.tensor<f32>, %arg5: !mgp.cpu.tensor<f32>):
// CHECK-NEXT:    %4 = "some.op"(%arg2) : (!mef.chain) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:    %5 = "some.op"(%arg4) : (!mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:    %6 = "some.op"(%arg2) : (!mef.chain) -> i1
// CHECK-NEXT:    "mef.output"(%arg2, %4, %5, %arg5, %6) : (!mef.chain, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, i1) -> ()
// CHECK-NEXT:  }, {
//              ### Block 3 ###
// CHECK-NEXT:  ^bb0(%arg2: !mef.chain, %arg3: !mgp.cpu.tensor<f32>, %arg4: !mgp.cpu.tensor<f32>, %arg5: !mgp.cpu.tensor<f32>):
// CHECK-NEXT:    %4 = "some.op"(%arg4) : (!mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:    "mef.output"(%arg2, %arg3, %4, %arg5) : (!mef.chain, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> ()
// CHECK-NEXT:  }) : (!mef.chain, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> (!mef.chain, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>)
// CHECK-NEXT:  %3 = "some.op"(%2#1, %2#2, %2#3) : (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  "mef.output"(%2#0, %3) : (!mef.chain, !mgp.cpu.tensor<f32>) -> ()
// CHECK-NEXT:}) {frameworkName = "tf", inputOutputTypes = () -> !mgp.cpu.tensor<f32>, sym_name = "simple"} : () -> ()

// CHECK: sameAs begin
// CHECK: %2#0 |-> {%0[0], %arg0, %arg2[2], %arg2[3]}
// CHECK: %2#1 |-> {%4[2], %arg3[3]}
// CHECK: %2#2 |-> {%5[2], %arg4[3]}
// CHECK: %2#3 |-> {%1, %arg5[2], %arg5[3]}
// CHECK: sameAs end

// CHECK: maybeSameAs begin
// CHECK: %2#0 |-> {%0[0], %arg0, %arg2[2], %arg2[3]}
// CHECK: %2#1 |-> {%4[2], %arg3[3]}
// CHECK: %2#2 |-> {%5[2], %arg4[3]}
// CHECK: %2#3 |-> {%1, %arg5[2], %arg5[3]}
// CHECK: maybeSameAs end

// CHECK: impliesReady begin
// CHECK: %4[3] |-> {%0[0], %arg0, %1, %arg4[2], %5[2], %arg4[3]}
// CHECK: %2#0 |-> {%0[0], %arg0, %1, %arg2[2], %arg4[2], %arg5[2], %4[2], %5[2], %arg2[3], %arg3[3], %arg4[3], %arg5[3], %4[3]}
// CHECK: impliesReady end

// CHECK: runsLater begin
// CHECK: %4[2] |-> {%3}
// CHECK: runsLater end

// Nothing is exclusive.
// CHECK: exclusive begin
// CHECK-NEXT: exclusive end

mgp.model @simple : tf init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %cxt: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<f32>) {
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
  %y = "some.op"(%ch0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
  %ch1, %i, %j, %k = grt.while (%ch0: !mef.chain, %x: !mgp.cpu.tensor<f32>, %y: !mgp.cpu.tensor<f32>, %y as %z: !mgp.cpu.tensor<f32>) {
    %c = "some.op"(%ch0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
    %d = "some.op"(%y) : (!mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
    %cond = "some.op"(%ch0) : (!mef.chain) -> i1
    mef.output %ch0, %c, %d, %z, %cond : !mef.chain, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, i1
  } do {
    %e = "some.op"(%y) : (!mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
    mef.output %ch0, %x, %e, %z : !mef.chain, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>
  }
  %l = "some.op"(%i, %j, %k) : (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
  mef.output %ch1, %l : !mef.chain, !mgp.cpu.tensor<f32>
}

//
// Unstable loop
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() ({
//              ### Block 0 ###
// CHECK-NEXT:  %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK-NEXT:  %1 = "mgp.context.create"() {numSlots = 0 : ui64} : () -> !mgp.context
// CHECK-NEXT:  "mef.output"(%0, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK-NEXT:}, {
//            ### Block 1 ###
// CHECK-NEXT:^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK-NEXT:  %0 = "some.op"(%arg0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  %1 = "some.op"(%arg0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  %2:2 = "grt.while"(%0, %1) ({
//              ### Block 2 ###
// CHECK-NEXT:  ^bb0(%arg2: !mgp.cpu.tensor<f32>, %arg3: !mgp.cpu.tensor<f32>):
// CHECK-NEXT:    %4 = "some.op"() : () -> i1
// CHECK-NEXT:    "mef.output"(%arg2, %arg3, %4) : (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, i1) -> ()
// CHECK-NEXT:  }, {
//              ### Block 3 ###
// CHECK-NEXT:  ^bb0(%arg2: !mgp.cpu.tensor<f32>, %arg3: !mgp.cpu.tensor<f32>):
// CHECK-NEXT:    "mef.output"(%arg3, %arg2) : (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> ()
// CHECK-NEXT:  }) : (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>)
// CHECK-NEXT:  %3 = "some.op"(%2#0, %2#1) : (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  "mef.output"(%arg0, %3) : (!mef.chain, !mgp.cpu.tensor<f32>) -> ()
// CHECK-NEXT:}) {frameworkName = "tf", inputOutputTypes = () -> !mgp.cpu.tensor<f32>, sym_name = "crissCross"} : () -> ()

// CHECK: sameAs begin
// CHECK: %2#0 |-> {%arg2[2], %arg2[3]}
// CHECK: %2#1 |-> {%arg3[2], %arg3[3]}
// CHECK: sameAs end

// CHECK: maybeSameAs begin
// CHECK: %2#0 |-> {%0, %1, %arg2[2], %arg3[2], %arg2[3], %arg3[3]}
// CHECK: %2#1 |-> {%0, %1, %arg2[2], %arg3[2], %arg2[3], %arg3[3]}
// CHECK: maybeSameAs end

mgp.model @crissCross : tf init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %cxt: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<f32>) {
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
  %y = "some.op"(%ch0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
  %i, %j = grt.while (%x: !mgp.cpu.tensor<f32>, %y: !mgp.cpu.tensor<f32>) {
    %c = "some.op"() : () -> i1
    mef.output %x, %y, %c : !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, i1
  } do {
    mef.output %y, %x: !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>
  }
  %l = "some.op"(%i, %j) : (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
  mef.output %ch0, %l : !mef.chain, !mgp.cpu.tensor<f32>
}

//
// Loop requiring more than one iteration to reach fixed point.
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() ({
//              ### Block 0 ###
// CHECK-NEXT:  %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK-NEXT:  %1 = "mgp.context.create"() {numSlots = 0 : ui64} : () -> !mgp.context
// CHECK-NEXT:  "mef.output"(%0, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK-NEXT:}, {
//            ### Block 1 ###
// CHECK-NEXT:^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK-NEXT:  %0 = "some.op"(%arg0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  %1:3 = "grt.while"(%0, %0, %0) ({
//              ### Block 2 ###
// CHECK-NEXT:  ^bb0(%arg2: !mgp.cpu.tensor<f32>, %arg3: !mgp.cpu.tensor<f32>, %arg4: !mgp.cpu.tensor<f32>):
// CHECK-NEXT:    %3 = "some.op"() : () -> i1
// CHECK-NEXT:    "mef.output"(%arg2, %arg3, %arg4, %3) : (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, i1) -> ()
// CHECK-NEXT:  }, {
//              ### Block 3 ###
// CHECK-NEXT:  ^bb0(%arg2: !mgp.cpu.tensor<f32>, %arg3: !mgp.cpu.tensor<f32>, %arg4: !mgp.cpu.tensor<f32>):
// CHECK-NEXT:    %3 = "some.op"() : () -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:    "mef.output"(%arg3, %arg4, %3) : (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> ()
// CHECK-NEXT:  }) : (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>)
// CHECK-NEXT:  %2 = "some.op"(%1#0, %1#1, %1#2) : (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  "mef.output"(%arg0, %2) : (!mef.chain, !mgp.cpu.tensor<f32>) -> ()
// CHECK-NEXT:}) {frameworkName = "tf", inputOutputTypes = () -> !mgp.cpu.tensor<f32>, sym_name = "shift"} : () -> ()

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

mgp.model @shift : tf init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %cxt: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<f32>) {
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
  %i, %j, %k = grt.while (%x: !mgp.cpu.tensor<f32>, %x as %y: !mgp.cpu.tensor<f32>, %x as %z: !mgp.cpu.tensor<f32>) {
    %c = "some.op"() : () -> i1
    mef.output %x, %y, %z, %c : !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, i1
  } do {
    %n = "some.op"() : () -> !mgp.cpu.tensor<f32>
    mef.output %y, %z, %n: !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>
  }
  %l = "some.op"(%i, %j, %k) : (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
  mef.output %ch0, %l : !mef.chain, !mgp.cpu.tensor<f32>
}

//
// Realistic grt.while with tensors in and out
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() ({
//              ### Block 0 ###
// CHECK-NEXT:  %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK-NEXT:  %1 = "mgp.context.create"() {numSlots = 0 : ui64} : () -> !mgp.context
// CHECK-NEXT:  "mef.output"(%0, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK-NEXT:}, {
//            ### Block 1 ###
// CHECK-NEXT:^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK-NEXT:  %0 = "mgp.cpu.buffer.constant"() {aValue = #M.aligned_bytes<"0x00000004", align 8> : !M.array<4xui8>, bRawAlign = 8 : ui64} : () -> !mgp.cpu.buffer<4>
// CHECK-NEXT:  %1 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<> : !M.array<0xui64>, bRawDType = 138 : ui8} : () -> !mgp.cpu.tensor_spec<ui32>
// CHECK-NEXT:  %2 = "mgp.cpu.tensor.create"(%arg0, %0, %1) : (!mef.chain, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>) -> !mgp.cpu.tensor<ui32>
// CHECK-NEXT:  %3 = "mgp.cpu.buffer.alloc.static"(%arg0) {aSize = 32 : ui64, bRawAlign = 18446744073709551615 : ui64} : (!mef.chain) -> !mgp.cpu.buffer<32>
// CHECK-NEXT:  %4 = "mgp.cpu.buffer.set"(%3) {value = 0 : ui8} : (!mgp.cpu.buffer<32>) -> !mef.chain
// CHECK-NEXT:  %5 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<32> : !M.array<1xui64>, bRawDType = 134 : ui8} : () -> !mgp.cpu.tensor_spec<32xui8>
// CHECK-NEXT:  %6 = "mgp.cpu.tensor.create"(%4, %3, %5) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>) -> !mgp.cpu.tensor<32xui8>
// CHECK-NEXT:  %7 = "mgp.cpu.buffer.alloc.static"(%arg0) {aSize = 32 : ui64, bRawAlign = 18446744073709551615 : ui64} : (!mef.chain) -> !mgp.cpu.buffer<32>
// CHECK-NEXT:  %8 = "mgp.cpu.buffer.set"(%7) {value = 1 : ui8} : (!mgp.cpu.buffer<32>) -> !mef.chain
// CHECK-NEXT:  %9 = "mgp.cpu.tensor.create"(%8, %7, %5) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>) -> !mgp.cpu.tensor<32xui8>
// CHECK-NEXT:  %10:4 = "grt.while"(%arg0, %2, %6, %9) ({
//              ### Block 2 ###
// CHECK-NEXT:  ^bb0(%arg2: !mef.chain, %arg3: !mgp.cpu.tensor<ui32>, %arg4: !mgp.cpu.tensor<32xui8>, %arg5: !mgp.cpu.tensor<32xui8>):
// CHECK-NEXT:    %13 = "mgp.cpu.buffer.constant"() {aValue = #M.aligned_bytes<"0x00000000", align 8> : !M.array<4xui8>, bRawAlign = 8 : ui64} : () -> !mgp.cpu.buffer<4>
// CHECK-NEXT:    %14:2 = "mgp.cpu.tensor.extract"(%arg3) : (!mgp.cpu.tensor<ui32>) -> (!mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>)
// CHECK-NEXT:    %15 = "mgp.cpu.buffer.alloc.static"(%arg2) {aSize = 1 : ui64, bRawAlign = 32 : ui64} : (!mef.chain) -> !mgp.cpu.buffer<1>
// CHECK-NEXT:    %16 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<> : !M.array<0xui64>, bRawDType = 1 : ui8} : () -> !mgp.cpu.tensor_spec<bool>
// CHECK-NEXT:    %17 = "mop.kernels.greater"(%arg2, %14#0, %14#1, %13, %14#1, %15, %16) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>, !mgp.cpu.buffer<1>, !mgp.cpu.tensor_spec<bool>) -> !mef.chain
// CHECK-NEXT:    %18 = "mgp.cpu.tensor.create"(%17, %15, %16) : (!mef.chain, !mgp.cpu.buffer<1>, !mgp.cpu.tensor_spec<bool>) -> !mgp.cpu.tensor<bool>
// CHECK-NEXT:    %19:2 = "mgp.cpu.tensor.extract"(%18) : (!mgp.cpu.tensor<bool>) -> (!mgp.cpu.buffer<1>, !mgp.cpu.tensor_spec<bool>)
// CHECK-NEXT:    %20 = "mgp.to_grt_bool"(%arg2, %19#0) : (!mef.chain, !mgp.cpu.buffer<1>) -> i1
// CHECK-NEXT:    "mef.output"(%arg2, %arg3, %arg4, %arg5, %20) : (!mef.chain, !mgp.cpu.tensor<ui32>, !mgp.cpu.tensor<32xui8>, !mgp.cpu.tensor<32xui8>, i1) -> ()
// CHECK-NEXT:  }, {
//              ### Block 3 ###
// CHECK-NEXT:  ^bb0(%arg2: !mef.chain, %arg3: !mgp.cpu.tensor<ui32>, %arg4: !mgp.cpu.tensor<32xui8>, %arg5: !mgp.cpu.tensor<32xui8>):
// CHECK-NEXT:    %13:2 = "mgp.cpu.tensor.extract"(%arg4) : (!mgp.cpu.tensor<32xui8>) -> (!mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>)
// CHECK-NEXT:    %14:2 = "mgp.cpu.tensor.extract"(%arg5) : (!mgp.cpu.tensor<32xui8>) -> (!mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>)
// CHECK-NEXT:    %15 = "mgp.cpu.buffer.alloc.static"(%arg2) {aSize = 32 : ui64, bRawAlign = 18446744073709551615 : ui64} : (!mef.chain) -> !mgp.cpu.buffer<32>
// CHECK-NEXT:    %16 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<32> : !M.array<1xui64>, bRawDType = 134 : ui8} : () -> !mgp.cpu.tensor_spec<32xui8>
// CHECK-NEXT:    %17 = "mop.kernels.add"(%arg2, %13#0, %13#1, %14#0, %14#1, %15, %16) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>) -> !mef.chain
// CHECK-NEXT:    %18 = "mgp.cpu.tensor.create"(%17, %15, %16) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>) -> !mgp.cpu.tensor<32xui8>
// CHECK-NEXT:    %19 = "mgp.cpu.buffer.constant"() {aValue = #M.aligned_bytes<"0x00000001", align 8> : !M.array<4xui8>, bRawAlign = 8 : ui64} : () -> !mgp.cpu.buffer<4>
// CHECK-NEXT:    %20:2 = "mgp.cpu.tensor.extract"(%arg3) : (!mgp.cpu.tensor<ui32>) -> (!mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>)
// CHECK-NEXT:    %21 = "mgp.cpu.buffer.alloc.static"(%arg2) {aSize = 4 : ui64, bRawAlign = 32 : ui64} : (!mef.chain) -> !mgp.cpu.buffer<4>
// CHECK-NEXT:    %22 = "mop.kernels.sub"(%arg2, %20#0, %20#1, %19, %20#1, %21, %20#1) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>) -> !mef.chain
// CHECK-NEXT:    %23 = "mgp.cpu.tensor.create"(%22, %21, %20#1) : (!mef.chain, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>) -> !mgp.cpu.tensor<ui32>
// CHECK-NEXT:    "mef.output"(%arg2, %23, %18, %arg5) : (!mef.chain, !mgp.cpu.tensor<ui32>, !mgp.cpu.tensor<32xui8>, !mgp.cpu.tensor<32xui8>) -> ()
// CHECK-NEXT:  }) : (!mef.chain, !mgp.cpu.tensor<ui32>, !mgp.cpu.tensor<32xui8>, !mgp.cpu.tensor<32xui8>) -> (!mef.chain, !mgp.cpu.tensor<ui32>, !mgp.cpu.tensor<32xui8>, !mgp.cpu.tensor<32xui8>)
// CHECK-NEXT:  %11:2 = "mgp.cpu.tensor.extract"(%10#2) : (!mgp.cpu.tensor<32xui8>) -> (!mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>)
// CHECK-NEXT:  %12 = "mgp.cpu.tensor.create"(%arg0, %11#0, %11#1) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>) -> !mgp.cpu.tensor<32xui8>
// CHECK-NEXT:  "mef.output"(%10#0, %12) : (!mef.chain, !mgp.cpu.tensor<32xui8>) -> ()
// CHECK-NEXT:}) {frameworkName = "tf", inputOutputTypes = () -> !mgp.cpu.tensor<32xui8>, sym_name = "while"} : () -> ()

// CHECK: sameAs begin
// CHECK: %10#0 |-> {%0[0], %arg0, %arg2[2], %arg2[3]}
// CHECK: %10#1 |-> {%arg3[2], %arg3[3]}
// CHECK: %10#2 |-> {%arg4[2], %arg4[3]}
// CHECK: %10#3 |-> {%9, %arg5[2], %arg5[3]}
// CHECK: sameAs end

// CHECK: impliesReady begin
// CHECK: %9 |-> {%0[0], %arg0, %5, %7, %8}
// CHECK: %arg3[2] |-> {%0[0], %arg0, %0, %1, %2}
// CHECK: %arg4[2] |-> {%0[0], %arg0, %3, %4, %5, %6}
// CHECK: %10#0 |-> {%0[0], %arg0, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %arg2[2], %arg3[2], %arg4[2], %arg5[2], %arg2[3], %arg3[3], %arg4[3], %arg5[3], %13#0[3], %13#1[3], %14#0[3], %14#1[3], %15[3], %16[3], %17[3], %18[3], %19[3], %20#0[3], %20#1[3], %21[3], %22[3], %23[3]}
// CHECK: %12 |-> {%0[0], %arg0, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %arg2[2], %arg3[2], %arg4[2], %arg5[2], %arg2[3], %arg3[3], %arg4[3], %arg5[3], %13#0[3], %13#1[3], %14#0[3], %14#1[3], %15[3], %16[3], %17[3], %18[3], %19[3], %20#0[3], %20#1[3], %21[3], %22[3], %23[3], %10#2, %11#0, %11#1}
// CHECK: impliesReady end

// CHECK: exclusive begin
// CHECK: %18[3] |-> {%3, %4, %5, %7, %8, %13#0[3], %13#1[3], %14#0[3], %14#1[3]}
// CHECK: exclusive end

mgp.model @while : tf init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<32xui8>) {
  %0 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x00000004", align 8> : !M.array<4xui8>
  %1 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<ui32>
  %2 = mgp.cpu.tensor.create %arg0, %0, %1  : !mgp.cpu.tensor<ui32>
  %3 = mgp.cpu.buffer.alloc.static %arg0, 32
  %4 = mgp.cpu.buffer.set %3 : !mgp.cpu.buffer<32>
  %5 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<32xui8>
  %6 = mgp.cpu.tensor.create %4, %3, %5  : !mgp.cpu.tensor<32xui8>
  %7 = mgp.cpu.buffer.alloc.static %arg0, 32
  %8 = mgp.cpu.buffer.set %7, 1 : !mgp.cpu.buffer<32>
  %9 = mgp.cpu.tensor.create %8, %7, %5  : !mgp.cpu.tensor<32xui8>
  %10:4 = grt.while (%arg0: !mef.chain, %2: !mgp.cpu.tensor<ui32>, %6: !mgp.cpu.tensor<32xui8>, %9: !mgp.cpu.tensor<32xui8>) {
    %12 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x00000000", align 8> : !M.array<4xui8>
    %bufferResult_0, %tensorSpecResult_1 = mgp.cpu.tensor.extract %2 : !mgp.cpu.tensor<ui32>
    %13 = mgp.cpu.buffer.alloc.static %arg0, 1, align 32
    %14 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<bool>
    %15 = "mop.kernels.greater"(%arg0, %bufferResult_0, %tensorSpecResult_1, %12, %tensorSpecResult_1, %13, %14) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>, !mgp.cpu.buffer<1>, !mgp.cpu.tensor_spec<bool>) -> !mef.chain
    %16 = mgp.cpu.tensor.create %15, %13, %14  : !mgp.cpu.tensor<bool>
    %bufferResult_2, %tensorSpecResult_3 = mgp.cpu.tensor.extract %16 : !mgp.cpu.tensor<bool>
    %17 = mgp.to_grt_bool(%arg0, %bufferResult_2) : (!mef.chain, !mgp.cpu.buffer<1>) -> i1
    mef.output %arg0, %2, %6, %9, %17 : !mef.chain, !mgp.cpu.tensor<ui32>, !mgp.cpu.tensor<32xui8>, !mgp.cpu.tensor<32xui8>, i1
  } do {
    %bufferResult_0, %tensorSpecResult_1 = mgp.cpu.tensor.extract %6 : !mgp.cpu.tensor<32xui8>
    %bufferResult_2, %tensorSpecResult_3 = mgp.cpu.tensor.extract %9 : !mgp.cpu.tensor<32xui8>
    %12 = mgp.cpu.buffer.alloc.static %arg0, 32
    %13 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<32xui8>
    %14 = "mop.kernels.add"(%arg0, %bufferResult_0, %tensorSpecResult_1, %bufferResult_2, %tensorSpecResult_3, %12, %13) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>) -> !mef.chain
    %15 = mgp.cpu.tensor.create %14, %12, %13  : !mgp.cpu.tensor<32xui8>
    %16 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x00000001", align 8> : !M.array<4xui8>
    %bufferResult_4, %tensorSpecResult_5 = mgp.cpu.tensor.extract %2 : !mgp.cpu.tensor<ui32>
    %17 = mgp.cpu.buffer.alloc.static %arg0, 4, align 32
    %18 = "mop.kernels.sub"(%arg0, %bufferResult_4, %tensorSpecResult_5, %16, %tensorSpecResult_5, %17, %tensorSpecResult_5) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>) -> !mef.chain
    %19 = mgp.cpu.tensor.create %18, %17, %tensorSpecResult_5  : !mgp.cpu.tensor<ui32>
    mef.output %arg0, %19, %15, %9 : !mef.chain, !mgp.cpu.tensor<ui32>, !mgp.cpu.tensor<32xui8>, !mgp.cpu.tensor<32xui8>
  }
  %bufferResult, %tensorSpecResult = mgp.cpu.tensor.extract %10#2 : !mgp.cpu.tensor<32xui8>
  %11 = mgp.cpu.tensor.create %arg0, %bufferResult, %tensorSpecResult  : !mgp.cpu.tensor<32xui8>
  mef.output %10#0, %11 : !mef.chain, !mgp.cpu.tensor<32xui8>
}
