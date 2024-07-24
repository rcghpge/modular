// RUN: mo-opt %s --mlir-disable-threading --allow-unregistered-dialect --async-dataflow-dependencies-test --check-buffer-read-write | FileCheck %s

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() ({
// ### Block 0 ###
// CHECK:   %[[SSA94:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA95:.*]] = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %[[SSA97:.*]] = "mgp.runtime.create"(%[[SSA94]], %[[SSA95]])
// CHECK:   "mef.output"(%[[SSA97]], %[[SSA95]]
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%[[arg43:[a-z0-9]*]]: !mef.chain, %[[arg44:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA89:.*]] = "some.op"(%[[arg43]]) : (!mef.chain) -> si32
// CHECK:   %[[SSA90:.*]] = "some.op"(%[[arg43]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA91:.*]] = "grt.repeat.i32"(%[[SSA89]], %[[SSA90]]) ({
// ### Block 2 ###
// CHECK:   ^bb0(%[[arg45:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     %[[SSA93:.*]] = "some.op"(%[[arg45]]) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:     "mef.output"(%[[SSA93]]) : (!mgp.tensor<f32>) -> ()
// CHECK:   }) : (si32, !mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   %[[SSA92:.*]] = "some.op"(%[[SSA91]]) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%[[arg43]], %[[SSA92]]) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: })


// Note the repeat input, result, sub-block arg and sub-block output are all
// in the same equivalence class.
// CHECK: maybeSameAs begin
// CHECK:   %[[arg45]][2] |-> {%[[SSA90]], %[[SSA93]][2]}
// CHECK:   %[[SSA91]] |-> {%[[SSA90]], %[[arg45]][2], %[[SSA93]][2]}
// CHECK: maybeSameAs end

mgp.model @repeat1 : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %arg0 : !mef.chain, !mgp.context, !mgp.opaque
} execute (%ch0: !mef.chain, %cxt: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, !mgp.tensor<f32>) {
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
// CHECK:   %[[SSA85:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA86:.*]] = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %[[SSA88:.*]] = "mgp.runtime.create"(%[[SSA85]], %[[SSA86]])
// CHECK:   "mef.output"(%[[SSA88]], %[[SSA86]]
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%[[arg40:[a-z0-9]*]]: !mef.chain, %[[arg41:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA81:.*]] = "some.op"(%[[arg40]]) : (!mef.chain) -> si32
// CHECK:   %[[SSA82:.*]] = "some.op"(%[[arg40]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA83:.*]] = "grt.repeat.i32"(%[[SSA81]], %[[SSA82]]) ({
// ### Block 2 ###
// CHECK:   ^bb0(%[[arg42:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     "mef.output"(%[[arg42]]) : (!mgp.tensor<f32>) -> ()
// CHECK:   }) : (si32, !mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   %[[SSA84:.*]] = "some.op"(%[[SSA83]]) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%[[arg40]], %[[SSA84]]) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: })

// CHECK: maybeSameAs begin
// CHECK:   %[[arg42]][2] |-> {%[[SSA82]], %[[arg42]][2]}
// CHECK:   %[[SSA83]] |-> {%[[SSA82]], %[[arg42]][2]}
// CHECK: maybeSameAs end

mgp.model @repeat2 : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %arg0 : !mef.chain, !mgp.context, !mgp.opaque
} execute (%ch0: !mef.chain, %cxt: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, !mgp.tensor<f32>) {
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
// CHECK:   %[[SSA77:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA78:.*]] = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %[[SSA80:.*]] = "mgp.runtime.create"(%[[SSA77]], %[[SSA78]])
// CHECK:   "mef.output"(%[[SSA80]], %[[SSA78]]
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%[[arg36:[a-z0-9]*]]: !mef.chain, %[[arg37:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA72:.*]] = "some.op"(%[[arg36]]) : (!mef.chain) -> si32
// CHECK:   %[[SSA73:.*]] = "some.op"(%[[arg36]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA74:.*]]:2 = "grt.repeat.i32"(%[[SSA72]], %[[SSA73]], %[[SSA73]]) ({
// ### Block 2 ###
// CHECK:   ^bb0(%[[arg38:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg39:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     %[[SSA76:.*]] = "some.op"(%[[arg38]]) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:     "mef.output"(%[[SSA76]], %[[arg39]]) : (!mgp.tensor<f32>, !mgp.tensor<f32>) -> ()
// CHECK:   }) : (si32, !mgp.tensor<f32>, !mgp.tensor<f32>) -> (!mgp.tensor<f32>, !mgp.tensor<f32>)
// CHECK:   %[[SSA75:.*]] = "some.op"(%[[SSA74]]#0) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%[[arg36]], %[[SSA75]]) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: })

// CHECK: maybeSameAs begin
// CHECK:   %[[arg38]][2] |-> {%[[SSA73]], %[[SSA76]][2]}
// CHECK:   %[[arg39]][2] |-> {%[[SSA73]], %[[arg39]][2]}
// CHECK:   %[[SSA74]]#0 |-> {%[[SSA73]], %[[arg38]][2], %[[SSA76]][2]}
// CHECK:   %[[SSA74]]#1 |-> {%[[SSA73]], %[[arg39]][2]}
// CHECK: maybeSameAs end

mgp.model @repeat3 : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %arg0 : !mef.chain, !mgp.context, !mgp.opaque
} execute (%ch0: !mef.chain, %cxt: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, !mgp.tensor<f32>) {
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
// CHECK:   %[[SSA68:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA69:.*]] = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %[[SSA71:.*]] = "mgp.runtime.create"(%[[SSA68]], %[[SSA69]])
// CHECK:   "mef.output"(%[[SSA71]], %[[SSA69]]
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%[[arg26:[a-z0-9]*]]: !mef.chain, %[[arg27:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA60:.*]] = "some.op"(%[[arg26]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA61:.*]] = "some.op"(%[[arg26]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA62:.*]]:4 = "grt.while"(%[[arg26]], %[[SSA60]], %[[SSA61]], %[[SSA61]]) ({
// ### Block 2 ###
// CHECK:   ^bb0(%[[arg32:[a-z0-9]*]]: !mef.chain, %[[arg33:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg34:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg35:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     %[[SSA65:.*]] = "some.op"(%[[arg32]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:     %[[SSA66:.*]] = "some.op"(%[[arg34]]) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:     %[[SSA67:.*]] = "some.op"(%[[arg32]]) : (!mef.chain) -> i1
// CHECK:     "mef.output"(%[[arg32]], %[[SSA65]], %[[SSA66]], %[[arg35]], %[[SSA67]]) : (!mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>, i1) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%[[arg28:[a-z0-9]*]]: !mef.chain, %[[arg29:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg30:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg31:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     %[[SSA64:.*]] = "some.op"(%[[arg30]]) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:     "mef.output"(%[[arg28]], %[[arg29]], %[[SSA64]], %[[arg31]]) : (!mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> ()
// CHECK:   }) : (!mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> (!mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>)
// CHECK:   %[[SSA63:.*]] = "some.op"(%[[SSA62]]#1, %[[SSA62]]#2, %[[SSA62]]#3) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%[[SSA62]]#0, %[[SSA63]]) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: })

// CHECK: sameAs begin
// CHECK:   %[[SSA62]]#0 |-> {%[[SSA71]][0], %[[arg26]], %[[arg32]][2], %[[arg28]][3]}
// CHECK:   %[[SSA62]]#1 |-> {%[[SSA65]][2], %[[arg29]][3]}
// CHECK:   %[[SSA62]]#2 |-> {%[[SSA66]][2], %[[arg30]][3]}
// CHECK:   %[[SSA62]]#3 |-> {%[[SSA61]], %[[arg35]][2], %[[arg31]][3]}
// CHECK: sameAs end

// CHECK: maybeSameAs begin
// CHECK:   %[[SSA62]]#0 |-> {%[[SSA71]][0], %[[arg26]], %[[arg32]][2], %[[arg28]][3]}
// CHECK:   %[[SSA62]]#1 |-> {%[[SSA65]][2], %[[arg29]][3]}
// CHECK:   %[[SSA62]]#2 |-> {%[[SSA66]][2], %[[arg30]][3]}
// CHECK:   %[[SSA62]]#3 |-> {%[[SSA61]], %[[arg35]][2], %[[arg31]][3]}
// CHECK: maybeSameAs end

// CHECK: impliesReady begin
// CHECK:   %[[SSA64]][3] |-> {%[[SSA68]][0], %[[SSA69]][0], %[[SSA71]][0], %[[arg26]], %[[SSA61]], %[[arg34]][2], %[[SSA66]][2], %[[arg30]][3]}
// CHECK:   %[[SSA62]]#0 |-> {%[[SSA68]][0], %[[SSA69]][0], %[[SSA71]][0], %[[arg26]], %[[SSA61]], %[[arg32]][2], %[[arg34]][2], %[[arg35]][2], %[[SSA65]][2], %[[SSA66]][2], %[[arg28]][3], %[[arg29]][3], %[[arg30]][3], %[[arg31]][3], %[[SSA64]][3]}
// CHECK: impliesReady end

// No branches, so this relation is empty.
// CHECK: structurallyAtMostOne begin
// CHECK-NEXT: structurallyAtMostOne end

// But the loop cond and body blocks are known not to overlap.
// CHECK: structurallyCannotOverlapRelation begin
// CHECK:   %[[SSA65]][2] |-> {%[[SSA64]][3]}
// CHECK:   %[[SSA66]][2] |-> {%[[SSA64]][3]}
// CHECK:   %[[SSA67]][2] |-> {%[[SSA64]][3]}
// CHECK:   %[[SSA64]][3] |-> {%[[SSA65]][2], %[[SSA66]][2], %[[SSA67]][2]}
// CHECK: structurallyCannotOverlapRelation end

// CHECK: runsLater begin
// CHECK:   %[[SSA65]][2] |-> {%[[SSA63]]}
// CHECK: runsLater end

mgp.model @simple : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %arg0 : !mef.chain, !mgp.context, !mgp.opaque
} execute (%ch0: !mef.chain, %cxt: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, !mgp.tensor<f32>) {
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
// CHECK:   %[[SSA56:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA57:.*]] = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %[[SSA59:.*]] = "mgp.runtime.create"(%[[SSA56]], %[[SSA57]])
// CHECK:   "mef.output"(%[[SSA59]], %[[SSA57]]
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%[[arg20:[a-z0-9]*]]: !mef.chain, %[[arg21:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA51:.*]] = "some.op"(%[[arg20]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA52:.*]] = "some.op"(%[[arg20]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA53:.*]]:2 = "grt.while"(%[[SSA51]], %[[SSA52]]) ({
// ### Block 2 ###
// CHECK:   ^bb0(%[[arg24:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg25:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     %[[SSA55:.*]] = "some.op"() : () -> i1
// CHECK:     "mef.output"(%[[arg24]], %[[arg25]], %[[SSA55]]) : (!mgp.tensor<f32>, !mgp.tensor<f32>, i1) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%[[arg22:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg23:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     "mef.output"(%[[arg23]], %[[arg22]]) : (!mgp.tensor<f32>, !mgp.tensor<f32>) -> ()
// CHECK:   }) : (!mgp.tensor<f32>, !mgp.tensor<f32>) -> (!mgp.tensor<f32>, !mgp.tensor<f32>)
// CHECK:   %[[SSA54:.*]] = "some.op"(%[[SSA53]]#0, %[[SSA53]]#1) : (!mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%[[arg20]], %[[SSA54]]) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: })

// CHECK: sameAs begin
// CHECK:   %[[SSA53]]#0 |-> {%[[arg24]][2], %[[arg22]][3]}
// CHECK:   %[[SSA53]]#1 |-> {%[[arg25]][2], %[[arg23]][3]}
// CHECK: sameAs end

// CHECK: maybeSameAs begin
// CHECK:   %[[SSA53]]#0 |-> {%[[SSA51]], %[[SSA52]], %[[arg24]][2], %[[arg25]][2], %[[arg22]][3], %[[arg23]][3]}
// CHECK:   %[[SSA53]]#1 |-> {%[[SSA51]], %[[SSA52]], %[[arg24]][2], %[[arg25]][2], %[[arg22]][3], %[[arg23]][3]}
// CHECK: maybeSameAs end

mgp.model @crissCross : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %arg0 : !mef.chain, !mgp.context, !mgp.opaque
} execute (%ch0: !mef.chain, %cxt: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, !mgp.tensor<f32>) {
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
// CHECK:   %[[SSA47:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA48:.*]] = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %[[SSA50:.*]] = "mgp.runtime.create"(%[[SSA47]], %[[SSA48]])
// CHECK:   "mef.output"(%[[SSA50]], %[[SSA48]]
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%[[arg12:[a-z0-9]*]]: !mef.chain, %[[arg13:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA42:.*]] = "some.op"(%[[arg12]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA43:.*]]:3 = "grt.while"(%[[SSA42]], %[[SSA42]], %[[SSA42]]) ({
// ### Block 2 ###
// CHECK:   ^bb0(%[[arg17:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg18:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg19:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     %[[SSA46:.*]] = "some.op"() : () -> i1
// CHECK:     "mef.output"(%[[arg17]], %[[arg18]], %[[arg19]], %[[SSA46]]) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>, i1) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%[[arg14:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg15:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg16:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     %[[SSA45:.*]] = "some.op"() : () -> !mgp.tensor<f32>
// CHECK:     "mef.output"(%[[arg15]], %[[arg16]], %[[SSA45]]) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> ()
// CHECK:   }) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>)
// CHECK:   %[[SSA44:.*]] = "some.op"(%[[SSA43]]#0, %[[SSA43]]#1, %[[SSA43]]#2) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%[[arg12]], %[[SSA44]]) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: })

// CHECK: sameAs begin
// CHECK:   %[[SSA43]]#0 |-> {%[[arg17]][2], %[[arg14]][3]}
// CHECK:   %[[SSA43]]#1 |-> {%[[arg18]][2], %[[arg15]][3]}
// CHECK:   %[[SSA43]]#2 |-> {%[[arg19]][2], %[[arg16]][3]}
// CHECK: sameAs end

// CHECK: maybeSameAs begin
// CHECK:   %[[SSA43]]#0 |-> {%[[SSA42]], %[[arg17]][2], %[[arg18]][2], %[[arg19]][2], %[[arg14]][3], %[[arg15]][3], %[[arg16]][3], %[[SSA45]][3]}
// CHECK:   %[[SSA43]]#1 |-> {%[[SSA42]], %[[arg18]][2], %[[arg19]][2], %[[arg15]][3], %[[arg16]][3], %[[SSA45]][3]}
// CHECK:   %[[SSA43]]#2 |-> {%[[SSA42]], %[[arg19]][2], %[[arg16]][3], %[[SSA45]][3]}
// CHECK: maybeSameAs end

// CHECK: impliesReady begin
// CHECK:   %[[SSA44]] |-> {%[[arg17]][2], %[[arg18]][2], %[[arg19]][2], %[[arg14]][3], %[[arg15]][3], %[[arg16]][3], %[[SSA45]][3], %[[SSA43]]#0, %[[SSA43]]#1, %[[SSA43]]#2}
// CHECK: impliesReady end

mgp.model @shift : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %arg0 : !mef.chain, !mgp.context, !mgp.opaque
} execute (%ch0: !mef.chain, %cxt: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, !mgp.tensor<f32>) {
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
// CHECK:   %[[SSA38:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA39:.*]] = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %[[SSA41:.*]] = "mgp.runtime.create"(%[[SSA38]], %[[SSA39]])
// CHECK:   "mef.output"(%[[SSA41]], %[[SSA39]]
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%[[arg0:[a-z0-9]*]]: !mef.chain, %[[arg1:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA0:.*]] = "mgp.chain.host_to_device"(%[[arg0]], %[[arg1]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mgp.chain
// CHECK:   %[[SSA1:.*]] = "mgp.buffer.constant"(%[[arg0]], %[[arg1]]) {aRuntimeSlot = 1 : ui64, bValue = #M.aligned_bytes<"0x00000004", align 8> : !M.array<4xui8>, cRawAlign = 8 : ui64, dDevice = #M.device_ref<"cpu", 0>, eIsTrivial = true} : (!mef.chain, !mgp.context) -> !mgp.buffer<4>
// CHECK:   %[[SSA2:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<> : !M.array<0xui64>, bRawDType = 138 : ui8} : () -> !mgp.tensor_spec<ui32>
// CHECK:   %[[SSA3:.*]] = "mgp.tensor.create"(%[[SSA0]], %[[arg1]], %[[SSA1]], %[[SSA2]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<4>, !mgp.tensor_spec<ui32>) -> !mgp.tensor<ui32>
// CHECK:   %[[SSA4:.*]] = "mgp.buffer.alloc.static"(%[[SSA0]], %[[arg1]]) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %[[SSA5:.*]] = "initialize"(%[[SSA4]]) {tempInOutSig = #M.inout_sig<"o">} : (!mgp.buffer<32>) -> !mgp.chain
// CHECK:   %[[SSA6:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<32> : !M.array<1xui64>, bRawDType = 134 : ui8} : () -> !mgp.tensor_spec<32xui8>
// CHECK:   %[[SSA7:.*]] = "mgp.tensor.create"(%[[SSA5]], %[[arg1]], %[[SSA4]], %[[SSA6]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mgp.tensor<32xui8>
// CHECK:   %[[SSA8:.*]] = "mgp.buffer.alloc.static"(%[[SSA0]], %[[arg1]]) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %[[SSA9:.*]] = "initialize"(%[[SSA8]]) {tempInOutSig = #M.inout_sig<"o">} : (!mgp.buffer<32>) -> !mgp.chain
// CHECK:   %[[SSA10:.*]] = "mgp.tensor.create"(%[[SSA9]], %[[arg1]], %[[SSA8]], %[[SSA6]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mgp.tensor<32xui8>
// CHECK:   %[[SSA11:.*]]:5 = "grt.while"(%[[SSA0]], %[[SSA3]], %[[SSA7]], %[[SSA10]], %[[arg1]]) ({
// ### Block 2 ###
// CHECK:   ^bb0(%[[arg7:[a-z0-9]*]]: !mgp.chain, %[[arg8:[a-z0-9]*]]: !mgp.tensor<ui32>, %[[arg9:[a-z0-9]*]]: !mgp.tensor<32xui8>, %[[arg10:[a-z0-9]*]]: !mgp.tensor<32xui8>, %[[arg11:[a-z0-9]*]]: !mgp.context):
// CHECK:     %[[SSA29:.*]] = "mgp.buffer.constant"(%[[SSA37:.*]], %[[arg11]]) {aRuntimeSlot = 1 : ui64, bValue = #M.aligned_bytes<"0x00000000", align 8> : !M.array<4xui8>, cRawAlign = 8 : ui64, dDevice = #M.device_ref<"cpu", 0>, eIsTrivial = true} : (!mef.chain, !mgp.context) -> !mgp.buffer<4>
// CHECK:     %[[SSA30:.*]] = "mgp.tensor.extract.buffer"(%[[arg11]], %[[arg8]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<ui32>) -> !mgp.buffer<4>
// CHECK:     %[[SSA31:.*]] = "mgp.buffer.alloc.static"(%[[arg7]], %[[arg11]]) {aRuntimeSlot = 1 : ui64, bSize = 1 : ui64, cRawAlign = 32 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<1>
// CHECK:     %[[SSA32:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<> : !M.array<0xui64>, bRawDType = 1 : ui8} : () -> !mgp.tensor_spec<bool>
// CHECK:     %[[SSA33:.*]] = "greater"(%[[arg7]], %[[SSA30]], %[[SSA29]], %[[SSA31]]) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<4>, !mgp.buffer<4>, !mgp.buffer<1>) -> !mgp.chain
// CHECK:     %[[SSA34:.*]] = "mgp.tensor.create"(%[[SSA33]], %[[arg11]], %[[SSA31]], %[[SSA32]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<1>, !mgp.tensor_spec<bool>) -> !mgp.tensor<bool>
// CHECK:     %[[SSA35:.*]] = "mgp.tensor.extract.buffer"(%[[arg11]], %[[SSA34]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<bool>) -> !mgp.buffer<1>
// CHECK:     %[[SSA36:.*]] = "mgp.buffer.to_bool"(%[[arg7]], %[[arg11]], %[[SSA35]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<1>) -> i1
// CHECK:     %[[SSA37]] = "mgp.chain.device_to_host"(%[[arg7]], %[[arg11]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mef.chain
// CHECK:     "mef.output"(%[[arg7]], %[[arg8]], %[[arg9]], %[[arg10]], %[[arg11]], %[[SSA36]]) : (!mgp.chain, !mgp.tensor<ui32>, !mgp.tensor<32xui8>, !mgp.tensor<32xui8>, !mgp.context, i1) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%[[arg2:[a-z0-9]*]]: !mgp.chain, %[[arg3:[a-z0-9]*]]: !mgp.tensor<ui32>, %[[arg4:[a-z0-9]*]]: !mgp.tensor<32xui8>, %[[arg5:[a-z0-9]*]]: !mgp.tensor<32xui8>, %[[arg6:[a-z0-9]*]]: !mgp.context):
// CHECK:     %[[SSA16:.*]] = "mgp.tensor.extract.buffer"(%[[arg6]], %[[arg4]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<32xui8>) -> !mgp.buffer<32>
// CHECK:     %[[SSA17:.*]] = "mgp.tensor.extract.buffer"(%[[arg6]], %[[arg5]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<32xui8>) -> !mgp.buffer<32>
// CHECK:     %[[SSA18:.*]] = "mgp.buffer.alloc.static"(%[[arg2]], %[[arg6]]) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:     %[[SSA19:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<32> : !M.array<1xui64>, bRawDType = 134 : ui8} : () -> !mgp.tensor_spec<32xui8>
// CHECK:     %[[SSA20:.*]] = "add"(%[[arg2]], %[[SSA16]], %[[SSA17]], %[[SSA18]]) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
// CHECK:     %[[SSA21:.*]] = "mgp.tensor.create"(%[[SSA20]], %[[arg6]], %[[SSA18]], %[[SSA19]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mgp.tensor<32xui8>
// CHECK:     %[[SSA22:.*]] = "mgp.buffer.constant"(%[[SSA28:.*]], %[[arg6]]) {aRuntimeSlot = 1 : ui64, bValue = #M.aligned_bytes<"0x00000001", align 8> : !M.array<4xui8>, cRawAlign = 8 : ui64, dDevice = #M.device_ref<"cpu", 0>, eIsTrivial = true} : (!mef.chain, !mgp.context) -> !mgp.buffer<4>
// CHECK:     %[[SSA23:.*]] = "mgp.tensor.extract.buffer"(%[[arg6]], %[[arg3]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<ui32>) -> !mgp.buffer<4>
// CHECK:     %[[SSA24:.*]] = "mgp.buffer.alloc.static"(%[[arg2]], %[[arg6]]) {aRuntimeSlot = 1 : ui64, bSize = 4 : ui64, cRawAlign = 32 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<4>
// CHECK:     %[[SSA25:.*]] = "sub"(%[[arg2]], %[[SSA23]], %[[SSA22]], %[[SSA24]]) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<4>, !mgp.buffer<4>, !mgp.buffer<4>) -> !mgp.chain
// CHECK:     %[[SSA26:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<> : !M.array<0xui64>, bRawDType = 138 : ui8} : () -> !mgp.tensor_spec<ui32>
// CHECK:     %[[SSA27:.*]] = "mgp.tensor.create"(%[[SSA25]], %[[arg6]], %[[SSA24]], %[[SSA26]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<4>, !mgp.tensor_spec<ui32>) -> !mgp.tensor<ui32>
// CHECK:     %[[SSA28]] = "mgp.chain.device_to_host"(%[[arg2]], %[[arg6]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mef.chain
// CHECK:     "mef.output"(%[[arg2]], %[[SSA27]], %[[SSA21]], %[[arg5]], %[[arg6]]) : (!mgp.chain, !mgp.tensor<ui32>, !mgp.tensor<32xui8>, !mgp.tensor<32xui8>, !mgp.context) -> ()
// CHECK:   }) : (!mgp.chain, !mgp.tensor<ui32>, !mgp.tensor<32xui8>, !mgp.tensor<32xui8>, !mgp.context) -> (!mgp.chain, !mgp.tensor<ui32>, !mgp.tensor<32xui8>, !mgp.tensor<32xui8>, !mgp.context)
// CHECK:   %[[SSA12:.*]] = "mgp.tensor.extract.buffer"(%[[arg1]], %[[SSA11]]#2) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<32xui8>) -> !mgp.buffer<32>
// CHECK:   %[[SSA13:.*]] = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<32> : !M.array<1xui64>, bRawDType = 134 : ui8} : () -> !mgp.tensor_spec<32xui8>
// CHECK:   %[[SSA14:.*]] = "mgp.tensor.create"(%[[SSA0]], %[[arg1]], %[[SSA12]], %[[SSA13]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mgp.tensor<32xui8>
// CHECK:   %[[SSA15:.*]] = "mgp.chain.device_to_host"(%[[SSA11]]#0, %[[arg1]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%[[SSA15]], %[[SSA14]]) : (!mef.chain, !mgp.tensor<32xui8>) -> ()
// CHECK: })

// CHECK: sameAs begin
// CHECK:   %[[SSA11]]#0 |-> {%[[SSA0]], %[[arg7]][2], %[[arg2]][3]}
// CHECK:   %[[SSA11]]#1 |-> {%[[arg8]][2], %[[arg3]][3]}
// CHECK:   %[[SSA11]]#2 |-> {%[[arg9]][2], %[[arg4]][3]}
// CHECK:   %[[SSA11]]#3 |-> {%[[SSA10]], %[[arg10]][2], %[[arg5]][3]}
// CHECK:   %[[SSA11]]#4 |-> {%[[SSA39]][0], %[[arg1]], %[[arg11]][2], %[[arg6]][3]}
// CHECK: sameAs end

// CHECK: impliesReady begin
// CHECK:   %[[SSA10]] |-> {%[[SSA38]][0], %[[SSA39]][0], %[[SSA41]][0], %[[arg0]], %[[arg1]], %[[SSA0]], %[[SSA6]], %[[SSA8]], %[[SSA9]]}
// CHECK:   %[[arg8]][2] |-> {%[[SSA38]][0], %[[SSA39]][0], %[[SSA41]][0], %[[arg0]], %[[arg1]], %[[SSA0]], %[[SSA1]], %[[SSA2]], %[[SSA3]]}
// CHECK:   %[[arg9]][2] |-> {%[[SSA38]][0], %[[SSA39]][0], %[[SSA41]][0], %[[arg0]], %[[arg1]], %[[SSA0]], %[[SSA4]], %[[SSA5]], %[[SSA6]], %[[SSA7]]}
// CHECK:   %[[SSA11]]#0 |-> {%[[SSA38]][0], %[[SSA39]][0], %[[SSA41]][0], %[[arg0]], %[[arg1]], %[[SSA0]], %[[SSA1]], %[[SSA2]], %[[SSA3]], %[[SSA4]], %[[SSA5]], %[[SSA6]], %[[SSA7]], %[[SSA8]], %[[SSA9]], %[[SSA10]], %[[arg7]][2], %[[arg8]][2], %[[arg9]][2], %[[arg10]][2], %[[arg11]][2], %[[arg2]][3], %[[arg3]][3], %[[arg4]][3], %[[arg5]][3], %[[arg6]][3], %[[SSA16]][3], %[[SSA17]][3], %[[SSA18]][3], %[[SSA19]][3], %[[SSA20]][3], %[[SSA21]][3], %[[SSA22]][3], %[[SSA23]][3], %[[SSA24]][3], %[[SSA25]][3], %[[SSA26]][3], %[[SSA27]][3], %[[SSA28]][3]}
// CHECK:   %[[SSA14]] |-> {%[[SSA38]][0], %[[SSA39]][0], %[[SSA41]][0], %[[arg0]], %[[arg1]], %[[SSA0]], %[[SSA1]], %[[SSA2]], %[[SSA3]], %[[SSA4]], %[[SSA5]], %[[SSA6]], %[[SSA7]], %[[SSA8]], %[[SSA9]], %[[SSA10]], %[[arg7]][2], %[[arg8]][2], %[[arg9]][2], %[[arg10]][2], %[[arg11]][2], %[[arg2]][3], %[[arg3]][3], %[[arg4]][3], %[[arg5]][3], %[[arg6]][3], %[[SSA16]][3], %[[SSA17]][3], %[[SSA18]][3], %[[SSA19]][3], %[[SSA20]][3], %[[SSA21]][3], %[[SSA22]][3], %[[SSA23]][3], %[[SSA24]][3], %[[SSA25]][3], %[[SSA26]][3], %[[SSA27]][3], %[[SSA28]][3], %[[SSA11]]#2, %[[SSA12]], %[[SSA13]]}
// CHECK: impliesReady end

// The tensor.{create,extract} aliasing excludes entries.
// CHECK: exclusive begin
// CHECK:   %[[SSA21]][3] |-> {%[[SSA38]][0], %[[SSA5]], %[[SSA6]], %[[SSA9]]}
// CHECK: exclusive end

mgp.model @while : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %arg0 : !mef.chain, !mgp.context, !mgp.opaque
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, !mgp.tensor<32xui8>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %0 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x00000004", align 8> : !M.array<4xui8>
  %1 = mgp.tensor_spec.create  : !mgp.tensor_spec<ui32>
  %2 = mgp.tensor.create %ch1, %ctx[1], %0, %1  : !mgp.tensor<ui32>
  %3 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %4 = "initialize"(%3) {tempInOutSig=#M.inout_sig<"o">} : (!mgp.buffer<32>) -> !mgp.chain
  %5 = mgp.tensor_spec.create  : !mgp.tensor_spec<32xui8>
  %6 = mgp.tensor.create %4, %ctx[1], %3, %5  : !mgp.tensor<32xui8>
  %7 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %8 = "initialize"(%7) {tempInOutSig=#M.inout_sig<"o">} : (!mgp.buffer<32>) -> !mgp.chain
  %9 = mgp.tensor.create %8, %ctx[1],  %7, %5  : !mgp.tensor<32xui8>
  %10:5 = grt.while (%ch1: !mgp.chain, %2: !mgp.tensor<ui32>, %6: !mgp.tensor<32xui8>, %9: !mgp.tensor<32xui8>, %ctx: !mgp.context) {
    %12 = mgp.buffer.constant %ch1h, %ctx[1], #M.aligned_bytes<"0x00000000", align 8> : !M.array<4xui8>
    %bufferResult_0 = mgp.tensor.extract.buffer %ctx[1], %2 : !mgp.tensor<ui32>
    %13 = mgp.buffer.alloc.static %ch1, %ctx[1], align 32 : !mgp.buffer<1>
    %14 = mgp.tensor_spec.create  : !mgp.tensor_spec<bool>
    %15 = "greater"(%ch1, %bufferResult_0, %12, %13) {tempInOutSig=#M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<4>, !mgp.buffer<4>, !mgp.buffer<1>) -> !mgp.chain
    %16 = mgp.tensor.create %15, %ctx[1],  %13, %14  : !mgp.tensor<bool>
    %bufferResult_2 = mgp.tensor.extract.buffer %ctx[1], %16 : !mgp.tensor<bool>
    %17 = mgp.buffer.to_bool %ch1, %ctx[1], %bufferResult_2 : !mgp.buffer<1>
    %ch1h = mgp.chain.device_to_host %ch1, %ctx[1]
    mef.output %ch1, %2, %6, %9, %ctx, %17 : !mgp.chain, !mgp.tensor<ui32>, !mgp.tensor<32xui8>, !mgp.tensor<32xui8>, !mgp.context, i1
  } do {
    %bufferResult_0 = mgp.tensor.extract.buffer %ctx[1], %6 : !mgp.tensor<32xui8>
    %bufferResult_2 = mgp.tensor.extract.buffer %ctx[1], %9 : !mgp.tensor<32xui8>
    %12 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
    %13 = mgp.tensor_spec.create  : !mgp.tensor_spec<32xui8>
    %14 = "add"(%ch1, %bufferResult_0, %bufferResult_2, %12) {tempInOutSig=#M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
    %15 = mgp.tensor.create %14, %ctx[1],  %12, %13  : !mgp.tensor<32xui8>
    %16 = mgp.buffer.constant %ch1h, %ctx[1], #M.aligned_bytes<"0x00000001", align 8> : !M.array<4xui8>
    %bufferResult_4 = mgp.tensor.extract.buffer %ctx[1], %2 : !mgp.tensor<ui32>
    %17 = mgp.buffer.alloc.static %ch1, %ctx[1], align 32 : !mgp.buffer<4>
    %18 = "sub"(%ch1, %bufferResult_4, %16, %17) {tempInOutSig=#M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<4>, !mgp.buffer<4>, !mgp.buffer<4>) -> !mgp.chain
    %tensorSpecResult_5 = mgp.tensor_spec.create : !mgp.tensor_spec<ui32>
    %19 = mgp.tensor.create %18, %ctx[1],  %17, %tensorSpecResult_5  : !mgp.tensor<ui32>
    %ch1h = mgp.chain.device_to_host %ch1, %ctx[1]
    mef.output %ch1, %19, %15, %9, %ctx : !mgp.chain, !mgp.tensor<ui32>, !mgp.tensor<32xui8>, !mgp.tensor<32xui8>, !mgp.context
  }
  %bufferResult = mgp.tensor.extract.buffer %ctx[1], %10#2 : !mgp.tensor<32xui8>
  %tensorSpecResult = mgp.tensor_spec.create : !mgp.tensor_spec<32xui8>
  %11 = mgp.tensor.create %ch1, %ctx[1],  %bufferResult, %tensorSpecResult : !mgp.tensor<32xui8>
  %ch2 = mgp.chain.device_to_host %10#0, %ctx[1]
  mef.output %ch2, %11 : !mef.chain, !mgp.tensor<32xui8>
}
