// RUN: mo-opt %s -allow-unregistered-dialect --lower-primitives | FileCheck %s

//--- Test lowering of MIP Index Primitives

//--- index.constant will be tested throughtout the tests ---/

// CHECK-LABEL: mgp.model @mip_add
// CHECK: mogg.call "index.add"
mgp.model @mip_add : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant -20
  %res1 = index.add %lhs, %rhs
  %ch1 = "test.sink"(%res1) : (index) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_div
// CHECK: mogg.call "index.divs"
mgp.model @mip_div : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant -2
  %res1 = index.divs %lhs, %rhs
  %ch1 = "test.sink"(%res1) : (index) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_div_ceil
// CHECK: mogg.call "index.ceildivs"
mgp.model @mip_div_ceil : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant -4
  %res1 = index.ceildivs %lhs, %rhs
  %ch1 = "test.sink"(%res1) : (index) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_cmp_eq
// CHECK: mogg.call "index.cmp"({{.*}}) {pred = "eq"}
mgp.model @mip_cmp_eq : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 4
  %rhs = index.constant 4
  %res1 = index.cmp eq(%lhs, %rhs)
  %lhs1 = index.constant 2
  %res2 = index.cmp eq(%lhs1, %rhs)
  mef.output %ch0 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_cmp_lt
// CHECK: mogg.call "index.cmp"({{.*}}) {pred = "slt"}
mgp.model @mip_cmp_lt : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 4
  %rhs = index.constant 4
  %res1 = index.cmp slt(%lhs, %rhs)
  %lhs1 = index.constant 2
  %res2 = index.cmp slt(%lhs1, %rhs)
  mef.output %ch0 : !mef.chain
}


// CHECK-LABEL: mgp.model @mip_cmp_le
// CHECK: mogg.call "index.cmp"({{.*}}) {pred = "sle"}
mgp.model @mip_cmp_le : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 5
  %rhs = index.constant 4
  %res1 = index.cmp sle(%lhs, %rhs)
  %lhs1 = index.constant 4
  %res2 = index.cmp sle(%lhs1, %rhs)
  mef.output %ch0 : !mef.chain
}


// CHECK-LABEL: mgp.model @mip_max
// CHECK: mogg.call "index.maxs"
mgp.model @mip_max : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant -2
  %res1 = index.maxs %lhs, %rhs
  %ch1 = "test.sink"(%res1) : (index) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_min
// CHECK: mogg.call "index.mins"
mgp.model @mip_min : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant -2
  %res1 = index.mins %lhs, %rhs
  %ch1 = "test.sink"(%res1) : (index) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_mod
// CHECK: mogg.call "index.rems"
mgp.model @mip_mod : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant 4
  %res1 = index.rems %lhs, %rhs
  %ch1 = "test.sink"(%res1) : (index) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_mul
// CHECK: mogg.call "index.mul"
mgp.model @mip_mul : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant 4
  %res1 = index.mul %lhs, %rhs
  %ch1 = "test.sink"(%res1) : (index) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_and
// CHECK: mogg.call "arith.andi"
mgp.model @mip_and : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, i1) {
  %ch0 = grt.chain.create ()
  %zero = index.constant 0
  %one = index.constant 1
  %lhs = index.bool.constant true
  %rhs = index.bool.constant true
  %res1 = arith.andi %lhs, %rhs: i1
  mef.output %ch0, %res1: !mef.chain, i1
}

// CHECK-LABEL: mgp.model @mip_or
// CHECK: mogg.call "arith.ori"
mgp.model @mip_or : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, i1) {
  %ch0 = grt.chain.create ()
  %zero = index.constant 0
  %one = index.constant 1
  %lhs = index.bool.constant true
  %rhs = index.bool.constant false
  %res1 = arith.ori %lhs, %rhs: i1
  mef.output %ch0, %res1: !mef.chain, i1
}

// CHECK-LABEL: mgp.model @mip_xor
// CHECK: mogg.call "arith.xori"
mgp.model @mip_xor : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, i1) {
  %ch0 = grt.chain.create ()
  %zero = index.constant 0
  %one = index.constant 1
  %lhs = index.bool.constant true
  %rhs = index.bool.constant false
  %res1 = arith.xori %lhs, %rhs: i1
  mef.output %ch0, %res1: !mef.chain, i1
}

// CHECK-LABEL: mgp.model @mip_select
// CHECK: mogg.call "arith.select"
mgp.model @mip_select : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain) {
  %ch0 = grt.chain.create ()
  %zero = index.constant 0
  %one = index.constant 1
  %cond = index.bool.constant true
  %res1 = arith.select %cond, %one, %zero : index
  %ch1 = "test.sink"(%res1) : (index) -> !mef.chain
  mef.output %ch1: !mef.chain
}
