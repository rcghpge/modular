// RUN: mo-opt %s --lower-primitives | FileCheck %s

//--- Test lowering of MIP Index Primitives

//--- index.constant/mip.print.index will be tested throughtout the tests ---/

// CHECK-LABEL: mgp.model @mip_basic
// CHECK: mogg.call "index.constant"
// CHECK: mogg.call "mip.print.index"
mgp.model @mip_basic : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %ch1 = mip.print.index %lhs, %ch0
  mef.output %ch1 : !mef.chain
}

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
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_div
// CHECK: mogg.call "mip.div"
mgp.model @mip_div : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant -2
  %res1 = mip.div %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_div_ceil
// CHECK: mogg.call "mip.div.ceil"
mgp.model @mip_div_ceil : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant -4
  %res1 = mip.div.ceil %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_cmp_eq
// CHECK: mogg.call "mip.cmp.eq"
mgp.model @mip_cmp_eq : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 4
  %rhs = index.constant 4
  %res1 = mip.cmp.eq %lhs, %rhs
  %lhs1 = index.constant 2
  %res2 = mip.cmp.eq %lhs1, %rhs
  mef.output %ch0 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_cmp_lt
// CHECK: mogg.call "mip.cmp.lt"
mgp.model @mip_cmp_lt : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 4
  %rhs = index.constant 4
  %res1 = mip.cmp.lt %lhs, %rhs
  %lhs1 = index.constant 2
  %res2 = mip.cmp.lt %lhs1, %rhs
  mef.output %ch0 : !mef.chain
}


// CHECK-LABEL: mgp.model @mip_cmp_le
// CHECK: mogg.call "mip.cmp.le"
mgp.model @mip_cmp_le : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 5
  %rhs = index.constant 4
  %res1 = mip.cmp.le %lhs, %rhs
  %lhs1 = index.constant 4
  %res2 = mip.cmp.le %lhs1, %rhs
  mef.output %ch0 : !mef.chain
}


// CHECK-LABEL: mgp.model @mip_max
// CHECK: mogg.call "mip.max"
mgp.model @mip_max : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant -2
  %res1 = mip.max %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_min
// CHECK: mogg.call "mip.min"
mgp.model @mip_min : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant -2
  %res1 = mip.min %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_mod
// CHECK: mogg.call "mip.mod"
mgp.model @mip_mod : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant 4
  %res1 = mip.mod %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
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
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_nary_mul
// CHECK: mogg.call "mip.nary.mul"
mgp.model @mip_nary_mul : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %a = index.constant 3
  %b = index.constant -5
  %c = index.constant 7
  %res1 = mip.nary.mul (%a, %b, %c)
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_and
// CHECK: mogg.call "mip.bool.and"
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
  %res1 = mip.bool.and %lhs, %rhs
  mef.output %ch0, %res1: !mef.chain, i1
}

// CHECK-LABEL: mgp.model @mip_or
// CHECK: mogg.call "mip.bool.or"
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
  %res1 = mip.bool.or %lhs, %rhs
  mef.output %ch0, %res1: !mef.chain, i1
}

// CHECK-LABEL: mgp.model @mip_xor
// CHECK: mogg.call "mip.bool.xor"
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
  %res1 = mip.bool.xor %lhs, %rhs
  mef.output %ch0, %res1: !mef.chain, i1
}

// CHECK-LABEL: mgp.model @mip_select
// CHECK: mogg.call "mip.select"
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
  %res1 = mip.select %cond, %one, %zero : (i1, index, index) -> index
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1: !mef.chain
}
