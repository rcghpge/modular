// RUN: mo-opt %s --lower-primitives | FileCheck %s

//--- Test lowering of MIP Index Primitives

//--- mip.constant.index/mip.print.index will be tested throughtout the tests ---/

// CHECK-LABEL: mgp.model @mip_basic
// CHECK: mogg.call "mip.constant.index"
// CHECK: mogg.call "mip.print.index"
mgp.model @mip_basic : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = mip.constant.index 42
  %ch1 = mip.print.index %lhs, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_add
// CHECK: mogg.call "mip.add"
mgp.model @mip_add : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = mip.constant.index 42
  %rhs = mip.constant.index -20
  %res1 = mip.add %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_div
// CHECK: mogg.call "mip.div"
mgp.model @mip_div : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = mip.constant.index 42
  %rhs = mip.constant.index -2
  %res1 = mip.div %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_div_ceil
// CHECK: mogg.call "mip.div.ceil"
mgp.model @mip_div_ceil : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = mip.constant.index 42
  %rhs = mip.constant.index -4
  %res1 = mip.div.ceil %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_cmp_eq
// CHECK: mogg.call "mip.cmp.eq"
mgp.model @mip_cmp_eq : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = mip.constant.index 4
  %rhs = mip.constant.index 4
  %res1 = mip.cmp.eq %lhs, %rhs
  %lhs1 = mip.constant.index 2
  %res2 = mip.cmp.eq %lhs1, %rhs
  mef.output %ch0 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_cmp_lt
// CHECK: mogg.call "mip.cmp.lt"
mgp.model @mip_cmp_lt : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = mip.constant.index 4
  %rhs = mip.constant.index 4
  %res1 = mip.cmp.lt %lhs, %rhs
  %lhs1 = mip.constant.index 2
  %res2 = mip.cmp.lt %lhs1, %rhs
  mef.output %ch0 : !mef.chain
}


// CHECK-LABEL: mgp.model @mip_cmp_le
// CHECK: mogg.call "mip.cmp.le"
mgp.model @mip_cmp_le : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = mip.constant.index 5
  %rhs = mip.constant.index 4
  %res1 = mip.cmp.le %lhs, %rhs
  %lhs1 = mip.constant.index 4
  %res2 = mip.cmp.le %lhs1, %rhs
  mef.output %ch0 : !mef.chain
}


// CHECK-LABEL: mgp.model @mip_max
// CHECK: mogg.call "mip.max"
mgp.model @mip_max : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = mip.constant.index 42
  %rhs = mip.constant.index -2
  %res1 = mip.max %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_min
// CHECK: mogg.call "mip.min"
mgp.model @mip_min : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = mip.constant.index 42
  %rhs = mip.constant.index -2
  %res1 = mip.min %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_mod
// CHECK: mogg.call "mip.mod"
mgp.model @mip_mod : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = mip.constant.index 42
  %rhs = mip.constant.index 4
  %res1 = mip.mod %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_mul
// CHECK: mogg.call "mip.mul"
mgp.model @mip_mul : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = mip.constant.index 42
  %rhs = mip.constant.index 4
  %res1 = mip.mul %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_nary_mul
// CHECK: mogg.call "mip.nary.mul"
mgp.model @mip_nary_mul : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %a = mip.constant.index 3
  %b = mip.constant.index -5
  %c = mip.constant.index 7
  %res1 = mip.nary.mul (%a, %b, %c)
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_and
// CHECK: mogg.call "mip.bool.and"
mgp.model @mip_and : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, i1) {
  %ch0 = grt.chain.create ()
  %zero = mip.constant.index 0
  %one = mip.constant.index 1
  %lhs = mip.constant.bool true
  %rhs = mip.constant.bool true
  %res1 = mip.bool.and %lhs, %rhs
  mef.output %ch0, %res1: !mef.chain, i1
}

// CHECK-LABEL: mgp.model @mip_or
// CHECK: mogg.call "mip.bool.or"
mgp.model @mip_or : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, i1) {
  %ch0 = grt.chain.create ()
  %zero = mip.constant.index 0
  %one = mip.constant.index 1
  %lhs = mip.constant.bool true
  %rhs = mip.constant.bool false
  %res1 = mip.bool.or %lhs, %rhs
  mef.output %ch0, %res1: !mef.chain, i1
}

// CHECK-LABEL: mgp.model @mip_xor
// CHECK: mogg.call "mip.bool.xor"
mgp.model @mip_xor : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, i1) {
  %ch0 = grt.chain.create ()
  %zero = mip.constant.index 0
  %one = mip.constant.index 1
  %lhs = mip.constant.bool true
  %rhs = mip.constant.bool false
  %res1 = mip.bool.xor %lhs, %rhs
  mef.output %ch0, %res1: !mef.chain, i1
}

// CHECK-LABEL: mgp.model @mip_select
// CHECK: mogg.call "mip.select"
mgp.model @mip_select : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain) {
  %ch0 = grt.chain.create ()
  %zero = mip.constant.index 0
  %one = mip.constant.index 1
  %cond = mip.constant.bool true
  %res1 = mip.select %cond, %one, %zero : (i1, index, index) -> index
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1: !mef.chain
}
