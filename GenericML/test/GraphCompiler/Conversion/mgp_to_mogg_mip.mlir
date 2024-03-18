// RUN: mo-opt %s --lower-primitives | FileCheck %s

//--- Test lowering of MIP Index Primitives

//--- mip.constant.index/mip.print.index will be tested throughtout the tests ---/


// CHECK-LABEL: mgp.model @mip_add
// CHECK: mogg.call "mip.add"
mgp.model @mip_add : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = mip.constant.index 42
  %rhs = mip.constant.index -20
  %res1 = mip.add %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_div
// CHECK: mogg.call "mip.div"
mgp.model @mip_div : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = mip.constant.index 42
  %rhs = mip.constant.index -2
  %res1 = mip.div %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_div_ceil
// CHECK: mogg.call "mip.div.ceil"
mgp.model @mip_div_ceil : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = mip.constant.index 42
  %rhs = mip.constant.index -4
  %res1 = mip.div.ceil %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_cmp_eq
// CHECK: mogg.call "mip.cmp.eq"
mgp.model @mip_cmp_eq : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
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
mgp.model @mip_cmp_lt : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
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
mgp.model @mip_cmp_le : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
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
mgp.model @mip_max : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = mip.constant.index 42
  %rhs = mip.constant.index -2
  %res1 = mip.max %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_min
// CHECK: mogg.call "mip.min"
mgp.model @mip_min : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = mip.constant.index 42
  %rhs = mip.constant.index -2
  %res1 = mip.min %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_mod
// CHECK: mogg.call "mip.mod"
mgp.model @mip_mod : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = mip.constant.index 42
  %rhs = mip.constant.index 4
  %res1 = mip.mod %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_mul
// CHECK: mogg.call "mip.mul"
mgp.model @mip_mul : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = mip.constant.index 42
  %rhs = mip.constant.index 4
  %res1 = mip.mul %lhs, %rhs
  %ch1 = mip.print.index %res1, %ch0
  mef.output %ch1 : !mef.chain
}


