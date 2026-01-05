//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Benchmark MLIR for measuring control flow overhead with async operations.
// Derived from MLRT/test/GraphRT/mef-executor/control_flow_*.mlir
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// If/Else Benchmarks
//===----------------------------------------------------------------------===//

// Minimal if/else - measures branch evaluation overhead
mef.callable @bench_if_impl(%cond: i1) -> si32 {
  %res = grt.if %cond () -> (si32) {
    %a = grt.constant.si32 1
    mef.output %a : si32
  } else {
    %b = grt.constant.si32 2
    mef.output %b : si32
  }
  mef.output %res : si32
}

// Testcase: if with true branch
grt.testcase @bench_if_true {
  %true = grt.constant.i1 1
  %r = grt.invoke @bench_if_impl(%true) : (i1) -> si32
  mef.output %r : si32
}

// Testcase: if with false branch
grt.testcase @bench_if_false {
  %false = grt.constant.i1 0
  %r = grt.invoke @bench_if_impl(%false) : (i1) -> si32
  mef.output %r : si32
}

//===----------------------------------------------------------------------===//
// While Loop Benchmarks
//===----------------------------------------------------------------------===//

// Testcase: while loop 1 iteration
grt.testcase @bench_while_1 {
  %i = grt.constant.si32 0
  %n = grt.constant.si32 0  // Will iterate 1 time (0..0)

  %res:2 = grt.while (%i: si32, %n: si32) {
    %cond = grt.lessequal.si32 %i, %n
    mef.output %i, %n, %cond: si32, si32, i1
  } do {
    %one = grt.constant.si32 1
    %new_i = "test.async.add.i32"(%i, %one) : (si32, si32) -> si32
    mef.output %new_i, %n: si32, si32
  }
  mef.output %res#0 : si32
}

// Testcase: while loop 10 iterations
// Iterates while i <= n-1 (i.e., 10 times for n=9)
grt.testcase @bench_while_10 {
  %i = grt.constant.si32 0
  %n = grt.constant.si32 9  // Will iterate 10 times (0..9)

  %res:2 = grt.while (%i: si32, %n: si32) {
    %cond = grt.lessequal.si32 %i, %n
    mef.output %i, %n, %cond: si32, si32, i1
  } do {
    %one = grt.constant.si32 1
    %new_i = "test.async.add.i32"(%i, %one) : (si32, si32) -> si32
    mef.output %new_i, %n: si32, si32
  }
  mef.output %res#0 : si32
}

//===----------------------------------------------------------------------===//
// Repeat Loop Benchmarks
//===----------------------------------------------------------------------===//

// Testcase: repeat 1 time with minimal body
grt.testcase @bench_repeat_1 {
  %count = grt.constant.si32 1
  %ch = grt.chain.create ()
  %ch_out = grt.repeat.i32 %count (%ch: !mef.chain) {
    mef.output %ch : !mef.chain
  }
  mef.output %ch_out : !mef.chain
}

// Testcase: repeat 10 times with minimal body
grt.testcase @bench_repeat_10 {
  %count = grt.constant.si32 10
  %ch = grt.chain.create ()
  %ch_out = grt.repeat.i32 %count (%ch: !mef.chain) {
    mef.output %ch : !mef.chain
  }
  mef.output %ch_out : !mef.chain
}
