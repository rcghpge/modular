// RUN: bef-translate -export-bef -allow-unregistered-dialect %s > %t.bef
// RUN: bef-executor %t.bef | FileCheck %s

// CHECK: Initializing leak check allocator

bef.func @return_first_arg(%x: i32, %y: i32) -> i32 {
  bef.output %x : i32
}

// Test a nonstrict call with an unused argument.
// CHECK-LABEL: --- Running 'unused_nonstrict_arg'
bef.func @unused_nonstrict_arg() {
  %c1 = graphrt.constant.i32 1

  %v2 = "test.async.add.i32"(%c1, %c1) : (i32, i32) -> i32
  %v3 = "test.async.add.i32"(%v2, %v2) : (i32, i32) -> i32

  // BEFExecutor will allocate an IndirectAsyncValue for %v3, but %v3 is unused.
  %x = "test.call"(%c1, %v3) { fn = @return_first_arg, bef.nonstrict } : (i32, i32) -> i32

  // At this point %v3 is unused, but has not yet been forwarded. Setting %v3
  // must count as a use, otherwise the executor would immediately destroy %v3.

  %ch0 = graphrt.chain.new
  // CHECK-NEXT: int32 = 1
  %ch1 = graphrt.print.i32 %x, %ch0
  bef.output
}
