// RUN: bef-translate -export-bef -allow-unregistered-dialect %s > %t.bef
// RUN: bef-executor %t.bef | FileCheck %s

bef.invokable @return_first_arg(%x: si32, %y: si32) -> si32 {
  bef.output %x : si32
}

// Test a nonstrict call with an unused argument.
// CHECK-LABEL: --- Running 'unused_nonstrict_arg'
bef.invokable @unused_nonstrict_arg() {
  %c1 = grt.constant.i32 1

  %v2 = "test.async.add.i32"(%c1, %c1) : (si32, si32) -> si32
  %v3 = "test.async.add.i32"(%v2, %v2) : (si32, si32) -> si32

  // BEFExecutor will allocate an IndirectAsyncValue for %v3, but %v3 is unused.
  %x = grt.invoke @return_first_arg(%c1, %v3) : (si32, si32) -> si32

  // At this point %v3 is unused, but has not yet been forwarded. Setting %v3
  // must count as a use, otherwise the executor would immediately destroy %v3.

  %ch0 = grt.chain.new
  // CHECK-NEXT: int32 = 1
  %ch1 = grt.print.i32 %x, %ch0
  bef.output
}
