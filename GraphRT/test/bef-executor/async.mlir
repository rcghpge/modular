// RUN: bef-translate -export-bef -allow-unregistered-dialect %s > %t.bef
// RUN: bef-executor %t.bef | FileCheck %s

// CHECK: Initializing leak check allocator

// CHECK-LABEL: --- Running 'test_async_print_result'
bef.invokable @test_async_print_result() -> i32 {
  %c1 = grt.constant.i32 1
  %c42 = grt.constant.i32 42
  %a = "test.async.add.i32"(%c1, %c42) : (i32, i32) -> i32
  bef.output %a : i32
}
// CHECK: 'test_async_print_result' returned 43

// CHECK-LABEL: --- Running 'test_cancel'
bef.invokable @test_cancel() -> i32 {
  %c1 = grt.constant.i32 1
  %c42 = grt.constant.i32 42
  %a = "test.async.add.i32"(%c1, %c42) : (i32, i32) -> i32

  %ch0 = grt.chain.new
  %ch1 = "test.cancel"(%ch0) : (!grt.chain) -> (!grt.chain)

  %b = "test.async.add.i32"(%c1, %a) : (i32, i32) -> i32

  bef.output %b : i32
}
// CHECK-NEXT: returned <{{.*}}async.mlir:22:10: error: Cancelled!>

// CHECK-LABEL: --- Running 'test_add_any'
bef.invokable @test_add_any() -> (i32, !test.string, i32) {
  %c1 = grt.constant.i32 1
  %c42 = grt.constant.i32 42
  %a = "test.async.add.any"(%c1, %c42) : (i32, i32) -> i32

  %x = "test.get_string"() { value = "bark" } : () -> !test.string
  %y = "test.get_string"() { value = "rarf" } : () -> !test.string
  %b = "test.async.add.any"(%x, %y) : (!test.string, !test.string) -> !test.string

  %c = "test.async.add.any"(%c1, %y) : (i32, !test.string) -> i32

  bef.output %a, %b, %c : i32, !test.string, i32
}
// CHECK: 'test_add_any' returned 43, barkrarf, <{{.*}}async.mlir:40:8: error: invalid operands to async_add_any> 



