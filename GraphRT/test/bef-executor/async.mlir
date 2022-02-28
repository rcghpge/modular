// RUN: bef-translate -export-bef -allow-unregistered-dialect %s > %t.bef
// RUN: bef-executor %t.bef | FileCheck %s

// CHECK: Initializing leak check allocator

// CHECK-LABEL: --- Running 'test_async_print_result'
bef.func @test_async_print_result() -> i32 {
  %c1 = "test.constant.i32"() { value = 1: i32 } : () -> i32
  %c42 = "test.constant.i32"() { value = 42: i32 } : () -> i32
  %a = "test.async.add.i32"(%c1, %c42) : (i32, i32) -> i32
  bef.output %a : i32
}
// CHECK: 'test_async_print_result' returned 43




