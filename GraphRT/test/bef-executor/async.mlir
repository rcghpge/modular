// RUN: bef-translate -export-bef -allow-unregistered-dialect %s > %t.bef
// RUN: bef-executor %t.bef | FileCheck %s

// CHECK: Initializing leak check allocator

// CHECK-LABEL: --- Running 'test_async_print_result'
bef.func @test_async_print_result() -> i32 {
  %c1 = graphrt.constant.i32 1
  %c42 = graphrt.constant.i32 42
  %a = "test.async.add.i32"(%c1, %c42) : (i32, i32) -> i32
  bef.output %a : i32
}
// CHECK: 'test_async_print_result' returned 43




