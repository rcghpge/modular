// RUN: bef-translate -export-bef -allow-unregistered-dialect %s > %t.bef
// RUN: not --crash bef-executor %t.bef 2>&1 | FileCheck %s

// CHECK: --- Running 'test_leak_one_int32'
bef.func @test_leak_one_int32() -> () {
  "test.memory_leak_one_int32"() : () -> ()
  bef.output
}
// CHECK: Memory leak detected: 1 alive allocations, 4 alive bytes
