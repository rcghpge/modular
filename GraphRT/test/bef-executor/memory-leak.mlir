// Check all the allocators:

// Default allocator will notice the crash.
// RUN: bef-translate -export-bef -allow-unregistered-dialect %s > %t.bef
// RUN: not --crash bef-executor %t.bef 2>&1 | FileCheck %s

// Malloc allocator won't detect a problem
// RUN: bef-executor -allocator=malloc %t.bef

// Profiling allocator works like the default.
// RUN:  not --crash bef-executor -allocator=profiler %t.bef

// CHECK: --- Running 'test_leak_one_int32'
bef.invokable @test_leak_one_int32() -> () {
  "test.memory_leak_one_int32"() : () -> ()
  bef.output
}
// CHECK: Memory leak detected: 1 alive allocations, 4 alive bytes
