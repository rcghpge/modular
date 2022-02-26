// RUN: bef-translate -export-bef -allow-unregistered-dialect %s > %t.bef
// RUN: bef-executor %t.bef | FileCheck %s

// CHECK: Initializing leak check allocator

// CHECK: test1: raw
// CHECK: hello bef-executor 1
// CHECK: bef barf: 42 x

// CHECK: test2: RCRef
// CHECK: hello bef-executor 1
// CHECK: bef barf: 42 x

// CHECK: test3: AsyncValueRef
// CHECK: hello bef-executor 1
// CHECK: bef barf: 42 x

bef.func @args_results(%a: i32, %b: i8) -> (i8, i32) {
  bef.output %b, %a : i8, i32
}


bef.func @run_prim() {
  "test.something"() : () -> ()
  bef.output
}
