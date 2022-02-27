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


// CHECK-LABEL: Can not run 'args_results' because it has arguments
bef.func @args_results(%a: i32, %b: i8) -> (i8, i32) {
  bef.output %b, %a : i8, i32
}

// CHECK-LABEL: Running 'print_test':
bef.func @print_test() {
  // CHECK: hello bef executor!
  "test.print_hello"() : () -> ()
  bef.output
}

// CHECK-LABEL: --- Running 'three_print_error'
bef.func @three_print_error() {

  // CHECK: hello bef executor!
  "test.print_hello"() : () -> ()

  // CHECK: hello bef executor!
  "test.print_hello"() : () -> ()

  // CHECK: hello bef executor!
  "test.print_hello"() : () -> ()

  bef.output
}
