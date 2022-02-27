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
  %ch0 = "test.chain.new"() : () -> !test.chain
  "test.print_hello"(%ch0) : (!test.chain) -> !test.chain
  bef.output
}

// CHECK-LABEL: --- Running 'three_print_error'
bef.func @three_print_error() {
    %ch0 = "test.chain.new"() : () -> !test.chain

  // CHECK: hello bef executor!
  %ch1 = "test.print_hello"(%ch0) : (!test.chain) -> !test.chain

  // CHECK: hello bef executor!
  %ch2 = "test.print_hello"(%ch1) : (!test.chain) -> !test.chain

  // CHECK: hello bef executor!
  %ch3 = "test.print_hello"(%ch2) : (!test.chain) -> !test.chain

  bef.output
}

// CHECK-LABEL: --- Running 'basic.i32'
bef.func @basic.i32() {
  %x = "test.constant42.i32"() : () -> i32

  %ch0 = "test.chain.new"() : () -> !test.chain

  // CHECK: int32 = 42
  %ch1 = "test.print.i32"(%x, %ch0) : (i32, !test.chain) -> !test.chain

  %add = "test.add.i32"(%x, %x) : (i32, i32) -> i32

  // CHECK: int32 = 84
  %ch2 = "test.print.i32"(%add, %ch1) : (i32, !test.chain) -> !test.chain

  bef.output
}

// CHECK-LABEL: --- Running 'test_result'
bef.func @test_result() -> (i32, i32) {
  %x = "test.constant42.i32"() : () -> i32
  %add = "test.add.i32"(%x, %x) : (i32, i32) -> i32
  bef.output %x, %add : i32, i32
}
// CHECK: 'test_result' returned 42, 84

