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
  %ch0 = "test.chain.new"() : () -> !bef.chain
  "test.print_hello"(%ch0) : (!bef.chain) -> !bef.chain
  bef.output
}

// CHECK-LABEL: --- Running 'three_print'
bef.func @three_print() {
  %ch0 = "test.chain.new"() : () -> !bef.chain

  // CHECK: hello bef executor!
  %ch1 = "test.print_hello"(%ch0) : (!bef.chain) -> !bef.chain

  // CHECK: hello bef executor!
  %ch2 = "test.print_hello"(%ch1) : (!bef.chain) -> !bef.chain

  // CHECK: hello bef executor!
  %ch3 = "test.print_hello"(%ch2) : (!bef.chain) -> !bef.chain

  bef.output
}

// CHECK-LABEL: --- Running 'basic.i32'
bef.func @basic.i32() {
  %x = "test.constant.i32"() { value = 42 : i32} : () -> i32

  %ch0 = "test.chain.new"() : () -> !bef.chain

  // CHECK: int32 = 42
  %ch1 = "test.print.i32"(%x, %ch0) : (i32, !bef.chain) -> !bef.chain

  %add = "test.add.i32"(%x, %x) : (i32, i32) -> i32

  // CHECK: int32 = 84
  %ch2 = "test.print.i32"(%add, %ch1) : (i32, !bef.chain) -> !bef.chain

  bef.output
}

// CHECK-LABEL: --- Running 'test_result'
bef.func @test_result() -> (i32, i32, !bef.chain) {
  %ch0 = "test.chain.new"() : () -> !bef.chain
  %x = "test.constant.i32"() { value = 42 : i32 } : () -> i32
  %y = "test.constant.i32"() { value = 13 : i32 } : () -> i32
  %add = "test.add.i32"(%x, %y) : (i32, i32) -> i32
  // CHECK: 'test_result' returned 42, 55
  bef.output %x, %add, %ch0 : i32, i32, !bef.chain
}


// CHECK-LABEL: --- Running 'errors'
bef.func @errors() -> (i32, i32, i32) {
  %a = "test.fail"() : () -> i32
  %b, %c = "test.partial_fail"() : () -> (i32, i32)

  bef.output %a, %b, %c : i32, i32, i32
  // CHECK: 'errors' returned <error: something bad happened>,
  // CHECK: 42, <error: something partially bad happened>
}


