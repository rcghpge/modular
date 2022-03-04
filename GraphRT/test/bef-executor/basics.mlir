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

// CHECK: test4: IndirectAsyncValueRef
// CHECK: av1 = 1
// CHECK: av2 = 1234
// CHECK: av3 = 42
// CHECK: av4 = 42

// CHECK-LABEL: Can not run 'args_results' because it has arguments
bef.func @args_results(%a: i32, %b: i8) -> (i8, i32) {
  bef.output %b, %a : i8, i32
}

// CHECK-LABEL: Running 'print_test':
bef.func @print_test() {
  // CHECK: hello bef executor!
  %ch0 = grt.chain.new
  "test.print_hello"(%ch0) : (!grt.chain) -> !grt.chain
  bef.output
}

// CHECK-LABEL: --- Running 'three_print'
bef.func @three_print() {
  %ch0 = grt.chain.new

  // CHECK: hello bef executor!
  %ch1 = "test.print_hello"(%ch0) : (!grt.chain) -> !grt.chain

  // CHECK: hello bef executor!
  %ch2 = "test.print_hello"(%ch1) : (!grt.chain) -> !grt.chain

  // CHECK: hello bef executor!
  %ch3 = "test.print_hello"(%ch2) : (!grt.chain) -> !grt.chain

  bef.output
}

// CHECK-LABEL: --- Running 'basic.i32'
bef.func @basic.i32() {
  %x = grt.constant.i32 42

  %ch0 = grt.chain.new

  // CHECK: int32 = 42
  %ch1 = grt.print.i32 %x, %ch0

  %add = grt.add.i32 %x, %x

  // CHECK: int32 = 84
  %ch2 = grt.print.i32 %add, %ch1

  bef.output
}

// CHECK-LABEL: --- Running 'test_result'
bef.func @test_result() -> (i32, i32, !grt.chain) {
  %ch0 = grt.chain.new
  %x = grt.constant.i32 42
  %y = grt.constant.i32 13
  %add = grt.add.i32 %x, %y
  // CHECK: 'test_result' returned 42, 55
  bef.output %x, %add, %ch0 : i32, i32, !grt.chain
}

// CHECK-LABEL: --- Running 'errors'
bef.func @errors() -> (i32, i32, i32) {
  %a = "test.fail"() : () -> i32
  %b, %c = "test.partial_fail"() : () -> (i32, i32)

  bef.output %a, %b, %c : i32, i32, i32
  // CHECK: 'errors' returned <error: something bad happened>,
  // CHECK: 42, <error: something partially bad happened>
}

// CHECK-LABEL: --- Running 'simple_calls'
bef.func @simple_calls() -> (i32, i32) {
  // CHECK: int32 = 42
  // CHECK: int32 = 84
  grt.call @basic.i32() : () -> ()

  %a, %b, %c = grt.call @errors() : () -> (i32, i32, i32)

  bef.output %a, %b : i32, i32
  // CHECK: 'simple_calls' returned <error: something bad happened>, 42
}

// CHECK-LABEL: --- Can not run 'call_print.i32' because it has arguments
bef.func @call_print.i32(%x: i32) -> !grt.chain {
  %ch0 = grt.chain.new
  %ch1 = grt.print.i32 %x, %ch0
  bef.output %ch1 : !grt.chain
}

// CHECK-LABEL: --- Can not run 'add_one' because it has arguments
bef.func @add_one(%x: i32) -> i32 {
  %c1 = grt.constant.i32 1
  %y = grt.add.i32 %x, %c1
  bef.output %y : i32
}

// CHECK-LABEL: --- Running 'caller'
bef.func @caller() -> !grt.chain {
  %c1 = grt.constant.i32 1

  // CHECK-NEXT: int32 = 1
  %ch0 = grt.call @call_print.i32(%c1) : (i32) -> (!grt.chain)

  %x = grt.call @add_one(%c1) : (i32) -> i32

  // CHECK-NEXT: int32 = 2
  %ch1 = grt.print.i32 %x, %ch0

  bef.output %ch1 : !grt.chain

  // CHECK-NEXT: 'caller' returned chain
}
