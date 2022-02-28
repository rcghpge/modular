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
  %ch0 = "bef.chain.new"() : () -> !bef.chain
  "test.print_hello"(%ch0) : (!bef.chain) -> !bef.chain
  bef.output
}

// CHECK-LABEL: --- Running 'three_print'
bef.func @three_print() {
  %ch0 = "bef.chain.new"() : () -> !bef.chain

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

  %ch0 = "bef.chain.new"() : () -> !bef.chain

  // CHECK: int32 = 42
  %ch1 = "test.print.i32"(%x, %ch0) : (i32, !bef.chain) -> !bef.chain

  %add = "test.add.i32"(%x, %x) : (i32, i32) -> i32

  // CHECK: int32 = 84
  %ch2 = "test.print.i32"(%add, %ch1) : (i32, !bef.chain) -> !bef.chain

  bef.output
}

// CHECK-LABEL: --- Running 'test_result'
bef.func @test_result() -> (i32, i32, !bef.chain) {
  %ch0 = "bef.chain.new"() : () -> !bef.chain
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

// CHECK-LABEL: --- Running 'simple_calls'
bef.func @simple_calls() -> (i32, i32) {
  // CHECK: int32 = 42
  // CHECK: int32 = 84
  "bef.call"() { fn = @basic.i32 } : () -> ()

  %a, %b, %c = "bef.call"() { fn = @errors } : () -> (i32, i32, i32)

  bef.output %a, %b : i32, i32
  // CHECK: 'simple_calls' returned <error: something bad happened>, 42
}

// CHECK-LABEL: --- Can not run 'call_print.i32' because it has arguments
bef.func @call_print.i32(%x: i32) -> !bef.chain {
  %ch0 = "bef.chain.new"() : () -> !bef.chain
  %ch1 = "test.print.i32"(%x, %ch0) : (i32, !bef.chain) -> !bef.chain
  bef.output %ch1 : !bef.chain
}

// CHECK-LABEL: --- Can not run 'add_one' because it has arguments
bef.func @add_one(%x: i32) -> i32 {
  %c1 = "test.constant.i32"() { value = 1 : i32 } : () -> i32
  %y = "test.add.i32"(%x, %c1) : (i32, i32) -> i32
  bef.output %y : i32
}

// CHECK-LABEL: --- Running 'caller'
bef.func @caller() -> !bef.chain {
  %c1 = "test.constant.i32"() { value = 1 : i32 } : () -> i32

  // CHECK-NEXT: int32 = 1
  %ch0 = "bef.call"(%c1) { fn = @call_print.i32 } : (i32) -> (!bef.chain)

  %x = "bef.call"(%c1) {fn = @add_one } : (i32) -> i32

  // CHECK-NEXT: int32 = 2
  %ch1 = "test.print.i32"(%x, %ch0) : (i32, !bef.chain) -> !bef.chain

  bef.output %ch1 : !bef.chain

  // CHECK-NEXT: 'caller' returned chain
}
