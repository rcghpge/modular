// RUN: bef-translate -export-bef -allow-unregistered-dialect %s > %t.bef
// RUN: bef-executor %t.bef | FileCheck %s

// CHECK-LABEL: Can not run 'args_results' because it has arguments
bef.invokable @args_results(%a: i32, %b: i8) -> (i8, i32) {
  bef.output %b, %a : i8, i32
}

// COM: This will show up before every test function because we create a new
// COM: Runtime instance for each Function in the test.
// CHECK: Initializing leak check allocator

// CHECK-LABEL: Running 'print_test':
bef.invokable @print_test() {
  // CHECK: hello bef executor!
  %ch0 = grt.chain.new
  "test.print_hello"(%ch0) : (!grt.chain) -> !grt.chain
  bef.output
}

// CHECK-LABEL: --- Running 'three_print'
bef.invokable @three_print() {
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
bef.invokable @basic.i32() {
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
bef.invokable @test_result() -> (i32, i32, !grt.chain) {
  %ch0 = grt.chain.new
  %x = grt.constant.i32 42
  %y = grt.constant.i32 13
  %add = grt.add.i32 %x, %y
  // CHECK: 'test_result' returned 42, 55
  bef.output %x, %add, %ch0 : i32, i32, !grt.chain
}

// CHECK-LABEL: --- Running 'errors'
bef.invokable @errors() -> (i32, i32, i32) {
  %a = "test.fail"() : () -> i32
  %b, %c = "test.partial_fail"() : () -> (i32, i32)

  bef.output %a, %b, %c : i32, i32, i32
  // CHECK: 'errors' returned <error: something bad happened>,
  // CHECK: 42, <error: something partially bad happened>
}

// CHECK-LABEL: --- Running 'simple_calls'
bef.invokable @simple_calls() -> (i32, i32) {
  // CHECK: int32 = 42
  // CHECK: int32 = 84
  grt.invoke @basic.i32() : () -> ()

  %a, %b, %c = grt.invoke @errors() : () -> (i32, i32, i32)

  bef.output %a, %b : i32, i32
  // CHECK: 'simple_calls' returned <error: something bad happened>, 42
}

// CHECK-LABEL: --- Can not run 'call_print.i32' because it has arguments
bef.invokable @call_print.i32(%x: i32) -> !grt.chain {
  %ch0 = grt.chain.new
  %ch1 = grt.print.i32 %x, %ch0
  bef.output %ch1 : !grt.chain
}

// CHECK-LABEL: --- Can not run 'add_one' because it has arguments
bef.invokable @add_one(%x: i32) -> i32 {
  %c1 = grt.constant.i32 1
  %y = grt.add.i32 %x, %c1
  bef.output %y : i32
}

// CHECK-LABEL: --- Running 'caller'
bef.invokable @caller() -> !grt.chain {
  %c1 = grt.constant.i32 1

  // CHECK-NEXT: int32 = 1
  %ch0 = grt.invoke @call_print.i32(%c1) : (i32) -> (!grt.chain)

  %x = grt.invoke @add_one(%c1) : (i32) -> i32

  // CHECK-NEXT: int32 = 2
  %ch1 = grt.print.i32 %x, %ch0

  bef.output %ch1 : !grt.chain

  // CHECK-NEXT: 'caller' returned chain
}

// CHECK-LABEL: --- Running 'basic_string_and_arrays'
bef.invokable @basic_string_and_arrays() {
  %x = "test.get_string"() { value = "bark" } : () -> !test.string
  %y = "test.get_string"() { value = "rarf" } : () -> !test.string
  %ch0 = grt.chain.new

  // CHECK: string = bark
  %ch1 = "test.print_string"(%x, %ch0) : (!test.string, !grt.chain) -> (!grt.chain)

  %z = "test.append_string"(%x, %y) : (!test.string, !test.string) -> !test.string

  // CHECK: string = barkrarf
  %ch2 = "test.print_string"(%z, %ch1) : (!test.string, !grt.chain) -> (!grt.chain)

  // CHECK: array = [4, 1, 5712321]
  %ch3 = "test.print_array_i32"(%ch2) { value = [4: i32, 1: i32, 5712321:i32] } : (!grt.chain) -> (!grt.chain)

  // CHECK: dictionary[theNumber3] = 3
  %ch4 = "test.print_dict_i32"(%ch3) { key = "theNumber3", value = { theNumber3 = 3 : i32, foo = "bar" } } : (!grt.chain) -> (!grt.chain)

  // CHECK: dictionary[theNumber4] = NOTFOUND
  %ch5 = "test.print_dict_i32"(%ch4) { key = "theNumber4", value = { theNumber3 = 3 : i32, foo = "bar" } } : (!grt.chain) -> (!grt.chain)

  // CHECK: dictionary = aString : stringAttr, theNumber3 : 3, theNumber5 : 5
  %ch6 = "test.print_all_dict"(%ch5) { value = { theNumber3 = 3 : i32, theNumber5 = 5 : i32, aString = "stringAttr" } } : (!grt.chain) -> (!grt.chain)

  bef.output
}

// CHECK-LABEL: --- Running 'test_dynamic_attr'
bef.invokable @test_dynamic_attr() {
  %ch0 = grt.chain.new

  // CHECK: int32 = 4
  %ch1 = "test.print.dynamic.attr"(%ch0) {value = 4: i32} : (!grt.chain) -> !grt.chain
  // CHECK: string = foo string
  %ch2 = "test.print.dynamic.attr"(%ch1) {value = "foo string"} : (!grt.chain) -> !grt.chain
  // CHECK: array = [4, 6, 17]
  %ch3 = "test.print.dynamic.attr"(%ch2) {value = [4: i32, 6: i32, 17: i32]} : (!grt.chain) -> !grt.chain
  // CHECK: array = []
  %ch4 = "test.print.dynamic.attr"(%ch3) {value = []} : (!grt.chain) -> !grt.chain
  // CHECK: unsupported attribute kind
  %ch5 = "test.print.dynamic.attr"(%ch4) {value = 1: i1} : (!grt.chain) -> !grt.chain
  // CHECK: dictionary = aString : this is a string, anInt : 3125
  %ch6 = "test.print.dynamic.attr"(%ch5) {value = {aString = "this is a string", anInt = 3125 : i32}} : (!grt.chain) -> !grt.chain

  bef.output
}

