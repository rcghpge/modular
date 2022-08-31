// RUN: bef-translate -export-bef -allow-unregistered-dialect %s -o %t.bef
// RUN: bef-executor %t.bef -verify-diagnostics | FileCheck %s

bef.invokable @args_results(%a: si32, %b: ui8) -> (ui8, si32) {
  bef.output %b, %a : ui8, si32
}

// CHECK-LABEL: Running 'print_test':
grt.testcase @print_test {
  // CHECK: hello bef executor!
  %ch0 = grt.chain.new
  "test.print_hello"(%ch0) : (!bef.chain) -> !bef.chain
  bef.output
}

// CHECK-LABEL: --- Running 'three_print'
grt.testcase @three_print {
  %ch0 = grt.chain.new

  // CHECK: hello bef executor!
  %ch1 = "test.print_hello"(%ch0) : (!bef.chain) -> !bef.chain

  // CHECK: hello bef executor!
  %ch2 = "test.print_hello"(%ch1) : (!bef.chain) -> !bef.chain

  // CHECK: hello bef executor!
  %ch3 = "test.print_hello"(%ch2) : (!bef.chain) -> !bef.chain

  bef.output
}

// CHECK-LABEL: --- Running 'basic.i32_driver'
bef.invokable @basic.i32() {
  %x = grt.constant.si32 42

  %ch0 = grt.chain.new

  // CHECK: int32 = 42
  %ch1 = grt.print.si32 %x, %ch0

  %add = grt.add.si32 %x, %x

  // CHECK: int32 = 84
  %ch2 = grt.print.si32 %add, %ch1

  bef.output
}

grt.testcase @basic.i32_driver {
  grt.invoke @basic.i32() : () -> ()
  bef.output
}

// CHECK-LABEL: --- Running 'test_result'
grt.testcase @test_result {
  %ch0 = grt.chain.new
  %x = grt.constant.si32 42
  %y = grt.constant.si32 13
  %add = grt.add.si32 %x, %y
  // CHECK: 'test_result' returned 42, 55
  bef.output %x, %add, %ch0 : si32, si32, !bef.chain
}

// CHECK-LABEL: --- Running 'errors_driver'
bef.invokable @errors() -> (si32, si32, si32) {
  // expected-error @+1 {{something bad happened}}
  %a = "test.fail"() : () -> si32
  // expected-error @+1 {{something partially bad happened}}
  %b, %c = "test.partial_fail"() : () -> (si32, si32)

  bef.output %a, %b, %c : si32, si32, si32
  // CHECK: 'errors_driver' returned <error: something bad happened>,
  // CHECK: 42, <error: something partially bad happened>
}

grt.testcase @errors_driver {
  %a, %b, %c = grt.invoke @errors() : () -> (si32, si32, si32)
  bef.output %a, %b, %c : si32, si32, si32
}

// CHECK-LABEL: --- Running 'simple_calls'
grt.testcase @simple_calls {
  // CHECK: int32 = 42
  // CHECK: int32 = 84
  grt.invoke @basic.i32() : () -> ()

  %a, %b, %c = grt.invoke @errors() : () -> (si32, si32, si32)

  bef.output %a, %b : si32, si32
  // CHECK: 'simple_calls' returned <error: something bad happened>, 42
}

bef.invokable @call_print.i32(%x: si32) -> !bef.chain {
  %ch0 = grt.chain.new
  %ch1 = grt.print.si32 %x, %ch0
  bef.output %ch1 : !bef.chain
}

bef.invokable @add_one(%x: si32) -> si32 {
  %c1 = grt.constant.si32 1
  %y = grt.add.si32 %x, %c1
  bef.output %y : si32
}

// CHECK-LABEL: --- Running 'caller'
grt.testcase @caller {
  %c1 = grt.constant.si32 1

  // CHECK-NEXT: int32 = 1
  %ch0 = grt.invoke @call_print.i32(%c1) : (si32) -> (!bef.chain)

  %x = grt.invoke @add_one(%c1) : (si32) -> si32

  // CHECK-NEXT: int32 = 2
  %ch1 = grt.print.si32 %x, %ch0

  bef.output %ch1 : !bef.chain

  // CHECK-NEXT: 'caller' returned chain
}

// CHECK-LABEL: --- Running 'basic_string_and_arrays'
grt.testcase @basic_string_and_arrays {
  %x = "test.get_string"() { value = "bark" } : () -> !test.string
  %y = "test.get_string"() { value = "rarf" } : () -> !test.string
  %ch0 = grt.chain.new

  // CHECK: string = bark
  %ch1 = "test.print_string"(%x, %ch0) : (!test.string, !bef.chain) -> (!bef.chain)

  %z = "test.append_string"(%x, %y) : (!test.string, !test.string) -> !test.string

  // CHECK: string = barkrarf
  %ch2 = "test.print_string"(%z, %ch1) : (!test.string, !bef.chain) -> (!bef.chain)

  // CHECK: array = [4, 1, 5712321]
  %ch3 = "test.print_array_i32"(%ch2) { value = array<si32: 4, 1, 5712321> } : (!bef.chain) -> (!bef.chain)

  // CHECK: dictionary[theNumber3] = 3
  %ch4 = "test.print_dict_i32"(%ch3) { key = "theNumber3", value = { theNumber3 = 3 : si32, foo = "bar" } } : (!bef.chain) -> (!bef.chain)

  // CHECK: dictionary[theNumber4] = NOTFOUND
  %ch5 = "test.print_dict_i32"(%ch4) { key = "theNumber4", value = { theNumber3 = 3 : si32, foo = "bar" } } : (!bef.chain) -> (!bef.chain)

  // CHECK: dictionary = aString : stringAttr, theNumber3 : 3, theNumber5 : 5
  %ch6 = "test.print_all_dict"(%ch5) { value = { theNumber3 = 3 : si32, theNumber5 = 5 : si32, aString = "stringAttr" } } : (!bef.chain) -> (!bef.chain)

  bef.output
}

grt.testcase @decode_error {
  %ch0 = grt.chain.new
  // COM: Because we can't do regex in expected-error we have to end the
  // COM: llvm::SmallVector type early here to account for the sizing.
  // expected-error @+1 {{type mismatch, have si32 but caller expected llvm::SmallVector<float}}
  %ch1 = "test.print_dynamic_array_float"(%ch0) { value = 1 : si32 } : (!bef.chain) -> (!bef.chain)
  bef.output %ch1 : !bef.chain
}

// CHECK-LABEL: --- Running 'test_types'
grt.testcase @test_types {
  %ch0 = grt.chain.new

  // CHECK-NEXT: Type attribute: bool
  %ch1 = "test.print_type"(%ch0) {t = i1} : (!bef.chain) -> (!bef.chain)
  // CHECK-NEXT: Type attribute: si1
  %ch2 = "test.print_type"(%ch1) {t = si1} : (!bef.chain) -> (!bef.chain)
  // CHECK-NEXT: Type attribute: ui1
  %ch3 = "test.print_type"(%ch2) {t = ui1} : (!bef.chain) -> (!bef.chain)
  // CHECK-NEXT: Type attribute: f32
  %ch4 = "test.print_type"(%ch3) {t = f32} : (!bef.chain) -> (!bef.chain)
  // CHECK-NEXT: Type attribute: f64
  %ch5 = "test.print_type"(%ch4) {t = f64} : (!bef.chain) -> (!bef.chain)

  // CHECK-NEXT: Type attribute: ui8
  %ch6 = "test.print_type"(%ch5) {t = ui8} : (!bef.chain) -> (!bef.chain)
  // CHECK-NEXT: Type attribute: si8
  %ch7 = "test.print_type"(%ch6) {t = si8} : (!bef.chain) -> (!bef.chain)
  // CHECK-NEXT: Type attribute: ui16
  %ch8 = "test.print_type"(%ch7) {t = ui16} : (!bef.chain) -> (!bef.chain)
  // CHECK-NEXT: Type attribute: si16
  %ch9 = "test.print_type"(%ch8) {t = si16} : (!bef.chain) -> (!bef.chain)
  // CHECK-NEXT: Type attribute: ui32
  %ch10 = "test.print_type"(%ch9) {t = ui32} : (!bef.chain) -> (!bef.chain)
  // CHECK-NEXT: Type attribute: si32
  %ch11 = "test.print_type"(%ch10) {t = si32} : (!bef.chain) -> (!bef.chain)
  // CHECK-NEXT: Type attribute: ui64
  %ch12 = "test.print_type"(%ch11) {t = ui64} : (!bef.chain) -> (!bef.chain)
  // CHECK-NEXT: Type attribute: si64
  %ch13 = "test.print_type"(%ch12) {t = si64} : (!bef.chain) -> (!bef.chain)

  // CHECK-NEXT: Type attribute: (si64, ui8) -> bool
  %ch14 = "test.print_type"(%ch13) {t = (si64, ui8) -> i1} : (!bef.chain) -> (!bef.chain)

  // Nested function type.
  // CHECK-NEXT: Type attribute: (si64, (ui8) -> (f32, f64)) -> bool
  %ch15 = "test.print_type"(%ch14) {t = (si64, (ui8)->(f32, f64)) -> i1} : (!bef.chain) -> (!bef.chain)

  // CHECK-NEXT: Type attribute: tensor<4x0x123xf32>
  %ch16 = "test.print_type"(%ch15) {t = tensor<4x0x123xf32>} : (!bef.chain) -> (!bef.chain)

  // CHECK-NEXT: Type attribute: quant<si8:f32, 0.100000:-128>
  %ch17 = "test.print_type"(%ch16) {t = !quant.uniform<i8:f32, 0.1:-128>} : (!bef.chain) -> (!bef.chain)

  // CHECK-NEXT: Type attribute: quantPerAxis<si16:f32:1, {0.200000:128, 0.100000:-128}>
  %ch18 = "test.print_type"(%ch17) {t = !quant.uniform<i16:f32:1, {0.2:128, 0.1:-128}>} : (!bef.chain) -> (!bef.chain)

 // CHECK-NEXT: Type attribute: si2
  %ch19 = "test.print_type"(%ch18) {t = si2} : (!bef.chain) -> (!bef.chain)

  // CHECK-NEXT: Type attribute: ui128
  %ch20 = "test.print_type"(%ch19) {t = ui128} : (!bef.chain) -> (!bef.chain)

  // CHECK-NEXT: Type attribute: complex<ui128>
  %ch21 = "test.print_type"(%ch20) {t = complex<ui128>} : (!bef.chain) -> (!bef.chain)
  bef.output
}

// CHECK-LABEL: --- Running 'test_dynamic_attr'
grt.testcase @test_dynamic_attr {
  %ch0 = grt.chain.new

  // CHECK: int32 = 4
  %ch1 = "test.print.dynamic.attr"(%ch0) {value = 4: si32} : (!bef.chain) -> !bef.chain
  // CHECK: string = foo string
  %ch2 = "test.print.dynamic.attr"(%ch1) {value = "foo string"} : (!bef.chain) -> !bef.chain
  // CHECK: array = [4, 6, 17]
  %ch3 = "test.print.dynamic.attr"(%ch2) {value = array<si32: 4, 6, 17>} : (!bef.chain) -> !bef.chain
  // CHECK: array = []
  %ch4 = "test.print.dynamic.attr"(%ch3) {value = array<si32>} : (!bef.chain) -> !bef.chain
  // CHECK: unsupported attribute kind
  %ch5 = "test.print.dynamic.attr"(%ch4) {value = 1: i1} : (!bef.chain) -> !bef.chain
  // CHECK: dictionary {
  // CHECK-NEXT: 'aString': string = this is a string
  // CHECK-NEXT: 'anInt': sint32 = 3125
  // CHECK-NEXT: }
  %ch6 = "test.print.dynamic.attr"(%ch5) {value =
     {aString = "this is a string", anInt = 3125 : si32}
  } : (!bef.chain) -> !bef.chain

  // CHECK:     list [
  // CHECK-NEXT:  string = this is a string
  // CHECK-NEXT:  sint32 = 3125
  // CHECK-NEXT:  list [
  // CHECK-NEXT:  ]
  // CHECK-NEXT:  dictionary {
  // CHECK-NEXT:    'a': uint32 = 42
  // CHECK-NEXT:    'b': string = foo
  // CHECK-NEXT:   }
  // CHECK-NEXT: ]
  %ch7 = "test.print.dynamic.attr"(%ch6) {value =
     ["this is a string", 3125 : si32, [], {a = 42: ui32, b = "foo"}]
  } : (!bef.chain) -> !bef.chain

  // CHECK-NEXT: Type attribute: f32
  %ch8 = "test.print.dynamic.attr"(%ch7) {t = f32} : (!bef.chain) -> !bef.chain

  bef.output
}
