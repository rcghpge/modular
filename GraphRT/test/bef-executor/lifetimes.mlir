// RUN: bef-translate -export-bef -allow-unregistered-dialect %s > %t.bef
// RUN: bef-executor %t.bef | FileCheck %s

// CHECK-LABEL: --- Running 'test_linear'
bef.func @test_linear() {
  %ch0 = grt.chain.new

  // executed immediately.
  // CHECK-NEXT: constructed vt.value(1)
  // CHECK-NEXT: move constructed vt.value(1)
  // CHECK-NEXT: constructed vt.value(2)
  // CHECK-NEXT: move constructed vt.value(2)
  // CHECK-NEXT: constructed vt.value(3)
  // CHECK-NEXT: move constructed vt.value(3)
  // CHECK-NEXT: destroyed vt.value(1)
  // CHECK-NEXT: destroyed vt.value(2)
  // CHECK-NEXT: print vt.value(3)
  // CHECK-NEXT: destroyed vt.value(3)
  // CHECK-NEXT: constructed vt.value(4)
  // CHECK-NEXT: move constructed vt.value(4)
  // CHECK-NEXT: destroyed vt.value(4)
  // CHECK-NEXT: constructed vt.value(5)
  // CHECK-NEXT: move constructed vt.value(5)
  // CHECK-NEXT: destroyed vt.value(5)

  %v1 = "test.vt.constant"() { value = 1 : i32 } : () -> !vt.value
  %v2 = "test.vt.constant"() { value = 2 : i32 } : () -> !vt.value
  %r = "test.vt.add"(%v1, %v2) : (!vt.value,  !vt.value) -> !vt.value
  "test.vt.print"(%r, %ch0) : (!vt.value, !grt.chain) -> (!grt.chain)
  "test.vt.constant"() { value = 4 : i32 } : () -> !vt.value
  "test.vt.constant"() { value = 5 : i32 } : () -> !vt.value

  bef.output
}

// CHECK-LABEL: 'no_op' because it has arguments
bef.func @no_op(%x: !vt.value) -> !vt.value {
  bef.output %x : !vt.value
}

// CHECK-LABEL: --- Running 'call_no_op'
bef.func @call_no_op() {
  // This test uses one vt.value(2) that is all of these:
  // - Function argument: passed to no_op.
  // - Function result: returned by no_op.
  // - Op input: passed to grt.call.
  // - Op output: returned by grt.call.

  // CHECK-NEXT: constructed vt.value(2)
  // CHECK-NEXT: move constructed vt.value(2)
  %c1 = "test.vt.constant"() { value = 2 : i32 } : () -> !vt.value

  // no_op returns its argument, which extends vt.value(2)'s lifetime.
  %x = grt.call @no_op(%c1) : (!vt.value) -> !vt.value

  %ch0 = grt.chain.new

  // CHECK-NEXT: print vt.value(2)
  "test.vt.print"(%x, %ch0) : (!vt.value, !grt.chain) -> (!grt.chain)
  // CHECK-NEXT: destroyed vt.value(2)
  bef.output
}

// CHECK-LABEL: 'add_one' because it has arguments
bef.func @add_one(%x: !vt.value) -> !vt.value {
  %v1 = "test.vt.constant"() { value = 1 : i32 } : () -> !vt.value
  %r = "test.vt.add"(%v1, %x) : (!vt.value,  !vt.value) -> !vt.value
  bef.output %r : !vt.value
}

// CHECK-LABEL: --- Running 'caller'

bef.func @caller() {
  // CHECK-NEXT: constructed vt.value(3)
  // CHECK-NEXT: move constructed vt.value(3)
  %c1 = "test.vt.constant"() { value = 3 : i32 } : () -> !vt.value

  // CHECK-NEXT: constructed vt.value(1)
  // CHECK-NEXT: move constructed vt.value(1)

  // CHECK-NEXT: constructed vt.value(4)
  // CHECK-NEXT: move constructed vt.value(4)
  // CHECK-NEXT: destroyed vt.value(1)

  %x = grt.call @add_one(%c1) : (!vt.value) -> !vt.value

  // TODO: If calls were non-strict, then 3 could be destroyed at the add,
  // before the return.
  // CHECK-NEXT: destroyed vt.value(3)

  %ch0 = grt.chain.new

  // CHECK-NEXT: print vt.value(4)
  "test.vt.print"(%x, %ch0) : (!vt.value, !grt.chain) -> (!grt.chain)
  // CHECK-NEXT: destroyed vt.value(4)
  bef.output
}

// CHECK-NEXT: --- Running 'test_hello'
bef.func @test_hello() {
 // CHECK: hello bef executor!
  %ch0 = grt.chain.new
  %ch1 = "test.print_hello"(%ch0) : (!grt.chain) -> !grt.chain
  bef.output
}
