// RUN: bef-translate -export-bef -allow-unregistered-dialect %s > %t.bef
// RUN: bef-executor -verify-diagnostics %t.bef | FileCheck %s

// CHECK: Initializing leak check allocator

// CHECK-LABEL: --- Running 'basic_test'
bef.invokable @basic_test() -> !grt.chain {
  %ch0 = grt.chain.new
  %v1 = "test.vec.atoi"() { size = 5 : i32 } : () -> !test.vec
  %ch1 = "test.vec.print" (%v1, %ch0) : (!test.vec, !grt.chain) -> !grt.chain
  // CHECK-NEXT: vec = 0, 1, 2, 3, 4 

  %v2 = "test.vec.constant"() { size = 5 : i32, value = 3 : i32 } : () -> !test.vec
  %ch2 = "test.vec.print" (%v2, %ch1) : (!test.vec, !grt.chain) -> !grt.chain
  // CHECK-NEXT: vec = 3, 3, 3, 3, 3 

  %v3 = "test.vec.add"(%v1, %v2) : (!test.vec, !test.vec) -> !test.vec
  %ch3 = "test.vec.print" (%v3, %ch2) : (!test.vec, !grt.chain) -> !grt.chain
  // CHECK-NEXT: vec = 3, 4, 5, 6, 7

  %v4 = "test.vec.add"(%v3, %v1) : (!test.vec, !test.vec) -> !test.vec
  %ch4 = "test.vec.print" (%v4, %ch3) : (!test.vec, !grt.chain) -> !grt.chain
  // CHECK-NEXT: vec = 3, 5, 7, 9, 11

  // This will produce an error due to mismatched lengths.
  %v5 = "test.vec.constant"() { size = 3 : i32, value = 3 : i32 } : () -> !test.vec

  // expected-error @+1 {{vector length mismatch}}
  %v6 = "test.vec.add"(%v5, %v1) : (!test.vec, !test.vec) -> !test.vec
  %ch5 = "test.vec.print" (%v6, %ch4) : (!test.vec, !grt.chain) -> !grt.chain
  // CHECK-NEXT: 'basic_test' returned <error: vector length mismatch> 

  bef.output %ch5 : !grt.chain
}

// CHECK-LABEL: --- Running 'async_add_test'
bef.invokable @async_add_test() -> !grt.chain {
  %ch0 = grt.chain.new
  %v1 = "test.vec.atoi"() { size = 5 : i32 } : () -> !test.vec
  %ch1 = "test.vec.print" (%v1, %ch0) : (!test.vec, !grt.chain) -> !grt.chain
  // CHECK-NEXT: vec = 0, 1, 2, 3, 4 

  %v2 = "test.vec.constant"() { size = 5 : i32, value = 3 : i32 } : () -> !test.vec
  %ch2 = "test.vec.print" (%v2, %ch1) : (!test.vec, !grt.chain) -> !grt.chain
  // CHECK-NEXT: vec = 3, 3, 3, 3, 3 

  %v3 = "test.vec.add.async"(%v1, %v2) : (!test.vec, !test.vec) -> !test.vec
  %ch3 = "test.vec.print" (%v3, %ch2) : (!test.vec, !grt.chain) -> !grt.chain
  // CHECK-NEXT: vec = 3, 4, 5, 6, 7

  %v4 = "test.vec.add.async"(%v3, %v1) : (!test.vec, !test.vec) -> !test.vec
  %ch4 = "test.vec.print" (%v4, %ch3) : (!test.vec, !grt.chain) -> !grt.chain
  // CHECK-NEXT: vec = 3, 5, 7, 9, 11

  // This will produce an error due to mismatched lengths.
  %v5 = "test.vec.constant"() { size = 3 : i32, value = 3 : i32 } : () -> !test.vec

  // expected-error @+1 {{vector length mismatch}}
  %v6 = "test.vec.add.async"(%v5, %v1) : (!test.vec, !test.vec) -> !test.vec
  %ch5 = "test.vec.print" (%v6, %ch4) : (!test.vec, !grt.chain) -> !grt.chain
  // CHECK-NEXT: 'async_add_test' returned <error: vector length mismatch> 

  bef.output %ch5 : !grt.chain
}

// CHECK-LABEL: --- Running 'async_add_test_par_foreach'
bef.invokable @async_add_test_par_foreach() -> !grt.chain {
  %ch0 = grt.chain.new
  %v1 = "test.vec.atoi"() { size = 5 : i32 } : () -> !test.vec
  %ch1 = "test.vec.print" (%v1, %ch0) : (!test.vec, !grt.chain) -> !grt.chain
  // CHECK-NEXT: vec = 0, 1, 2, 3, 4 

  %v2 = "test.vec.constant"() { size = 5 : i32, value = 3 : i32 } : () -> !test.vec
  %ch2 = "test.vec.print" (%v2, %ch1) : (!test.vec, !grt.chain) -> !grt.chain
  // CHECK-NEXT: vec = 3, 3, 3, 3, 3 

  %v3 = "test.vec.add.async2"(%v1, %v2) : (!test.vec, !test.vec) -> !test.vec
  %ch3 = "test.vec.print" (%v3, %ch2) : (!test.vec, !grt.chain) -> !grt.chain
  // CHECK-NEXT: vec = 3, 4, 5, 6, 7

  %v4 = "test.vec.add.async3"(%v3, %v1) : (!test.vec, !test.vec) -> !test.vec
  %ch4 = "test.vec.print" (%v4, %ch3) : (!test.vec, !grt.chain) -> !grt.chain
  // CHECK-NEXT: vec = 3, 5, 7, 9, 11

  %v5 = "test.vec.constant"() { size = 5 : i32, value = 12 : i32 } : () -> !test.vec
  %v6 = "test.vec.add.async4"(%v5, %v1) : (!test.vec, !test.vec) -> !test.vec
  %ch5 = "test.vec.print" (%v6, %ch4) : (!test.vec, !grt.chain) -> !grt.chain
  // CHECK-NEXT: vec = 12, 13, 14, 15, 16

  // CHECK-NEXT: 'async_add_test_par_foreach' returned chain

  bef.output %ch5 : !grt.chain
}
