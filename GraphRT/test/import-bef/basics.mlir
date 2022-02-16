// RUN: graphrt-translate -export-bef -allow-unregistered-dialect %s > %t
// RUN: graphrt-translate -import-bef %t | FileCheck %s

// Check to make sure the magic number is right, and the first section is there.

// CHECK: module {

// CHECK-LABEL: bef.func @trivial() {
// CHECK-NEXT:    %0:2 = "bef.opaque"() {attrs = [], opname = "simple.primitive"} : () -> (!bef.unknown, !bef.unknown)
// CHECK-NEXT:    "bef.opaque"(%0#1, %0#0) {attrs = [42 : i8], opname = "other.primitive"} : (!bef.unknown, !bef.unknown) -> ()
// CHECK-NEXT:    bef.output
// CHECK-NEXT:  }
bef.func @trivial() {
  %a, %b = "simple.primitive"() : () -> (i8, i16)
  "other.primitive"(%b, %a) { x = 42 : i8 }: (i16, i8) -> ()
  bef.output
}

// CHECK-LABEL:  bef.func @args_results(%arg0: !bef.unknown, %arg1: !bef.unknown) -> (!bef.unknown, !bef.unknown) {
// CHECK-NEXT:     bef.output %arg1, %arg0 : !bef.unknown, !bef.unknown
// CHECK-NEXT:   }
bef.func @args_results(%a: i32, %b: i8) -> (i8, i32) {
  bef.output %b, %a : i8, i32
}

// CHECK-LABEL:  bef.func @attributes() {
bef.func @attributes() {
  "a.b"() { a = 1 : i1 }: () -> ()   // CHECK-NEXT: attrs = [true]
  "a.b"() { a = 42 : i8 }: () -> ()  // CHECK-NEXT: attrs = [42 : i8]
  "a.b"() { a = 0x0102 : i16 }: () -> ()// CHECK-NEXT: attrs = [258 : i16]
  "a.b"() { a = 32751241 : i32 }: () -> () // CHECK-NEXT: attrs = [32751241 : i32]
  "a.b"() { a = 3275124123121 : i64 }: () -> ()  // CHECK-NEXT: attrs = [3275124123121]
  "a.b"() { a = "foo\00bar" : i64 }: () -> () // CHECK-NEXT: attrs = ["foo\00bar"]
  "a.b"() { a = [] }: () -> () // CHECK-NEXT: attrs = {{.}}[]]
  "a.b"() { a = [true, false, true] }: () -> () // CHECK-NEXT: [true, false, true]

  bef.output  // CHECK-NEXT:     bef.output
}  // CHECK-NEXT:   }

// CHECK-NEXT: }
