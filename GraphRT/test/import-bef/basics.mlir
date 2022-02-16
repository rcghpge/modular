// RUN: graphrt-translate -export-bef -allow-unregistered-dialect %s > %t
// RUN: graphrt-translate -import-bef %t | FileCheck %s

// RUN: graphrt-translate -export-bef -allow-unregistered-dialect -enable-bef-metadata %s > %t2
// RUN: graphrt-translate -import-bef %t2 | FileCheck %s --check-prefix=MD

// Check to make sure the magic number is right, and the first section is there.

// CHECK: module {

// CHECK-LABEL: bef.func @trivial() {
// CHECK-NEXT:    %0:2 = "bef.opaque"() {attrs = [], opname = "simple.primitive"} : () -> (!bef.unknown, !bef.unknown)
// CHECK-NEXT:    "bef.opaque"(%0#1, %0#0) {attrs = [42 : i8], nonstrict, opname = "other.primitive"} : (!bef.unknown, !bef.unknown) -> ()
// CHECK-NEXT:    bef.output
// CHECK-NEXT:  }
// MD-LABEL: bef.func @trivial() {
// MD-NEXT:    %0:2 = "bef.opaque"() {attrs = [], opname = "simple.primitive"} : () -> (!bef.unknown, !bef.unknown)
// MD-NEXT:    "bef.opaque"(%0#1, %0#0) {attrs = [42 : i8], bef.attr.names = ["x"], nonstrict, opname = "other.primitive"} : (!bef.unknown, !bef.unknown) -> ()
// MD-NEXT:    bef.output
// MD-NEXT:  }
bef.func @trivial() {
  %a, %b = "simple.primitive"() : () -> (i8, i16)
  "other.primitive"(%b, %a) { x = 42 : i8, bef.nonstrict }: (i16, i8) -> ()
  bef.output
}

// CHECK-LABEL:  bef.func @args_results(%arg0: !bef.unknown, %arg1: !bef.unknown) -> (!bef.unknown, !bef.unknown) {
// CHECK-NEXT:     bef.output %arg1, %arg0 : !bef.unknown, !bef.unknown
// CHECK-NEXT:   }
bef.func @args_results(%a: i32, %b: i8) -> (i8, i32) {
  bef.output %b, %a : i8, i32
}

// CHECK-LABEL:  bef.func @attributes() {
// MD-LABEL:  bef.func @attributes() {
bef.func @attributes() {
  // CHECK-NEXT: attrs = [true]
  // MD-NEXT: attrs = [true], bef.attr.names = ["a"]
  "a.b"() { a = 1 : i1 }: () -> ()

  // CHECK-NEXT: attrs = [42 : i8, 17 : i8]
  // MD-NEXT: attrs = [42 : i8, 17 : i8], bef.attr.names = ["b", "x"],
  "a.b"() { b = 42 : i8, x = 17 : i8 }: () -> ()

  // CHECK-NEXT: attrs = [258 : i16]
  "a.b"() { c = 0x0102 : i16 }: () -> ()

  // CHECK-NEXT: attrs = [32751241 : i32]
  "a.b"() { d = 32751241 : i32 }: () -> ()

   // CHECK-NEXT: attrs = [3275124123121]
  "a.b"() { a = 3275124123121 : i64 }: () -> ()

  // CHECK-NEXT: attrs = ["foo\00bar"]
  "a.b"() { a = "foo\00bar" : i64 }: () -> ()

  // CHECK-NEXT: attrs = {{.}}[]]
  "a.b"() { a = [] }: () -> ()

  // CHECK-NEXT: [true, false, true]
  "a.b"() { a = [true, false, true] }: () -> ()

  // CHECK-NEXT: attrs = [2.000000e+00 : f32]
  "a.b"() { a = 2.0 : f32 }: () -> ()

  // CHECK-NEXT: attrs = [3.000000e+00]
  "a.b"() { a = 3.0 : f64 }: () -> ()

  bef.output
   // CHECK-NEXT:     bef.output
}  // CHECK-NEXT:   }

// CHECK-LABEL: bef.func @opaque_round_trip() {
// CHECK-NEXT:    %0:2 = "bef.opaque"() {attrs = [], opname = "simple.primitive"} : () -> (!bef.unknown, !bef.unknown)
// CHECK-NEXT:    "bef.opaque"(%0#1, %0#0) {attrs = [42 : i8], opname = "other.primitive"} : (!bef.unknown, !bef.unknown) -> ()
// CHECK-NEXT:    bef.output
// CHECK-NEXT:  }

bef.func @opaque_round_trip() {
  %0:2 = "bef.opaque"() {attrs = [], opname = "simple.primitive"} : () -> (!bef.unknown, !bef.unknown)
  "bef.opaque"(%0#1, %0#0) {attrs = [42 : i8], opname = "other.primitive"} : (!bef.unknown, !bef.unknown) -> ()
  bef.output
}

// CHECK-NEXT: }
