// RUN: bef-translate -export-bef -allow-unregistered-dialect %s > %t
// RUN: bef-translate -import-bef %t | FileCheck %s

// RUN: bef-translate -export-bef -allow-unregistered-dialect -enable-bef-metadata %s > %t2
// RUN: bef-translate -import-bef -allow-unregistered-dialect %t2 | FileCheck %s --check-prefix=MD

// Check to make sure the magic number is right, and the first section is there.

// CHECK: module {

// CHECK-LABEL: bef.invokable @trivial() {
// CHECK-NEXT:    %0:2 = bef.opaque "simple.primitive" []() : () -> (!bef.unknown, !bef.unknown)
// CHECK-NEXT:    bef.opaque "grt.invoke" [@trivial]() : () -> ()  
// CHECK-NEXT:    bef.output
// CHECK-NEXT:  }
// MD-LABEL: bef.invokable @trivial() {
// MD-NEXT:    %0:2 = "simple.primitive"() : () -> (i8, i16)
// MD-NEXT:    "grt.invoke"() {target = @trivial} : () -> ()
// MD-NEXT:    bef.output
// MD-NEXT:  }
bef.invokable @trivial() {
  %a, %b = "simple.primitive"() : () -> (i8, i16)
  grt.invoke @trivial() : () -> ()
  bef.output
}

// CHECK-LABEL:  bef.invokable @args_results(%arg0: !bef.unknown, %arg1: !bef.unknown) -> (!bef.unknown, !bef.unknown) {
// CHECK-NEXT:     bef.output %arg1, %arg0 : !bef.unknown, !bef.unknown
// CHECK-NEXT:   }
// MD-LABEL:  bef.invokable @args_results(%arg0: i32, %arg1: i8) -> (i8, i32) {
// MD-NEXT:     bef.output %arg1, %arg0 : i8, i32
// MD-NEXT:   }
bef.invokable @args_results(%a: i32, %b: i8) -> (i8, i32) {
  bef.output %b, %a : i8, i32
}

// CHECK-LABEL:  bef.invokable @attributes() {
// MD-LABEL:  bef.invokable @attributes() {
bef.invokable @attributes() {
  // CHECK-NEXT: "a.b" [true]
  // MD-NEXT: "a.b"() {a = true}
  "a.b"() { a = 1 : i1 }: () -> ()

  // CHECK-NEXT: "a.b" [42 : i8, 17 : i8]
  // MD-NEXT: "a.b"() {b = 42 : i8, x = 17 : i8}
  "a.b"() { b = 42 : i8, x = 17 : i8 }: () -> ()

  // CHECK-NEXT: "a.b" [258 : i16]
  // MD-NEXT: "a.b"() {c = 258 : i16}
  "a.b"() { c = 0x0102 : i16 }: () -> ()

  // CHECK-NEXT: "a.b" [32751241 : i32]
  "a.b"() { d = 32751241 : i32 }: () -> ()

   // CHECK-NEXT: "a.b" [3275124123121]
  "a.b"() { a = 3275124123121 : i64 }: () -> ()

  // CHECK-NEXT: "a.b" ["foo\00bar"]
  "a.b"() { a = "foo\00bar" : i64 }: () -> ()

  // CHECK-NEXT: "a.b" {{.}}[]]
  "a.b"() { a = [] }: () -> ()

  // CHECK-NEXT: "a.b" {{.}}[true, false, true]]
  "a.b"() { a = [true, false, true] }: () -> ()

  // CHECK-NEXT: "a.b" [2.000000e+00 : f32]
  "a.b"() { a = 2.0 : f32 }: () -> ()

  // CHECK-NEXT: "a.b" [3.000000e+00]
  "a.b"() { a = 3.0 : f64 }: () -> ()

  // CHECK-NEXT: "a.b" [{b = 1 : i8, c = [true, false, false]}]
  "a.b"() { a = { b = 1 : i8, c = [true, false, false] } }: () -> ()

  bef.output
   // CHECK-NEXT:     bef.output
}  // CHECK-NEXT:   }

// CHECK-LABEL: bef.invokable @opaque_round_trip() {
// CHECK-NEXT:    %0:2 = bef.opaque "simple.primitive" []() : () -> (!bef.unknown, !bef.unknown)
// CHECK-NEXT:    bef.opaque "other.primitive" [42 : i8](%0#1, %0#0) : (!bef.unknown, !bef.unknown) -> ()
// CHECK-NEXT:    bef.output
// CHECK-NEXT:  }
bef.invokable @opaque_round_trip() {
  %0:2 = "bef.opaque"() {attrs = [], opname = "simple.primitive"} : () -> (!bef.unknown, !bef.unknown)
  "bef.opaque"(%0#1, %0#0) {attrs = [42 : i8], opname = "other.primitive"} : (!bef.unknown, !bef.unknown) -> ()
  bef.output
}

// CHECK-NEXT: }
