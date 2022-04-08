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
// MD-LABEL:  bef.invokable @args_results(%arg0: si32, %arg1: si8) -> (si8, si32) {
// MD-NEXT:     bef.output %arg1, %arg0 : si8, si32
// MD-NEXT:   }
bef.invokable @args_results(%a: si32, %b: si8) -> (si8, si32) {
  bef.output %b, %a : si8, si32
}

// CHECK-LABEL:  bef.invokable @attributes() {
// MD-LABEL:  bef.invokable @attributes() {
bef.invokable @attributes() {
  // CHECK-NEXT: "a.b" [true]
  // MD-NEXT: "a.b"() {a = true}
  "a.b"() { a = 1 : i1 }: () -> ()

  // CHECK-NEXT: "a.b" [42 : si8, 17 : si8]
  // MD-NEXT: "a.b"() {b = 42 : si8, x = 17 : si8}
  "a.b"() { b = 42 : si8, x = 17 : si8 }: () -> ()

  // CHECK-NEXT: "a.b" [258 : si16]
  // MD-NEXT: "a.b"() {c = 258 : si16}
  "a.b"() { c = 0x0102 : si16 }: () -> ()

  // CHECK-NEXT: "a.b" [32751241 : si32]
  "a.b"() { d = 32751241 : si32 }: () -> ()

   // CHECK-NEXT: "a.b" [3275124123121 : si64]
  "a.b"() { a = 3275124123121 : si64 }: () -> ()

  // CHECK-NEXT: "a.b" ["foo\00bar"]
  "a.b"() { a = "foo\00bar" : si64 }: () -> ()

  // CHECK-NEXT: "a.b" [dense<> : tensor<0xsi32>]
  "a.b"() { a = dense<[]> : tensor<0xsi32> }: () -> ()

  // CHECK-NEXT: "a.b" [dense<[true, false, true]> : tensor<3xi1>]
  "a.b"() { a = dense<[true, false, true]> : tensor<3xi1> }: () -> ()

  // CHECK-NEXT: "a.b" [2.000000e+00 : f32]
  "a.b"() { a = 2.0 : f32 }: () -> ()

  // CHECK-NEXT: "a.b" [3.000000e+00]
  "a.b"() { a = 3.0 : f64 }: () -> ()

  // Dictionary attribute.
  // CHECK-NEXT: bef.opaque "a.b" [{b = 1 : si8, c = [true, false, false]}]
  "a.b"() { a = { b = 1 : si8, c = [true, false, false] } }: () -> ()

  // List Attribute.
  // CHECK-NEXT: bef.opaque "a.b"  [1 : si8, [1 : ui32, 2 : si8]]
  // MD: "a.b"() {a = 1 : si8, b = [1 : ui32, 2 : si8]} : () -> ()
  "a.b"() {a = 1: si8, b = [1: ui32, 2: si8] } : () -> ()

  bef.output
   // CHECK-NEXT:     bef.output
}  // CHECK-NEXT:   }

// CHECK-LABEL: bef.invokable @opaque_round_trip() {
// CHECK-NEXT:    %0:2 = bef.opaque "simple.primitive" []() : () -> (!bef.unknown, !bef.unknown)
// CHECK-NEXT:    bef.opaque "other.primitive" [42 : si8](%0#1, %0#0) : (!bef.unknown, !bef.unknown) -> ()
// CHECK-NEXT:    bef.output
// CHECK-NEXT:  }
bef.invokable @opaque_round_trip() {
  %0:2 = "bef.opaque"() {attrs = [], opname = "simple.primitive"} : () -> (!bef.unknown, !bef.unknown)
  "bef.opaque"(%0#1, %0#0) {attrs = [42 : si8], opname = "other.primitive"} : (!bef.unknown, !bef.unknown) -> ()
  bef.output
}

// CHECK-NEXT: }
