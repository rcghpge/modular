// RUN: graphrt-translate -export-bef -allow-unregistered-dialect %s > %t
// RUN: graphrt-translate -import-bef %t | FileCheck %s

// Check to make sure the magic number is right, and the first section is there.

// CHECK: module {

// CHECK-LABEL: bef.func @trivial() {
// CHECK-NEXT:   "bef.opaque"() {attrs = [], opname = "simple.primitive"} : () -> ()
// CHECK-NEXT:   "bef.opaque"() {attrs = [42 : i8], opname = "other.primitive"} : () -> ()
// CHECK-NEXT:   bef.output
// CHECK-NEXT: }
bef.func @trivial() {
  "simple.primitive"() : () -> ()
  "other.primitive"() { x = 42 : i8 }: () -> ()
  bef.output
}

// CHECK-LABEL:  bef.func @args_results(%arg0: !bef.unknown, %arg1: !bef.unknown) -> (!bef.unknown, !bef.unknown) {
// CHECK-NEXT:     bef.output %arg1, %arg0 : !bef.unknown, !bef.unknown
// CHECK-NEXT:   }
bef.func @args_results(%a: i32, %b: i8) -> (i8, i32) {
  bef.output %b, %a : i8, i32
}


// CHECK-NEXT: }
