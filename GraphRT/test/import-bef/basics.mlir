// RUN: graphrt-translate -export-bef -allow-unregistered-dialect < %s > %t
// RUN: graphrt-translate -import-bef %t | FileCheck %s

// Check to make sure the magic number is right, and the first section is there.

// CHECK: module {
// CHECK-NEXT: }

bef.func @trivial() {
  "simple.primitive"() : () -> ()
  bef.output
}

bef.func @args_results(%a: i32, %b: i8) -> (i8, i32) {
  bef.output %b, %a : i8, i32
}
