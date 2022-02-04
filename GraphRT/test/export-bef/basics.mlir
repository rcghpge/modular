// RUN: graphrt-translate -export-bef < %s > %t
// RUN: od -t x1 %t | FileCheck %s

// Empty file.  Check to make sure the magic number is right.

// CHECK: 0000000 be f0 
// CHECK: 0000002

bef.func @trivial() {
  //"simple.primitive"() : () -> ()
  bef.output
}

bef.func @args_results(%a: i32, %b: i8) -> (i8, i32) {
  bef.output %b, %a : i8, i32
}
