// RUN: bef-translate -export-bef -allow-unregistered-dialect < %s > %t
// RUN: od -t x1 %t | FileCheck %s

// Check to make sure the magic number is right, and the first section is there.

// CHECK: 0000000 be f0 03

bef.invokable @trivial() {
  "simple.primitive"() { x = 42 : i8 }: () -> ()
  bef.output
}

bef.invokable @args_results(%a: i32, %b: i8) -> (i8, i32) {
  bef.output %b, %a : i8, i32
}

bef.invokable @unknown_args(%a: !bef.unknown, %b: !bef.unknown) -> !bef.unknown {
  bef.output %b : !bef.unknown
}
