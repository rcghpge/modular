// RUN: bef-translate -export-bef -allow-unregistered-dialect %s -o %t
// RUN: od -t x1 %t | FileCheck %s

// Check to make sure the magic number is right, and the first section is there.

// CHECK: 0000000 be f0 03

bef.invokable @trivial() {
  "simple.primitive"() { x = 42 : si8 }: () -> ()
  bef.output
}

bef.invokable @args_results(%a: si32, %b: si8) -> (si8, si32) {
  bef.output %b, %a : si8, si32
}

bef.invokable @unknown_args(%a: !bef.unknown, %b: !bef.unknown) -> !bef.unknown {
  bef.output %b : !bef.unknown
}
