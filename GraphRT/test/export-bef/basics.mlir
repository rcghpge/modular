// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t
// RUN: od -t x1 %t | FileCheck %s

// Check to make sure the magic number is right, and the first section is there.

// CHECK: 0000000 ee f0 03

mef.invokable @trivial() {
  "simple.primitive"() { x = 42 : si8 }: () -> ()
  mef.output
}

mef.invokable @args_results(%a: si32, %b: si8) -> (si8, si32) {
  mef.output %b, %a : si8, si32
}

mef.invokable @unknown_args(%a: !mef.unknown, %b: !mef.unknown) -> !mef.unknown {
  mef.output %b : !mef.unknown
}
