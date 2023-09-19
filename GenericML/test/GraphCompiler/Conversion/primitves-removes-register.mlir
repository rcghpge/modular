// RUN: mo-opt %s  --add-devices --jit-compile-kernels --mo-to-primitives --allow-unregistered-dialect | FileCheck %s
// Test invoking `mo-to-primitives` removes any lit register ops.

mo.graph @empty() -> () {
  mo.output
}


// CHECK-NOT: mogg.lit_register
