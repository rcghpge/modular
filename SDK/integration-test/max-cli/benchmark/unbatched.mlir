// (TODO: MSDK-486)
// REQUIRES: disabled
// RUN: max benchmark %S/Inputs/unbatched 2>&1 | FileCheck %s
// CHECK: Compiling model... OK (took {{.+}})
// CHECK: Benchmarking
// CHECK: Benchmark finished
