// RUN: max benchmark --input-generation-schema=%S/../Inputs/relu3x100x100-schema.yaml %S/../Inputs/relu3x100x100.torchscript | FileCheck %s
// CHECK: Compiling model... OK (took {{.+}})
// CHECK: Benchmarking
// CHECK: Result is : VALID
// CHECK: Benchmark finished
