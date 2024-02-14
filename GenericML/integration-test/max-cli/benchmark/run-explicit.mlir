// RUN: max benchmark --input-data-schema=%S/../Inputs/relu100x100x3-schema.yaml %S/../Inputs/relu100x100x3 | FileCheck %s
// CHECK: Modular Benchmarking Tool
// CHECK: Compiling model... OK (took {{.+}})
// CHECK: Benchmarking
// CHECK: Benchmark finished
