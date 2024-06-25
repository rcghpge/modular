// RUN: max benchmark --input-data-schema=%S/../../Inputs/relu3x100x100-schema.yaml %S/../../Inputs/relu3x100x100.torchscript | FileCheck %s
// CHECK: Modular Benchmarking Tool
// CHECK: Compiling model... OK (took {{.+}})
// CHECK: Benchmarking
// CHECK: Benchmark finished
