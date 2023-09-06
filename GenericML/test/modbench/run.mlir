// RUN: modbench %S/testdata/relu100x100x3 | FileCheck %s
// CHECK: Modular Benchmarking Tool
// CHECK: Modular version: {{.+}}
// CHECK: Compiling model... OK (took {{.+}})
// CHECK: Benchmarking
// CHECK: Benchmark finished
// CHECK: Average QPS: {{[0-9.]+}}
