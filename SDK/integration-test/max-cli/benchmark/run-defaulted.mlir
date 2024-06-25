// (TODO: MSDK-484)
// REQUIRES: disabled
// RUN: max benchmark %S/../../Inputs/relu100x100x3 | FileCheck %s
// CHECK: Modular Benchmarking Tool
// CHECK: Modular version: {{.+}}
// CHECK: MLPerf scenario: Single Stream (use --mlperf-scenario to change)
// CHECK: Compiling model... OK (took {{.+}})
// CHECK: Benchmarking
// CHECK: Result is : {{VALID|INVALID}}
// CHECK: QPS w/ loadgen overhead : {{[0-9.e+-]+}}
// CHECK: Min latency (ns) : {{[0-9]+}}
// CHECK: Max latency (ns) : {{[0-9]+}}
// CHECK: 50.00 percentile latency (ns) : {{[0-9]+}}
// CHECK: 99.00 percentile latency (ns) : {{[0-9]+}}
// CHECK: Benchmark finished
