// RUN: max benchmark --input-data-schema=%S/../Inputs/relu100x100x3-schema.yaml --mlperf-scenario=offline %S/../Inputs/relu100x100x3 | FileCheck %s
// CHECK: MLPerf scenario: Offline (use --mlperf-scenario to change)
