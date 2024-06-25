// RUN: max benchmark --input-data-schema=%S/../../Inputs/relu3x100x100-schema.yaml --mlperf-scenario=offline %S/../../Inputs/relu3x100x100.torchscript | FileCheck %s
// CHECK: MLPerf scenario: Offline (use --mlperf-scenario to change)
