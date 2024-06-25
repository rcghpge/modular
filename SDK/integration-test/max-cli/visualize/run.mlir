// (TODO: MSDK-487)
// REQUIRES: disabled
// RUN: cmake -E rm "serving_default.maxviz" 2>/dev/null || true

// RUN: max visualize %S/../../Inputs/relu100x100x3 | FileCheck %s
// CHECK: Modular Visualization Tool
// CHECK: Modular version: {{.+}}

// RUN: [ -e "serving_default.maxviz" ] || { exit 1; }
