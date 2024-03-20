// RUN: cmake -E rm "relu100x100x3.maxviz" 2>/dev/null || true

// RUN: max visualize %S/../Inputs/relu100x100x3 | FileCheck %s
// CHECK: Modular Visualization Tool
// CHECK: Modular version: {{.+}}

// RUN: [ -e "relu100x100x3.maxviz" ] || { exit 1; }
