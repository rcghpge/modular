// RUN: cmake -E rm "$MODULAR_DERIVED_PATH/visualization-specs/serving_default.maxviz" 2>/dev/null || true

// RUN: max-engine visualize %S/../Inputs/relu100x100x3 | FileCheck %s
// CHECK: Modular Visualization Tool
// CHECK: Modular version: {{.+}}

// RUN: [ -e "$MODULAR_DERIVED_PATH/visualization-specs/serving_default.maxviz" ] || { exit 1; }
