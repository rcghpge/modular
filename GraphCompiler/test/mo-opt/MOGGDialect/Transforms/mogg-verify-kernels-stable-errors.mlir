// RUN: not mo-opt %s --mogg-import-packages="mojo-pkg-paths=%stable_invalid_pkg" --mogg-verify-kernels 2>&1 | FileCheck %s

// Each kernel in the stable_invalid_stubs package puts a stable tensor
// argument (`StableTensor` / `StableVariadicTensors`) where the verifier
// forbids it. Slot-backed arguments are a device-graph recording concept, so
// they may only appear as inputs of `build_graph`. The verifier reports every
// violation rather than stopping at the first.

// CHECK-DAG: error: stable tensor arguments (StableTensor / StableVariadicTensors) are only supported on build_graph functions (argument at position 1 of the execute function)
// CHECK-DAG: error: stable tensor arguments (StableTensor / StableVariadicTensors) are only supported on build_graph functions (argument at position 0 of the shape function)
// CHECK-DAG: error: a stable tensor argument cannot be an output (argument at position 0 of the build_graph function)

module {}
