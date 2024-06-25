// (TODO: MSDK-485)
// REQUIRES: disabled
// RUN: max benchmark --input-data-schema=%S/Inputs/sqrt1-schema.yaml %S/Inputs/sqrt1 | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: mojo package %S/../../Inputs/test_user_op -o %t.mojopkg
// RUN: max benchmark --input-data-schema=%S/Inputs/sqrt1-schema.yaml %S/Inputs/sqrt1 --custom-ops-path=%t.mojopkg | FileCheck %s --check-prefix=CHECK-WITH-CUSTOM-OPS

// CHECK-DEFAULT: Modular Benchmarking Tool
// CHECK-DEFAULT: Modular version: {{.+}}
// CHECK-DEFAULT: Compiling model... OK (took {{.+}})
// CHECK-DEFAULT: Benchmarking
// CHECK-DEFAULT: Benchmark finished
// CHECK-DEFAULT-NOT: My sqrt

// CHECK-WITH-CUSTOM-OPS: Modular Benchmarking Tool
// CHECK-WITH-CUSTOM-OPS: Modular version: {{.+}}
// CHECK-WITH-CUSTOM-OPS: Compiling model... OK (took {{.+}})
// CHECK-WITH-CUSTOM-OPS: Benchmarking
// CHECK-WITH-CUSTOM-OPS: My sqrt
// CHECK-WITH-CUSTOM-OPS: Benchmark finished
