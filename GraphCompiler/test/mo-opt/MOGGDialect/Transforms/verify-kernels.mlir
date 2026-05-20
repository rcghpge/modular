// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mogg-verify-kernels | FileCheck %s

// Verify that mogg-verify-kernels passes without error when processing
// mogg.kernel.decl ops produced by mogg-import-packages (KernelDeclAdaptor
// path).

// CHECK-DAG: mogg.kernel.decl @MOGGKernelAPITest.MOGGKernelAPITest.Foo registration = "imposter_add"
// CHECK-DAG: mogg.kernel.decl @MOGGKernelAPITest.MOGGKernelAPITest.AddFuseLHS registration = "imposter_add_lhs"
// CHECK-DAG: mogg.kernel.decl @MOGGKernelAPITest.MOGGKernelAPITest.AddElementwise registration = "imposter_add_elementwise"

module {}
