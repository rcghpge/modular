// RUN: mo-opt %s --mogg-import-packages -mo-to-mogg -mogg-to-mgp -canonicalize -mogg-outline-kernels -mogg-add-try-except-blocks 2>&1 | FileCheck %s

// Verify that the three ops introduced by the MOGG ABI lowering are present
// after mogg-outline-kernels and mogg-add-try-except-blocks:
//
//   mogg.with_trace  -- created by mogg-outline-kernels (createKernelStub)
//                       to wrap the kernel terminator in a Trace context
//   mogg.try / mogg.raise -- created by mogg-add-try-except-blocks for MOGG
//                            ABI stubs; the except block re-raises via mogg.raise

// The MGP ABI stub (@region_0) wraps the call to the MOGG stub in a
// mogg.try/except that returns an async error value instead of raising.
// CHECK-LABEL: mogg.kernel.stub @region_0
// CHECK:         mogg.try {
// CHECK-NOT:       mogg.raise
// CHECK:         } except {

// The MOGG ABI stub (@stub_0) has mogg.with_trace around the kernel
// terminator and mogg.try/mogg.raise for error propagation.
// CHECK-LABEL: mogg.kernel.stub @stub_0
// CHECK:         mogg.try {
// CHECK:           mogg.with_trace
// CHECK:             mogg.call.foreach
// CHECK:           }
// CHECK:           mogg.yield
// CHECK:         } except {
// CHECK:           mogg.raise

mo.graph @relu(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %result = mo.activation(%x) {activation = "relu"} : !mo.tensor<[4], f32>
  mo.output %result : !mo.tensor<[4], f32>
}
