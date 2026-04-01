// RUN: mo-opt %s --canonicalize | FileCheck %s

// Test: tensor.bundle and tensor.unbundle round-trip.
// CHECK-LABEL: mo.graph @tensor_bundle_roundtrip
mo.graph @tensor_bundle_roundtrip(
    %t0: !mo.tensor<[3], f32, gpu:0>,
    %t1: !mo.tensor<[3], f32, gpu:1>
) -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>) {
  %b = mo.tensor.bundle(%t0, %t1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>)
  // unbundle(bundle(x)) folds away:
  // CHECK-NOT: mo.tensor.bundle
  // CHECK-NOT: mo.tensor.unbundle
  %r0, %r1 = mo.tensor.unbundle(%b)
      : (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  mo.output %r0, %r1 : !mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>
}
