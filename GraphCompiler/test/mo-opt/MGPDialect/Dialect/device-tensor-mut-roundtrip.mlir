// RUN: mo-opt %s --allow-unregistered-dialect | FileCheck %s

// Round-trip of the optional `mut` marker on !mgp.tensor. The marker is the
// lowered form of !mo.buffer; unmarked tensors must print exactly as before.

// CHECK-LABEL: func.func @mut_type_round_trip
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<4x4xf32, mut>
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<464x4800xf16, <"gpu", 0>, mut>
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<?x?xf32, mut>
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<8xf32>
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<8xf32, <"gpu", 1>>
func.func @mut_type_round_trip(
    %host_mut: !mgp.tensor<4x4xf32, mut>,
    %gpu_mut: !mgp.tensor<464x4800xf16, <"gpu", 0>, mut>,
    %dyn_mut: !mgp.tensor<?x?xf32, mut>,
    %plain: !mgp.tensor<8xf32>,
    %gpu_plain: !mgp.tensor<8xf32, <"gpu", 1>>) {
  mef.output
}

// CHECK-LABEL: func.func @tensor_create_mut
func.func @tensor_create_mut(
    %ch: !mef.chain,
    %buf: !mgp.buffer<<"gpu", 0>>,
    %spec: !mgp.tensor_spec<3x5xf32>) -> !mgp.tensor<3x5xf32, <"gpu", 0>, mut> {
  // CHECK: mgp.tensor.create %{{.*}}, %{{.*}}, %{{.*}} {mut} : !mgp.tensor<3x5xf32, <"gpu", 0>, mut>
  %0 = mgp.tensor.create %ch, %buf, %spec {mut} : !mgp.tensor<3x5xf32, <"gpu", 0>, mut>
  mef.output %0 : !mgp.tensor<3x5xf32, <"gpu", 0>, mut>
}

// The slice result derives its mutability from the sliced tensor.
// CHECK-LABEL: func.func @tensor_slice_propagates_mut
func.func @tensor_slice_propagates_mut(
    %input: !mgp.tensor<5xf32, <"gpu", 0>, mut>,
    %spec: !mgp.tensor_spec<3xf32>,
    %start: !mgp.tensor<1xsi64>,
    %dctx: !mgp.device_context<<"gpu", 0>>) -> !mgp.tensor<3xf32, <"gpu", 0>, mut> {
  // CHECK: mgp.tensor.slice
  %sliced = mgp.tensor.slice %input, %spec, %start, %dctx : !mgp.tensor<5xf32, <"gpu", 0>, mut>, !mgp.tensor_spec<3xf32>, !mgp.tensor<1xsi64>, !mgp.device_context<<"gpu", 0>>
  // CHECK: mef.output %{{.*}} : !mgp.tensor<3xf32, <"gpu", 0>, mut>
  mef.output %sliced : !mgp.tensor<3xf32, <"gpu", 0>, mut>
}
