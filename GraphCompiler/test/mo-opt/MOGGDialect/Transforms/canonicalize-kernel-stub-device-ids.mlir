// RUN: mo-opt %s --mogg-canonicalize-kernel-stub-device-ids --allow-unregistered-dialect | FileCheck %s
// RUN: mo-opt %s --mogg-canonicalize-kernel-stub-device-ids --mogg-merge-identical-kernel-stubs --allow-unregistered-dialect | FileCheck %s --check-prefix=MERGE

//===----------------------------------------------------------------------===//
// The pass rewrites every device id reachable inside a kernel stub -- in op
// attributes and in the `!mgp.tensor` types it operates on -- to the canonical
// id 0, keeping the label. Per-device copies of a kernel (e.g. across EP shards)
// then hash equal and merge. Ops and types outside a stub keep their real
// device ids. See GEX-4239.
//===----------------------------------------------------------------------===//

// Two stubs identical but for the device id (gpu:3 vs gpu:5), in both an op
// attribute and a result type.

// CHECK-LABEL: mogg.kernel.stub @k0 private
// CHECK: "test.op"() {dev = #M.device_ref<"gpu", 0>}
// CHECK: -> !mgp.tensor<4xf32, <"gpu", 0>>
mogg.kernel.stub @k0 private() fusionInfo = k attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", isAsync = true} {
  "test.op"() {dev = #M.device_ref<"gpu", 3>} : () -> ()
  %0 = "test.produce"() : () -> !mgp.tensor<4xf32, <"gpu", 3>>
  mogg.output
}

// CHECK-LABEL: mogg.kernel.stub @k1 private
// CHECK: "test.op"() {dev = #M.device_ref<"gpu", 0>}
// CHECK: -> !mgp.tensor<4xf32, <"gpu", 0>>
mogg.kernel.stub @k1 private() fusionInfo = k attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", isAsync = true} {
  "test.op"() {dev = #M.device_ref<"gpu", 5>} : () -> ()
  %0 = "test.produce"() : () -> !mgp.tensor<4xf32, <"gpu", 5>>
  mogg.output
}

// After canonicalization the two stubs are identical, so both call sites in the
// caller collapse onto a single survivor.
// MERGE-LABEL: mogg.kernel.stub @caller private
// MERGE:      mogg.call.stub @[[K:k[01]]]
// MERGE-NEXT: mogg.call.stub @[[K]]
// MERGE-NOT:  mogg.call.stub @k
mogg.kernel.stub @caller private() fusionInfo = caller attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", isAsync = true} {
  mogg.call.stub @k0
  mogg.call.stub @k1
  mogg.output
}

// A device id OUTSIDE any kernel stub is left untouched.
// CHECK-LABEL: func.func @outside
// CHECK: #M.device_ref<"gpu", 3>
func.func @outside() attributes {dev = #M.device_ref<"gpu", 3>} {
  return
}
