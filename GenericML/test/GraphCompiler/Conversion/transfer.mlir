// RUN: mo-opt %s --mo-to-primitives | FileCheck %s


// This test makes sure we can properly lower `mo.transfer` with unknown dims.
// In a sense `mo.transfer` is similar to `mo.rebind`, because they both need to
// (and are able to) handle remaining `?` in the graph, which might come from
// non-MO op or list ops. They can easily handle this because their output has
// the same shape as the input.
// CHECK-LABEL: mgp.model @transfer_with_unknown_dims : mof
//       CHECK: } execute ([[CH:%.*]]: !mef.chain, [[CTX:%.*]]: !mgp.context, {{.*}}, [[INPUT:.*]]: !mgp.tensor<?xsi64>) -> (!mef.chain, !mgp.tensor<?xsi64, <"cpu", 1>>) {
//  CHECK-NEXT:   [[IN_BUF:%.*]] = mgp.tensor.extract.buffer [[CTX]][1], [[INPUT]] : !mgp.tensor<?xsi64>
//  CHECK-NEXT:   [[IN_SPEC:%.*]] = mgp.tensor.extract.tensor_spec [[CTX]][1], [[INPUT]] : !mgp.tensor<?xsi64>
//  CHECK-NEXT:   [[IN_SIZE:%.*]] = mgp.tensor_spec.size [[IN_SPEC]] : !mgp.tensor_spec<?xsi64>
//  CHECK-NEXT:   {{.*}} = mgp.chain.host_to_device
//  CHECK-NEXT:   [[NEW_BUF:%.*]] = mgp.buffer.alloc.dynamic {{.*}}, [[CTX]][2], [[IN_SIZE]], align {{.*}} : !mgp.buffer<, <"cpu", 1>>
//  CHECK-NEXT:   {{.*}} = mgp.chain.host_to_device
//  CHECK-NEXT:   [[OUT_CH:%.*]] = mgp.buffer.host_to_device {{.*}}, [[CTX]][1, 2], [[IN_BUF]], [[NEW_BUF]] : !mgp.buffer<> -> !mgp.buffer<, <"cpu", 1>>
//  CHECK-NEXT:   [[OUTPUT:%.*]] = mgp.tensor.create [[OUT_CH]], [[CTX]][2], [[NEW_BUF]], [[IN_SPEC]] : !mgp.tensor<?xsi64, <"cpu", 1>>
//  CHECK-NEXT:   mef.output [[CH]], [[OUTPUT]]
//  CHECK-NEXT: }
mo.graph @transfer_with_unknown_dims(
  %input: !mo.tensor<[?], si64, {device = #M.device_ref<"cpu", 0>}>
) -> !mo.tensor<[?], si64, {device = #M.device_ref<"cpu", 1>}> attributes {
  // Lowering requires us to supply device spec for each device ref.
  device_specs = #M.device_spec_collection<host = <"cpu", 0>, devices = [
    <ref = <"cpu", 0>, target = <triple = "arm64-apple-darwin22.4.0", arch = "apple-m1", features = "">>,
    <ref = <"cpu", 1>, target = <triple = "arm64-apple-darwin22.4.0", arch = "apple-m1", features = "">>
  ]>} {
  %output = mo.transfer %input : !mo.tensor<[?], si64, {device = #M.device_ref<"cpu", 0>}> to <"cpu", 1>
  mo.output %output : !mo.tensor<[?], si64, {device = #M.device_ref<"cpu", 1>}>
}
