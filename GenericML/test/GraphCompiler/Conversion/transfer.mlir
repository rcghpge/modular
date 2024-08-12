// RUN: mo-opt %s --mo-to-primitives | FileCheck %s


// This test makes sure we can properly lower `mo.transfer` with unknown dims.
// In a sense `mo.transfer` is similar to `mo.rebind`, because they both need to
// (and are able to) handle remaining `?` in the graph, which might come from
// non-MO op or list ops. They can easily handle this because their output has
// the same shape as the input.
// CHECK-LABEL: mgp.model @transfer_with_unknown_dims : mof
//       CHECK: } execute ([[CH:%.*]]: !mef.chain, [[CTX:%.*]]: !mgp.context, {{.*}}, [[INPUT:.*]]: !mgp.tensor<?xsi64>) -> (!mef.chain, !mgp.tensor<?xsi64, <"cpu", 1>>) {
//  CHECK-NEXT:   [[IN_BUF:%.*]] = mgp.tensor.extract.buffer [[CTX]][0], [[INPUT]] : !mgp.tensor<?xsi64>
//  CHECK-NEXT:   [[IN_SPEC:%.*]] = mgp.tensor.extract.tensor_spec [[CTX]][0], [[INPUT]] : !mgp.tensor<?xsi64>
//  CHECK-NEXT:   [[IN_SIZE:%.*]] = mgp.tensor_spec.size [[IN_SPEC]] : !mgp.tensor_spec<?xsi64>
//  CHECK-NEXT:   [[NEW_BUF:%.*]] = mgp.buffer.alloc.dynamic {{.*}}, [[CTX]][1], [[IN_SIZE]], align {{.*}} : !mgp.buffer<, <"cpu", 1>>
//  CHECK-NEXT:   [[OUT_CH:%.*]] = mgp.buffer.host_to_device [[CH]], [[CTX]][0, 1], [[IN_BUF]], [[NEW_BUF]] : !mgp.buffer<> -> !mgp.buffer<, <"cpu", 1>>
//  CHECK-NEXT:   [[SYNC:%.*]] = mgp.sync([[OUT_CH]]
//  CHECK-NEXT:   mgp.extend.lifetime [[SYNC]], [[IN_BUF]]
//  CHECK-NEXT:   [[OUTPUT:%.*]] = mgp.tensor.create [[OUT_CH]], [[CTX]][1], [[NEW_BUF]], [[IN_SPEC]] : !mgp.tensor<?xsi64, <"cpu", 1>>
//  CHECK-NEXT:   mef.output [[OUT_CH]], [[OUTPUT]]
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

// CHECK-LABEL: mgp.model @transfer_d2h : mof
//       CHECK: } execute ([[CH:%.*]]: !mef.chain, [[CTX:%.*]]: !mgp.context, {{.*}}
// CHECK: [[CPY_CH:%.*]] = mgp.buffer.device_to_host {{.*}}, [[DBUF:%.*]], [[HBUF:%.*]] : !mgp.buffer
// CHECK-NEXT: [[SYNC_CH:%.*]] = mgp.sync([[CPY_CH]]
// CHECK-NEXT: mgp.extend.lifetime [[SYNC_CH]], [[DBUF]], [[HBUF]]
// CHECK: mef.output [[SYNC_CH]]

mo.graph @transfer_d2h(
  %input: !mo.tensor<[5], si64, {device = #M.device_ref<"cuda", 0>}>
) -> !mo.tensor<[5], si64, {device = #M.device_ref<"cpu", 0>}> attributes {
  device_specs = #M.device_spec_collection<host = <"cpu", 0>, devices = [
    <ref = <"cpu", 0>, target = <triple = "", arch = "", features = "">>,
    <ref = <"cuda", 0>, target = <triple = "", arch = "", features = "">>
  ]>} {
  %output = mo.transfer %input : !mo.tensor<[5], si64, {device = #M.device_ref<"cuda", 0>}> to <"cpu", 0>
  mo.output %output : !mo.tensor<[5], si64, {device = #M.device_ref<"cpu", 0>}>
}
