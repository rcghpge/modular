// RUN: mo-opt %s --mo-to-primitives | FileCheck %s


// This test makes sure we can properly lower `mo.transfer` with unknown dims.
// In a sense `mo.transfer` is similar to `mo.rebind`, because they both need to
// (and are able to) handle remaining `?` in the graph, which might come from
// non-MO op or list ops. They can easily handle this because their output has
// the same shape as the input.
// CHECK-LABEL: mgp.model @transfer_with_unknown_dims : mof
//       CHECK: } execute ([[CH:%.*]]: !mef.chain, [[CTX:%.*]]: !mgp.context, {{.*}}, [[INPUT:.*]]: !mgp.tensor<?xsi64>) -> (!mef.chain, !mgp.tensor<?xsi64, <"cpu", 1>>) {
//  CHECK-NEXT:   [[CH0:%.*]] = grt.chain.create
//  CHECK-NEXT:   [[IN_BUF:%.*]] = mgp.tensor.extract.buffer [[INPUT]] : <?xsi64>
//  CHECK-NEXT:   [[IN_SPEC:%.*]] = mgp.tensor.extract.tensor_spec [[INPUT]] : <?xsi64>
//  CHECK-NEXT:   [[IDX8:%.*]] = index.constant 8
//  CHECK-NEXT:   [[DIM0:%.*]] = mgp.tensor_spec.get_dim[0] [[IN_SPEC]] : <?xsi64>
//  CHECK-NEXT:   [[IN_SIZE:%.*]] = index.mul [[DIM0]], [[IDX8]]
//  CHECK-NEXT:   [[NEW_BUF:%.*]] = mgp.buffer.alloc [[CH0]], [[IN_SIZE]], {{.*}}, align {{.*}} : !mgp.buffer<<"cpu", 1>>
//  CHECK-NEXT:   [[OUT_CH:%.*]] = mgp.buffer.host_to_device [[CH0]], [[IN_BUF]], [[NEW_BUF]], {{.*}} : !mgp.buffer -> !mgp.buffer<<"cpu", 1>>
//  CHECK-NEXT:   [[SYNC:%.*]] = mgp.sync([[OUT_CH]]
//  CHECK-NEXT:   mgp.extend.lifetime [[SYNC]], [[IN_BUF]]
//  CHECK-NEXT:   [[OUTPUT:%.*]] = mgp.tensor.create [[OUT_CH]], [[NEW_BUF]], [[IN_SPEC]] : !mgp.tensor<?xsi64, <"cpu", 1>>
//  CHECK-NEXT:   mef.output [[CH]], [[OUTPUT]]
mo.graph @transfer_with_unknown_dims(
  %input: !mo.tensor<[?], si64>
) -> !mo.tensor<[?], si64, cpu:1> attributes {
  // Lowering requires us to supply device spec for each device ref.
  device_specs = #M.device_spec_collection<host = <"cpu", 0>, devices = [
    <ref = <"cpu", 0>, target = <triple = "arm64-apple-darwin22.4.0", arch = "apple-m1", features = "">>,
    <ref = <"cpu", 1>, target = <triple = "arm64-apple-darwin22.4.0", arch = "apple-m1", features = "">>
  ]>} {
  %ch0 = mo.chain.create()
  %output, %ch1 = mo.transfer[%ch0] %input : !mo.tensor<[?], si64> to <"cpu", 1>
  mo.output %output : !mo.tensor<[?], si64, cpu:1>
}

// CHECK-LABEL: mgp.model @transfer_d2h : mof
//       CHECK: } execute ([[CH:%.*]]: !mef.chain, [[CTX:%.*]]: !mgp.context, {{.*}}
// CHECK: [[CPY_CH:%.*]] = mgp.buffer.device_to_host {{.*}}, [[DBUF:%.*]], [[HBUF:%.*]], {{.*}} : !mgp.buffer
// CHECK-NEXT: [[SYNC_CH:%.*]] = mgp.sync([[CPY_CH]]
// CHECK-NEXT: mgp.extend.lifetime [[SYNC_CH]], [[HBUF]]

// COM: Device to host needs to wait on the sync before using the host buffer.

// CHECK-NEXT: [[OUT_TENSOR:%.*]] = mgp.tensor.create [[SYNC_CH]]
// CHECK: mef.output [[CH]], [[OUT_TENSOR]]

mo.graph @transfer_d2h(
  %input: !mo.tensor<[5], si64, gpu:0>
) -> !mo.tensor<[5], si64> attributes {
  device_specs = #M.device_spec_collection<host = <"cpu", 0>, devices = [
    <ref = <"cpu", 0>, target = <triple = "", arch = "", features = "">>,
    <ref = <"gpu", 0>, target = <triple = "", arch = "", features = "">>
  ]>} {
  %ch0 = mo.chain.create()
  %output, %ch1 = mo.transfer[%ch0] %input : !mo.tensor<[5], si64, gpu:0> to <"cpu", 0>
  mo.output %output : !mo.tensor<[5], si64>
}
