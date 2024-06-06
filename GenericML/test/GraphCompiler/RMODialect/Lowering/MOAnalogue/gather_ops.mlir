// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @gather_dyn(%input: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si32>, %axis: !mo.tensor<?, si32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.gather(%input, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<?, si32>, !mo.tensor<?, si32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @gather_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si32>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si32>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si32> -> !mo.tensor<[[SH6]], si32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si32> -> !mo.tensor<[[SH7]], si32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH7]], si32>) -> !mo.tensor<[], si32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.gather<() -> [[RMO_SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]], [[V1]], [[V3]]) : (!mo.tensor<[[SH5]], f32>, !mo.tensor<[[SH6]], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[[RMO_SH9]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[RMO_SH9]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V5]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @gather_dyn_dims(%input: !mo.tensor<[?, ?, ?], f32>, %indices: !mo.tensor<[?, ?], si32>, %axis: !mo.tensor<[], si32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.gather(%input, %indices, %axis) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @gather_dyn_dims<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]] -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D4]], [[D5]]], si32>, [[ARG2:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[SH6]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D7:[A-Za-z0-9_]*]], [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D4]], [[D5]]], si32> -> !mo.tensor<[[[D10]], [[D11]]], si32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.gather<() -> [[RMO_D13:[A-Za-z0-9_]*]], [[RMO_D14:[A-Za-z0-9_]*]], [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]]>([[V0]], [[V1]], [[ARG2]]) : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>, !mo.tensor<[[[D10]], [[D11]]], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D13]], [[RMO_D14]], [[RMO_D15]], [[RMO_D16]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH12:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D13]], [[RMO_D14]], [[RMO_D15]], [[RMO_D16]]], f32>) -> !mo.tensor<[[SH12]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH12]]> [[V3]] : !mo.tensor<[[SH12]], f32>

// -----

mo.graph @gather_mix_dyn_dims(%input: !mo.tensor<[8, ?, ?], f32>, %indices: !mo.tensor<[?, 3], si32>, %axis: !mo.tensor<[], si32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.gather(%input, %indices, %axis) : (!mo.tensor<[8, ?, ?], f32>, !mo.tensor<[?, 3], si32>, !mo.tensor<[], si32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @gather_mix_dyn_dims<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]] -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[8, [[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D3]], 3], si32>, [[ARG2:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D5:[A-Za-z0-9_]*]], [[D6:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[8, [[D1]], [[D2]]], f32> -> !mo.tensor<[8, [[D5]], [[D6]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D7:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D3]], 3], si32> -> !mo.tensor<[[[D7]], 3], si32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.gather<() -> [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]], [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]]>([[V0]], [[V1]], [[ARG2]]) : (!mo.tensor<[8, [[D5]], [[D6]]], f32>, !mo.tensor<[[[D7]], 3], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D9]], [[RMO_D10]], [[RMO_D11]], [[RMO_D12]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]], [[RMO_D11]], [[RMO_D12]]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V3]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @gather_static(%input: !mo.tensor<[12, 2], f32>, %indices: !mo.tensor<[128], si32>, %axis: !mo.tensor<[], si32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.gather(%input, %indices, %axis) : (!mo.tensor<[12, 2], f32>, !mo.tensor<[128], si32>, !mo.tensor<[], si32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @gather_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[12, 2], f32>, [[ARG1:%.*]]: !mo.tensor<[128], si32>, [[ARG2:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.gather<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]], [[ARG2]]) : (!mo.tensor<[12, 2], f32>, !mo.tensor<[128], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @gather_nd_dyn(%input: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.gather_nd(%input, %indices) {batchDims = 1} : (!mo.tensor<?, f32>, !mo.tensor<?, si32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @gather_nd_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si32>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si32> -> !mo.tensor<[[SH5]], si32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.gather_nd<() -> [[RMO_SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]], [[V1]]) {[[BATCHDIMS:[A-Za-z0-9_]*]] = 1 : i64} : (!mo.tensor<[[SH4]], f32>, !mo.tensor<[[SH5]], si32>) -> !mo.tensor<[[RMO_SH7]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[RMO_SH7]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V3]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @gather_nd_dyn_dims(%input: !mo.tensor<[?, ?], f32>, %indices: !mo.tensor<[?, ?], si32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.gather_nd(%input, %indices) {batchDims = 1} : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], si32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @gather_nd_dyn_dims<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]] -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D3]], [[D4]]], si32>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D6]], [[D7]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D3]], [[D4]]], si32> -> !mo.tensor<[[[D8]], [[D9]]], si32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.gather_nd<() -> [[RMO_SH11:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]], [[V1]]) {[[BATCHDIMS:[A-Za-z0-9_]*]] = 1 : i64} : (!mo.tensor<[[[D6]], [[D7]]], f32>, !mo.tensor<[[[D8]], [[D9]]], si32>) -> !mo.tensor<[[RMO_SH11]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[RMO_SH11]], f32>) -> !mo.tensor<[[SH10]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH10]]> [[V3]] : !mo.tensor<[[SH10]], f32>

// -----

mo.graph @gather_nd_static(%input: !mo.tensor<[2, 2, 4], f32>, %indices: !mo.tensor<[3], si32>) -> !mo.tensor<?, f32> {
   %0 = rmo.mo.gather_nd(%input, %indices) {batchDims = 1} : (!mo.tensor<[2, 2, 4], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<?, f32>
   mo.output %0 : !mo.tensor<?, f32>
}
// COM: output is rank 0
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @gather_nd_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, 2, 4], f32>, [[ARG1:%.*]]: !mo.tensor<[3], si32>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.gather_nd([[ARG0]], [[ARG1]]) {[[BATCHDIMS:[A-Za-z0-9_]*]] = 1 : i64} : (!mo.tensor<[2, 2, 4], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @gather_nd_static_2(%input: !mo.tensor<[2, 2, 4], f32>, %indices: !mo.tensor<[4, 3], si32>) -> !mo.tensor<?, f32> {
   %0 = rmo.mo.gather_nd(%input, %indices) {batchDims = 0} : (!mo.tensor<[2, 2, 4], f32>, !mo.tensor<[4, 3], si32>) -> !mo.tensor<?, f32>
   mo.output %0 : !mo.tensor<?, f32>
}
// COM: output is rank 1
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @gather_nd_static_2<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, 2, 4], f32>, [[ARG1:%.*]]: !mo.tensor<[4, 3], si32>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.gather_nd<() -> [[RMO_D3:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) {[[BATCHDIMS:[A-Za-z0-9_]*]] = 0 : i64} : (!mo.tensor<[2, 2, 4], f32>, !mo.tensor<[4, 3], si32>) -> !mo.tensor<[[[RMO_D3]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D3]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @gather_nd_static_3(%input: !mo.tensor<[2, 2, 1], f32>, %indices: !mo.tensor<[2, 1], si32>) -> !mo.tensor<?, f32> {
   %0 = rmo.mo.gather_nd(%input, %indices) {batchDims = 1} : (!mo.tensor<[2, 2, 1], f32>, !mo.tensor<[2, 1], si32>) -> !mo.tensor<?, f32>
   mo.output %0 : !mo.tensor<?, f32>
}
// COM: output is rank 2
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @gather_nd_static_3<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, 2, 1], f32>, [[ARG1:%.*]]: !mo.tensor<[2, 1], si32>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.gather_nd<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) {[[BATCHDIMS:[A-Za-z0-9_]*]] = 1 : i64} : (!mo.tensor<[2, 2, 1], f32>, !mo.tensor<[2, 1], si32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>
