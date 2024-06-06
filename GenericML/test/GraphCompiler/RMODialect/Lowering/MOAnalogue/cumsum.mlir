// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @cumsum_dyn(%input: !mo.tensor<?, f32>, %axis: !mo.tensor<?, si64>) -> (!mo.tensor<?, f32>) {
  %0 = rmo.mo.cumsum(%input, %axis) {reverse = 0 : i64, exclusive = 0 : i64} : (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @cumsum_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH5]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.cumsum([[V0]], [[V2]]) {exclusive = 0 : i64, reverse = 0 : i64} : (!mo.tensor<[[SH4]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @cumsum_dyn_param_out(%input: !mo.tensor<?, f32>, %axis: !mo.tensor<?, si64>) -> (!mo.tensor<[?, ?], f32>) {
  %0 = rmo.mo.cumsum(%input, %axis) {reverse = 0 : i64, exclusive = 0 : i64} : (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @cumsum_dyn_param_out<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[[D3]], [[D4]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH6]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH6]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.cumsum([[V2]], [[V3]]) {exclusive = 0 : i64, reverse = 0 : i64} : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[D7:[A-Za-z0-9_]*]], [[D8:[A-Za-z0-9_]*]]>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>) -> !mo.tensor<[[[D7]], [[D8]]], f32>
//  CHECK-NEXT:   mo.output<[[D7]], [[D8]]> [[V5]] : !mo.tensor<[[[D7]], [[D8]]], f32>

// -----

mo.graph @cumsum_dyn_param(%input: !mo.tensor<[?, ?], f32>, %axis: !mo.tensor<[], si64>) -> (!mo.tensor<?, f32>) {
  %0 = rmo.mo.cumsum(%input, %axis) {reverse = 0 : i64, exclusive = 0 : i64} : (!mo.tensor<[?, ?], f32>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @cumsum_dyn_param<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]] -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D4]], [[D5]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.cumsum([[V0]], [[ARG1]]) {exclusive = 0 : i64, reverse = 0 : i64} : (!mo.tensor<[[[D4]], [[D5]]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D4]], [[D5]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[D4]], [[D5]]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V2]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @cumsum_static(%input: !mo.tensor<[3, 5], f32>, %axis: !mo.tensor<[], si64>) -> (!mo.tensor<?, f32>) {
  %0 = rmo.mo.cumsum(%input, %axis) {reverse = 0 : i64, exclusive = 0 : i64} : (!mo.tensor<[3, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @cumsum_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[3, 5], f32>, [[ARG1:%.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.cumsum([[ARG0]], [[ARG1]]) {exclusive = 0 : i64, reverse = 0 : i64} : (!mo.tensor<[3, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 5], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[3, 5], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>
