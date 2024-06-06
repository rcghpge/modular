// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @random_normal_dyn(%size: !mo.tensor<?, si64>, %mean: !mo.tensor<?, f32>, %variance: !mo.tensor<?, f32>, %seed: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.random.normal(%size, %mean, %variance, %seed) : (!mo.tensor<?, si64>, !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @random_normal_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], si64>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], f32>, [[ARG3:%.*]]: !mo.tensor<[[SH4]], si64>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH6]], si64>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH7]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], f32> -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG3]]) : !mo.tensor<[[SH4]], si64> -> !mo.tensor<[[SH9]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D11:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH6]], si64>) -> !mo.tensor<[[[RMO_D11]]], si64>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH7]], f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH8]], f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH9]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V8:%.*]] = mo.random.normal<() -> [[RMO_SH12:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]], [[V5]], [[V6]], [[V7]]) : (!mo.tensor<[[[RMO_D11]]], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[RMO_SH12]], f32>
//  CHECK-NEXT:   [[V9:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V8]]) {{.*}} : (!mo.tensor<[[RMO_SH12]], f32>) -> !mo.tensor<[[SH10]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH10]]> [[V9]] : !mo.tensor<[[SH10]], f32>

// -----

mo.graph @random_normal_dyn_out_param(%size: !mo.tensor<?, si64>, %mean: !mo.tensor<?, f32>, %variance: !mo.tensor<?, f32>, %seed: !mo.tensor<?, si64>) -> !mo.tensor<[?, ?], f32> {
  %0 = rmo.mo.random.normal(%size, %mean, %variance, %seed) : (!mo.tensor<?, si64>, !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @random_normal_dyn_out_param<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape -> [[D5:[A-Za-z0-9_]*]], [[D6:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[[SH1]], si64>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], f32>, [[ARG3:%.*]]: !mo.tensor<[[SH4]], si64>) -> !mo.tensor<[[[D5]], [[D6]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH7]], si64>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], f32> -> !mo.tensor<[[SH9]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG3]]) : !mo.tensor<[[SH4]], si64> -> !mo.tensor<[[SH10]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[SH7]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH8]], f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH9]], f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH10]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V8:%.*]] = mo.random.normal<() -> [[RMO_D13:[A-Za-z0-9_]*]], [[RMO_D14:[A-Za-z0-9_]*]]>([[V4]], [[V5]], [[V6]], [[V7]]) : (!mo.tensor<[2], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>
//  CHECK-NEXT:   [[V9:%.*]] = rmo.rebind_tensor_shape<() -> [[D11:[A-Za-z0-9_]*]], [[D12:[A-Za-z0-9_]*]]>([[V8]]) {{.*}} : (!mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>) -> !mo.tensor<[[[D11]], [[D12]]], f32>
//  CHECK-NEXT:   mo.output<[[D11]], [[D12]]> [[V9]] : !mo.tensor<[[[D11]], [[D12]]], f32>

// -----

mo.graph @random_normal_params(%size: !mo.tensor<[?], si64>, %mean: !mo.tensor<[], f32>, %variance: !mo.tensor<[], f32>, %seed: !mo.tensor<[], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.random.normal(%size, %mean, %variance, %seed) : (!mo.tensor<[?], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @random_normal_params<[[D1:[A-Za-z0-9_]*]] -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]]], si64>, [[ARG1:%.*]]: !mo.tensor<[], f32>, [[ARG2:%.*]]: !mo.tensor<[], f32>, [[ARG3:%.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[[SH2]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D3:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]]], si64> -> !mo.tensor<[[[D3]]], si64>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D5:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[D3]]], si64>) -> !mo.tensor<[[[RMO_D5]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.random.normal<() -> [[RMO_SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]], [[ARG1]], [[ARG2]], [[ARG3]]) : (!mo.tensor<[[[RMO_D5]]], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[RMO_SH6]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[RMO_SH6]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V3]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @random_normal_static(%size: !mo.tensor<[2], si64>, %mean: !mo.tensor<[], f32>, %variance: !mo.tensor<[], f32>, %seed: !mo.tensor<[], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.random.normal(%size, %mean, %variance, %seed) : (!mo.tensor<[2], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @random_normal_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2], si64>, [[ARG1:%.*]]: !mo.tensor<[], f32>, [[ARG2:%.*]]: !mo.tensor<[], f32>, [[ARG3:%.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.random.normal<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]]) : (!mo.tensor<[2], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>
