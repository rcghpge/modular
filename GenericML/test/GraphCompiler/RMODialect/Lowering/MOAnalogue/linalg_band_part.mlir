// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @band_part_dyn(%arg0: !mo.tensor<?, f32>, %num_lower: !mo.tensor<?, si64>, %num_upper: !mo.tensor<?, si64>, %exclude: !mo.tensor<?, bool>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.linalg.band_part(%arg0, %num_lower, %num_upper, %exclude) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, bool>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @band_part_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>, [[ARG3:%.*]]: !mo.tensor<[[SH4]], bool>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH7]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH8]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG3]]) : !mo.tensor<[[SH4]], bool> -> !mo.tensor<[[SH9]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH7]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH8]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH9]], bool>) -> !mo.tensor<[], bool>
//  CHECK-NEXT:   [[V7:%.*]] = mo.linalg.band_part([[V0]], [[V4]], [[V5]], [[V6]]) : (!mo.tensor<[[SH6]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V8:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V7]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[SH10]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH10]]> [[V8]] : !mo.tensor<[[SH10]], f32>

// -----

mo.graph @band_part_dyn_params(%arg0: !mo.tensor<[?, ?], f32>, %num_lower: !mo.tensor<?, si64>, %num_upper: !mo.tensor<[], si64>, %exclude: !mo.tensor<?, bool>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.linalg.band_part(%arg0, %num_lower, %num_upper, %exclude) : (!mo.tensor<[?, ?], f32>, !mo.tensor<?, si64>, !mo.tensor<[], si64>, !mo.tensor<?, bool>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @band_part_dyn_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH3]], si64>, [[ARG2:%.*]]: !mo.tensor<[], si64>, [[ARG3:%.*]]: !mo.tensor<[[SH4]], bool>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D6]], [[D7]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH8]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG3]]) : !mo.tensor<[[SH4]], bool> -> !mo.tensor<[[SH9]], bool>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH8]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH9]], bool>) -> !mo.tensor<[], bool>
//  CHECK-NEXT:   [[V5:%.*]] = mo.linalg.band_part([[V0]], [[V3]], [[ARG2]], [[V4]]) : (!mo.tensor<[[[D6]], [[D7]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> !mo.tensor<[[[D6]], [[D7]]], f32>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D6]], [[D7]]], f32>) -> !mo.tensor<[[SH10]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH10]]> [[V6]] : !mo.tensor<[[SH10]], f32>

// -----

mo.graph @band_part_mix_params(%arg0: !mo.tensor<[?, 3], f32>, %num_lower: !mo.tensor<[], si64>, %num_upper: !mo.tensor<[], si64>, %exclude: !mo.tensor<?, bool>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.linalg.band_part(%arg0, %num_lower, %num_upper, %exclude) : (!mo.tensor<[?, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<?, bool>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @band_part_mix_params<[[D1:[A-Za-z0-9_]*]], [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], 3], f32>, [[ARG1:%.*]]: !mo.tensor<[], si64>, [[ARG2:%.*]]: !mo.tensor<[], si64>, [[ARG3:%.*]]: !mo.tensor<[[SH2]], bool>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D4:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], 3], f32> -> !mo.tensor<[[[D4]], 3], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG3]]) : !mo.tensor<[[SH2]], bool> -> !mo.tensor<[[SH5]], bool>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], bool>) -> !mo.tensor<[], bool>
//  CHECK-NEXT:   [[V3:%.*]] = mo.linalg.band_part([[V0]], [[ARG1]], [[ARG2]], [[V2]]) : (!mo.tensor<[[[D4]], 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> !mo.tensor<[[[D4]], 3], f32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D4]], 3], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @band_part_static(%arg0: !mo.tensor<[7, 3], f32>, %num_lower: !mo.tensor<[], si64>, %num_upper: !mo.tensor<[], si64>, %exclude: !mo.tensor<[], bool>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.linalg.band_part(%arg0, %num_lower, %num_upper, %exclude) : (!mo.tensor<[7, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @band_part_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[7, 3], f32>, [[ARG1:%.*]]: !mo.tensor<[], si64>, [[ARG2:%.*]]: !mo.tensor<[], si64>, [[ARG3:%.*]]: !mo.tensor<[], bool>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.linalg.band_part([[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]]) : (!mo.tensor<[7, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> !mo.tensor<[7, 3], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[7, 3], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>
