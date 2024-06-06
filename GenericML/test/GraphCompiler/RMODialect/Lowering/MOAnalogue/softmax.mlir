// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @softmax_dyn(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.softmax(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @softmax_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH2]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.softmax([[V0]]) : (!mo.tensor<[[SH3]], f32>) -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH3]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V2]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @softmax_params(%arg0: !mo.tensor<[?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.softmax(%arg0) : (!mo.tensor<[?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @softmax_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]] -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]]], f32>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D4]], [[D5]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.softmax([[V0]]) : (!mo.tensor<[[[D4]], [[D5]]], f32>) -> !mo.tensor<[[[D4]], [[D5]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[D4]], [[D5]]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V2]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @softmax_params_mix(%arg0: !mo.tensor<[3, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.softmax(%arg0) : (!mo.tensor<[3, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @softmax_params_mix<[[D1:[A-Za-z0-9_]*]] -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[3, [[D1]]], f32>) -> !mo.tensor<[[SH2]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D3:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[3, [[D1]]], f32> -> !mo.tensor<[3, [[D3]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.softmax([[V0]]) : (!mo.tensor<[3, [[D3]]], f32>) -> !mo.tensor<[3, [[D3]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[3, [[D3]]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V2]] : !mo.tensor<[[SH4]], f32>


// -----

mo.graph @softmax_static(%arg0: !mo.tensor<[3, 7], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.softmax(%arg0) : (!mo.tensor<[3, 7], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @softmax_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[3, 7], f32>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.softmax([[ARG0]]) : (!mo.tensor<[3, 7], f32>) -> !mo.tensor<[3, 7], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[3, 7], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>
