// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @is_nan_dyn(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.is_nan(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @is_nan_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH2]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.is_nan([[V0]]) : (!mo.tensor<[[SH3]], f32>) -> !mo.tensor<[[SH3]], bool>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH3]], bool>) -> !mo.tensor<[[SH4]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V2]] : !mo.tensor<[[SH4]], bool>

// -----

mo.graph @is_nan_params(%arg0: !mo.tensor<[?], f32>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.is_nan(%arg0) : (!mo.tensor<[?], f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @is_nan_params<[[D1:[A-Za-z0-9_]*]] -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]]], f32>) -> !mo.tensor<[[SH2]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D3:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]]], f32> -> !mo.tensor<[[[D3]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.is_nan([[V0]]) : (!mo.tensor<[[[D3]]], f32>) -> !mo.tensor<[[[D3]]], bool>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[D3]]], bool>) -> !mo.tensor<[[SH4]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V2]] : !mo.tensor<[[SH4]], bool>

// -----

mo.graph @is_nan_static(%arg0: !mo.tensor<[2, 6], f32>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.is_nan(%arg0) : (!mo.tensor<[2, 6], f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @is_nan_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, 6], f32>) -> !mo.tensor<[[SH1]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.is_nan([[ARG0]]) : (!mo.tensor<[2, 6], f32>) -> !mo.tensor<[2, 6], bool>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[2, 6], bool>) -> !mo.tensor<[[SH2]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], bool>

// -----

mo.graph @is_inf_dyn(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.is_inf(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @is_inf_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH2]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.is_inf([[V0]]) : (!mo.tensor<[[SH3]], f32>) -> !mo.tensor<[[SH3]], bool>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH3]], bool>) -> !mo.tensor<[[SH4]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V2]] : !mo.tensor<[[SH4]], bool>
