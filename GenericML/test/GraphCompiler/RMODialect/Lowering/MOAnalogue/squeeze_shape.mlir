// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s

mo.graph @squeeze_shape_dyn_dyn(%shape: !mo.tensor<?, si64>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, si64> {
  %1 = rmo.mo.squeeze_shape(%shape, %indices) : (!mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<?, si64>
  mo.output %1: !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @squeeze_shape_dyn_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], si64>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH3]], si64> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH4]], si64>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH5]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH4]], si64>) -> !mo.tensor<[[[RMO_D7]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D8:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], si64>) -> !mo.tensor<[[[RMO_D8]]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.squeeze_shape<() -> [[RMO_D9:[A-Za-z0-9_]*]]>([[V2]], [[V3]]) : (!mo.tensor<[[[RMO_D7]]], si64>, !mo.tensor<[[[RMO_D8]]], si64>) -> !mo.tensor<[[[RMO_D9]]], si64>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D9]]], si64>) -> !mo.tensor<[[SH6]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V5]] : !mo.tensor<[[SH6]], si64>

// -----

mo.graph @squeeze_shape_dyn_params(%shape: !mo.tensor<?, si64>, %indices: !mo.tensor<[?], si64>) -> !mo.tensor<?, si64> {
  %1 = rmo.mo.squeeze_shape(%shape, %indices) : (!mo.tensor<?, si64>, !mo.tensor<[?], si64>) -> !mo.tensor<?, si64>
  mo.output %1: !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @squeeze_shape_dyn_params<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[D2:[A-Za-z0-9_]*]] -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], si64>, [[ARG1:%.*]]: !mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[[SH3]], si64> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], si64> -> !mo.tensor<[[SH4]], si64>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D5:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D2]]], si64> -> !mo.tensor<[[[D5]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH4]], si64>) -> !mo.tensor<[[[RMO_D7]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D8:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[D5]]], si64>) -> !mo.tensor<[[[RMO_D8]]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.squeeze_shape<() -> [[RMO_D9:[A-Za-z0-9_]*]]>([[V2]], [[V3]]) : (!mo.tensor<[[[RMO_D7]]], si64>, !mo.tensor<[[[RMO_D8]]], si64>) -> !mo.tensor<[[[RMO_D9]]], si64>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D9]]], si64>) -> !mo.tensor<[[SH6]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V5]] : !mo.tensor<[[SH6]], si64>

// -----

mo.graph @squeeze_shape_params_params(%shape: !mo.tensor<[?], si64>, %indices: !mo.tensor<[?], si64>) -> !mo.tensor<?, si64> {
  %1 = rmo.mo.squeeze_shape(%shape, %indices) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<?, si64>
  mo.output %1: !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @squeeze_shape_params_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]] -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]]], si64>, [[ARG1:%.*]]: !mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[[SH3]], si64> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D4:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]]], si64> -> !mo.tensor<[[[D4]]], si64>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D5:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D2]]], si64> -> !mo.tensor<[[[D5]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[D4]]], si64>) -> !mo.tensor<[[[RMO_D7]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D8:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[D5]]], si64>) -> !mo.tensor<[[[RMO_D8]]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.squeeze_shape<() -> [[RMO_D9:[A-Za-z0-9_]*]]>([[V2]], [[V3]]) : (!mo.tensor<[[[RMO_D7]]], si64>, !mo.tensor<[[[RMO_D8]]], si64>) -> !mo.tensor<[[[RMO_D9]]], si64>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D9]]], si64>) -> !mo.tensor<[[SH6]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V5]] : !mo.tensor<[[SH6]], si64>

// -----

mo.graph @squeeze_shape_static_params(%shape: !mo.tensor<[2], si64>, %indices: !mo.tensor<[?], si64>) -> !mo.tensor<?, si64> {
  %1 = rmo.mo.squeeze_shape(%shape, %indices) : (!mo.tensor<[2], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<?, si64>
  mo.output %1: !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @squeeze_shape_static_params<[[D1:[A-Za-z0-9_]*]] -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2], si64>, [[ARG1:%.*]]: !mo.tensor<[[[D1]]], si64>) -> !mo.tensor<[[SH2]], si64> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D3:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D1]]], si64> -> !mo.tensor<[[[D3]]], si64>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D5]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D6:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[D3]]], si64>) -> !mo.tensor<[[[RMO_D6]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.squeeze_shape<() -> [[RMO_D7:[A-Za-z0-9_]*]]>([[V1]], [[V2]]) : (!mo.tensor<[[[RMO_D5]]], si64>, !mo.tensor<[[[RMO_D6]]], si64>) -> !mo.tensor<[[[RMO_D7]]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[RMO_D7]]], si64>) -> !mo.tensor<[[SH4]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V4]] : !mo.tensor<[[SH4]], si64>

// -----

mo.graph @squeeze_shape_static_static(%shape: !mo.tensor<[4], si64>, %indices: !mo.tensor<[2], si64>) -> !mo.tensor<?, si64> {
  %1 = rmo.mo.squeeze_shape(%shape, %indices) : (!mo.tensor<[4], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<?, si64>
  mo.output %1: !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @squeeze_shape_static_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[4], si64>, [[ARG1:%.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[[SH1]], si64> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[4], si64>) -> !mo.tensor<[[[RMO_D3]]], si64>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D4]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.squeeze_shape<() -> [[RMO_D5:[A-Za-z0-9_]*]]>([[V0]], [[V1]]) : (!mo.tensor<[[[RMO_D3]]], si64>, !mo.tensor<[[[RMO_D4]]], si64>) -> !mo.tensor<[[[RMO_D5]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D5]]], si64>) -> !mo.tensor<[[SH2]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V3]] : !mo.tensor<[[SH2]], si64>
