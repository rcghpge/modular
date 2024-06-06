// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @argnonzero_dyn(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, si64> {
  %0 = rmo.mo.arg_nonzero(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, si64>
  mo.output %0 : !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @argnonzero_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH2]], si64> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.arg_nonzero<() -> [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]]>([[V0]]) : (!mo.tensor<[[SH3]], f32>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D5]], [[RMO_D6]]], si64>) -> !mo.tensor<[[SH4]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V2]] : !mo.tensor<[[SH4]], si64>



// -----

mo.graph @argnonzero_dyn_out_param(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<[?, 3], si64> {
  %0 = rmo.mo.arg_nonzero(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<[?, 3], si64>
  mo.output %0 : !mo.tensor<[?, 3], si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @argnonzero_dyn_out_param<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape -> [[D2:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[[D2]], 3], si64> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH3]], f32>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]], [[RMO_D7]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.arg_nonzero<() -> [[RMO_D8:[A-Za-z0-9_]*]]>([[V1]]) : (!mo.tensor<[[[RMO_D5]], [[RMO_D6]], [[RMO_D7]]], f32>) -> !mo.tensor<[[[RMO_D8]], 3], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[D4:[A-Za-z0-9_]*]]>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D8]], 3], si64>) -> !mo.tensor<[[[D4]], 3], si64>
//  CHECK-NEXT:   mo.output<[[D4]]> [[V3]] : !mo.tensor<[[[D4]], 3], si64>

// -----

mo.graph @argnonzero_param(%arg0: !mo.tensor<[?, ?], f32>) -> !mo.tensor<?, si64> {
  %0 = rmo.mo.arg_nonzero(%arg0) : (!mo.tensor<[?, ?], f32>) -> !mo.tensor<?, si64>
  mo.output %0 : !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @argnonzero_param<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]] -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]]], f32>) -> !mo.tensor<[[SH3]], si64> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D4]], [[D5]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[D4]], [[D5]]], f32>) -> !mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.arg_nonzero<() -> [[RMO_D9:[A-Za-z0-9_]*]]>([[V1]]) : (!mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f32>) -> !mo.tensor<[[[RMO_D9]], 2], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D9]], 2], si64>) -> !mo.tensor<[[SH6]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V3]] : !mo.tensor<[[SH6]], si64>

// -----

mo.graph @argnonzero_static(%arg0: !mo.tensor<[2, 7], f32>) -> !mo.tensor<?, si64> {
  %0 = rmo.mo.arg_nonzero(%arg0) : (!mo.tensor<[2, 7], f32>) -> !mo.tensor<?, si64>
  mo.output %0 : !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @argnonzero_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, 7], f32>) -> !mo.tensor<[[SH1]], si64> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[2, 7], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.arg_nonzero<() -> [[RMO_D5:[A-Za-z0-9_]*]]>([[V0]]) : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[[RMO_D5]], 2], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D5]], 2], si64>) -> !mo.tensor<[[SH2]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V2]] : !mo.tensor<[[SH2]], si64>
