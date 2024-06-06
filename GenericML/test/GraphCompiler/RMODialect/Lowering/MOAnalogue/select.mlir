// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @select_dyn(%arg0: !mo.tensor<?, bool>, %arg1: !mo.tensor<?, f32>, %arg2: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.select(%arg0, %arg1, %arg2) : (!mo.tensor<?, bool>, !mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @select_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], bool>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], f32>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], bool> -> !mo.tensor<[[SH5]], bool>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], f32> -> !mo.tensor<[[SH7]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[SH5]], bool>) -> !mo.tensor<[[SH6]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH7]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = mo.select([[V3]], [[V1]], [[V4]]) : (!mo.tensor<[[SH6]], bool>, !mo.tensor<[[SH6]], f32>, !mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V6]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @select_dyn_params(%arg0: !mo.tensor<[?, ?], bool>, %arg1: !mo.tensor<?, f32>, %arg2: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.select(%arg0, %arg1, %arg2) : (!mo.tensor<[?, ?], bool>, !mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @select_dyn_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]]], bool>, [[ARG1:%.*]]: !mo.tensor<[[SH3]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]]], bool> -> !mo.tensor<[[[D6]], [[D7]]], bool>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH3]], f32> -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH4]], f32> -> !mo.tensor<[[SH9]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH8]], f32>) -> !mo.tensor<[[[D6]], [[D7]]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH9]], f32>) -> !mo.tensor<[[[D6]], [[D7]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = mo.select([[V0]], [[V3]], [[V4]]) : (!mo.tensor<[[[D6]], [[D7]]], bool>, !mo.tensor<[[[D6]], [[D7]]], f32>, !mo.tensor<[[[D6]], [[D7]]], f32>) -> !mo.tensor<[[[D6]], [[D7]]], f32>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D6]], [[D7]]], f32>) -> !mo.tensor<[[SH10]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH10]]> [[V6]] : !mo.tensor<[[SH10]], f32>

// -----

mo.graph @select_params(%arg0: !mo.tensor<[?, ?], bool>, %arg1: !mo.tensor<[?, ?], f32>, %arg2: !mo.tensor<[?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.select(%arg0, %arg1, %arg2) : (!mo.tensor<[?, ?], bool>, !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @select_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]], [[D6:[A-Za-z0-9_]*]] -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]]], bool>, [[ARG1:%.*]]: !mo.tensor<[[[D3]], [[D4]]], f32>, [[ARG2:%.*]]: !mo.tensor<[[[D5]], [[D6]]], f32>) -> !mo.tensor<[[SH7]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]]], bool> -> !mo.tensor<[[[D8]], [[D9]]], bool>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D3]], [[D4]]], f32> -> !mo.tensor<[[[D10]], [[D11]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[D12:[A-Za-z0-9_]*]], [[D13:[A-Za-z0-9_]*]]>([[ARG2]]) : !mo.tensor<[[[D5]], [[D6]]], f32> -> !mo.tensor<[[[D12]], [[D13]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[[D8]], [[D9]]], bool>) -> !mo.tensor<[[[D10]], [[D11]]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[[D12]], [[D13]]], f32>) -> !mo.tensor<[[[D10]], [[D11]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = mo.select([[V3]], [[V1]], [[V4]]) : (!mo.tensor<[[[D10]], [[D11]]], bool>, !mo.tensor<[[[D10]], [[D11]]], f32>, !mo.tensor<[[[D10]], [[D11]]], f32>) -> !mo.tensor<[[[D10]], [[D11]]], f32>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape<() -> [[SH14:[A-Za-z0-9_]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D10]], [[D11]]], f32>) -> !mo.tensor<[[SH14]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH14]]> [[V6]] : !mo.tensor<[[SH14]], f32>

// -----

mo.graph @select_mix(%arg0: !mo.tensor<[2, ?, ?], bool>, %arg1: !mo.tensor<[2, ?, 5], f32>, %arg2: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.select(%arg0, %arg1, %arg2) : (!mo.tensor<[2, ?, ?], bool>, !mo.tensor<[2, ?, 5], f32>, !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @select_mix<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]] -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, [[D1]], [[D2]]], bool>, [[ARG1:%.*]]: !mo.tensor<[2, [[D3]], 5], f32>, [[ARG2:%.*]]: !mo.tensor<[[[D4]], [[D5]], 5], f32>) -> !mo.tensor<[[SH6]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D7:[A-Za-z0-9_]*]], [[D8:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[2, [[D1]], [[D2]]], bool> -> !mo.tensor<[2, [[D7]], [[D8]]], bool>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D9:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[2, [[D3]], 5], f32> -> !mo.tensor<[2, [[D9]], 5], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[ARG2]]) : !mo.tensor<[[[D4]], [[D5]], 5], f32> -> !mo.tensor<[[[D10]], [[D11]], 5], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[2, [[D7]], [[D8]]], bool>) -> !mo.tensor<[2, [[D9]], 5], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[[D10]], [[D11]], 5], f32>) -> !mo.tensor<[2, [[D9]], 5], f32>
//  CHECK-NEXT:   [[V5:%.*]] = mo.select([[V3]], [[V1]], [[V4]]) : (!mo.tensor<[2, [[D9]], 5], bool>, !mo.tensor<[2, [[D9]], 5], f32>, !mo.tensor<[2, [[D9]], 5], f32>) -> !mo.tensor<[2, [[D9]], 5], f32>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape<() -> [[SH12:[A-Za-z0-9_]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[2, [[D9]], 5], f32>) -> !mo.tensor<[[SH12]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH12]]> [[V6]] : !mo.tensor<[[SH12]], f32>

// -----

mo.graph @select_static(%arg0: !mo.tensor<[2, 8, 5], bool>, %arg1: !mo.tensor<[2, 8, 5], f32>, %arg2: !mo.tensor<[2, 8, 5], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.select(%arg0, %arg1, %arg2) : (!mo.tensor<[2, 8, 5], bool>, !mo.tensor<[2, 8, 5], f32>, !mo.tensor<[2, 8, 5], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @select_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, 8, 5], bool>, [[ARG1:%.*]]: !mo.tensor<[2, 8, 5], f32>, [[ARG2:%.*]]: !mo.tensor<[2, 8, 5], f32>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.select([[ARG0]], [[ARG1]], [[ARG2]]) : (!mo.tensor<[2, 8, 5], bool>, !mo.tensor<[2, 8, 5], f32>, !mo.tensor<[2, 8, 5], f32>) -> !mo.tensor<[2, 8, 5], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[2, 8, 5], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>
