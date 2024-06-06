// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @equal_dyn(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.equal(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @equal_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[SH3]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = mo.equal([[V0]], [[V2]]) : (!mo.tensor<[[SH4]], f32>, !mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[SH4]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[SH4]], bool>) -> !mo.tensor<[[SH6]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], bool>

// -----

mo.graph @equal_dyn_params(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<[?, ?], f32>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.equal(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<[?, ?], f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @equal_dyn_params<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]] -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D2]], [[D3]]], f32>) -> !mo.tensor<[[SH4]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D2]], [[D3]]], f32> -> !mo.tensor<[[[D6]], [[D7]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[[D6]], [[D7]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = mo.equal([[V2]], [[V1]]) : (!mo.tensor<[[[D6]], [[D7]]], f32>, !mo.tensor<[[[D6]], [[D7]]], f32>) -> !mo.tensor<[[[D6]], [[D7]]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D6]], [[D7]]], bool>) -> !mo.tensor<[[SH8]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V4]] : !mo.tensor<[[SH8]], bool>

// -----

mo.graph @equal_params(%arg0: !mo.tensor<[?, ?], f32>, %arg1: !mo.tensor<[?, ?], f32>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.equal(%arg0, %arg1) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @equal_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]] -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D3]], [[D4]]], f32>) -> !mo.tensor<[[SH5]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D6]], [[D7]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D3]], [[D4]]], f32> -> !mo.tensor<[[[D8]], [[D9]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D8]], [[D9]]], f32>) -> !mo.tensor<[[[D6]], [[D7]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = mo.equal([[V0]], [[V2]]) : (!mo.tensor<[[[D6]], [[D7]]], f32>, !mo.tensor<[[[D6]], [[D7]]], f32>) -> !mo.tensor<[[[D6]], [[D7]]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D6]], [[D7]]], bool>) -> !mo.tensor<[[SH10]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH10]]> [[V4]] : !mo.tensor<[[SH10]], bool>

// -----

mo.graph @equal_mix(%arg0: !mo.tensor<[?, 2, ?], f32>, %arg1: !mo.tensor<[?, ?, 3], f32>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.equal(%arg0, %arg1) : (!mo.tensor<[?, 2, ?], f32>, !mo.tensor<[?, ?, 3], f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @equal_mix<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]] -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], 2, [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D3]], [[D4]], 3], f32>) -> !mo.tensor<[[SH5]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], 2, [[D2]]], f32> -> !mo.tensor<[[[D6]], 2, [[D7]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D3]], [[D4]], 3], f32> -> !mo.tensor<[[[D8]], [[D9]], 3], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[[D6]], 2, [[D7]]], f32>) -> !mo.tensor<[[[D6]], 2, 3], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D8]], [[D9]], 3], f32>) -> !mo.tensor<[[[D6]], 2, 3], f32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.equal([[V2]], [[V3]]) : (!mo.tensor<[[[D6]], 2, 3], f32>, !mo.tensor<[[[D6]], 2, 3], f32>) -> !mo.tensor<[[[D6]], 2, 3], bool>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[D6]], 2, 3], bool>) -> !mo.tensor<[[SH10]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH10]]> [[V5]] : !mo.tensor<[[SH10]], bool>

// -----

mo.graph @equal_static(%arg0: !mo.tensor<[1, 2, 3], f32>, %arg1: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.equal(%arg0, %arg1) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @equal_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[1, 2, 3], f32>, [[ARG1:%.*]]: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH1]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.equal([[ARG0]], [[ARG1]]) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], bool>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[1, 2, 3], bool>) -> !mo.tensor<[[SH2]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], bool>

// -----

mo.graph @greater_dyn(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.greater(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @greater_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[SH3]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = mo.greater([[V0]], [[V2]]) : (!mo.tensor<[[SH4]], f32>, !mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[SH4]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[SH4]], bool>) -> !mo.tensor<[[SH6]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], bool>

// -----

mo.graph @greater_equal_dyn(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.greater_equal(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @greater_equal_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[SH3]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = mo.greater_equal([[V0]], [[V2]]) : (!mo.tensor<[[SH4]], f32>, !mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[SH4]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[SH4]], bool>) -> !mo.tensor<[[SH6]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], bool>

// -----

mo.graph @not_equal_dyn(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.not_equal(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @not_equal_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[SH3]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = mo.not_equal([[V0]], [[V2]]) : (!mo.tensor<[[SH4]], f32>, !mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[SH4]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[SH4]], bool>) -> !mo.tensor<[[SH6]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], bool>
