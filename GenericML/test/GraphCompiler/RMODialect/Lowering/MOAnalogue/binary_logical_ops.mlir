// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @and_dyn(%arg0: !mo.tensor<?, bool>, %arg1: !mo.tensor<?, bool>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.and(%arg0, %arg1) : (!mo.tensor<?, bool>, !mo.tensor<?, bool>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @and_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], bool>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], bool>) -> !mo.tensor<[[SH3]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], bool> -> !mo.tensor<[[SH4]], bool>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], bool> -> !mo.tensor<[[SH5]], bool>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], bool>) -> !mo.tensor<[[SH4]], bool>
//  CHECK-NEXT:   [[V3:%.*]] = mo.and([[V0]], [[V2]]) : !mo.tensor<[[SH4]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[SH4]], bool>) -> !mo.tensor<[[SH6]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], bool>

// -----

mo.graph @and_dyn_params(%arg0: !mo.tensor<?, bool>, %arg1: !mo.tensor<[?, ?], bool>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.and(%arg0, %arg1) : (!mo.tensor<?, bool>, !mo.tensor<[?, ?], bool>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @and_dyn_params<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]] -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], bool>, [[ARG1:%.*]]: !mo.tensor<[[[D2]], [[D3]]], bool>) -> !mo.tensor<[[SH4]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], bool> -> !mo.tensor<[[SH5]], bool>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D2]], [[D3]]], bool> -> !mo.tensor<[[[D6]], [[D7]]], bool>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[SH5]], bool>) -> !mo.tensor<[[[D6]], [[D7]]], bool>
//  CHECK-NEXT:   [[V3:%.*]] = mo.and([[V2]], [[V1]]) : !mo.tensor<[[[D6]], [[D7]]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D6]], [[D7]]], bool>) -> !mo.tensor<[[SH8]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V4]] : !mo.tensor<[[SH8]], bool>

// -----

mo.graph @and_params_params(%arg0: !mo.tensor<[?, ?], bool>, %arg1: !mo.tensor<[?, ?], bool>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.and(%arg0, %arg1) : (!mo.tensor<[?, ?], bool>, !mo.tensor<[?, ?], bool>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @and_params_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]] -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]]], bool>, [[ARG1:%.*]]: !mo.tensor<[[[D3]], [[D4]]], bool>) -> !mo.tensor<[[SH5]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]]], bool> -> !mo.tensor<[[[D6]], [[D7]]], bool>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D3]], [[D4]]], bool> -> !mo.tensor<[[[D8]], [[D9]]], bool>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D8]], [[D9]]], bool>) -> !mo.tensor<[[[D6]], [[D7]]], bool>
//  CHECK-NEXT:   [[V3:%.*]] = mo.and([[V0]], [[V2]]) : !mo.tensor<[[[D6]], [[D7]]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D6]], [[D7]]], bool>) -> !mo.tensor<[[SH10]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH10]]> [[V4]] : !mo.tensor<[[SH10]], bool>

// -----

mo.graph @and_mix(%arg0: !mo.tensor<[?, 2, ?], bool>, %arg1: !mo.tensor<[?, ?, 3], bool>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.and(%arg0, %arg1) : (!mo.tensor<[?, 2, ?], bool>, !mo.tensor<[?, ?, 3], bool>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @and_mix<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]] -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], 2, [[D2]]], bool>, [[ARG1:%.*]]: !mo.tensor<[[[D3]], [[D4]], 3], bool>) -> !mo.tensor<[[SH5]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], 2, [[D2]]], bool> -> !mo.tensor<[[[D6]], 2, [[D7]]], bool>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D3]], [[D4]], 3], bool> -> !mo.tensor<[[[D8]], [[D9]], 3], bool>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[[D6]], 2, [[D7]]], bool>) -> !mo.tensor<[[[D6]], 2, 3], bool>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D8]], [[D9]], 3], bool>) -> !mo.tensor<[[[D6]], 2, 3], bool>
//  CHECK-NEXT:   [[V4:%.*]] = mo.and([[V2]], [[V3]]) : !mo.tensor<[[[D6]], 2, 3], bool>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[D6]], 2, 3], bool>) -> !mo.tensor<[[SH10]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH10]]> [[V5]] : !mo.tensor<[[SH10]], bool>

// -----

mo.graph @and_static(%arg0: !mo.tensor<[3, 2], bool>, %arg1: !mo.tensor<[3, 2], bool>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.and(%arg0, %arg1) : (!mo.tensor<[3, 2], bool>, !mo.tensor<[3, 2], bool>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @and_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[3, 2], bool>, [[ARG1:%.*]]: !mo.tensor<[3, 2], bool>) -> !mo.tensor<[[SH1]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.and([[ARG0]], [[ARG1]]) : !mo.tensor<[3, 2], bool>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[3, 2], bool>) -> !mo.tensor<[[SH2]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], bool>

// -----

mo.graph @or_dyn(%arg0: !mo.tensor<?, bool>, %arg1: !mo.tensor<?, bool>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.or(%arg0, %arg1) : (!mo.tensor<?, bool>, !mo.tensor<?, bool>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @or_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], bool>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], bool>) -> !mo.tensor<[[SH3]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], bool> -> !mo.tensor<[[SH4]], bool>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], bool> -> !mo.tensor<[[SH5]], bool>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], bool>) -> !mo.tensor<[[SH4]], bool>
//  CHECK-NEXT:   [[V3:%.*]] = mo.or([[V0]], [[V2]]) : !mo.tensor<[[SH4]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[SH4]], bool>) -> !mo.tensor<[[SH6]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], bool>

// -----

mo.graph @xor_dyn(%arg0: !mo.tensor<?, bool>, %arg1: !mo.tensor<?, bool>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.xor(%arg0, %arg1) : (!mo.tensor<?, bool>, !mo.tensor<?, bool>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @xor_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], bool>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], bool>) -> !mo.tensor<[[SH3]], bool> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], bool> -> !mo.tensor<[[SH4]], bool>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], bool> -> !mo.tensor<[[SH5]], bool>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], bool>) -> !mo.tensor<[[SH4]], bool>
//  CHECK-NEXT:   [[V3:%.*]] = mo.xor([[V0]], [[V2]]) : !mo.tensor<[[SH4]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[SH4]], bool>) -> !mo.tensor<[[SH6]], bool>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], bool>
