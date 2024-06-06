// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @tile_dyn_dyn(%arg0: !mo.tensor<?, f32>, %repeats: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.tile(%arg0, %repeats) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @tile_dyn_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH5]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[RMO_SH7]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D8:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], si64>) -> !mo.tensor<[[[RMO_D8]]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.tile<() -> [[RMO_SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]], [[V3]]) : (!mo.tensor<[[RMO_SH7]], f32>, !mo.tensor<[[[RMO_D8]]], si64>) -> !mo.tensor<[[RMO_SH9]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[RMO_SH9]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V5]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @tile_dyn_dims(%arg0: !mo.tensor<?, f32>, %repeats: !mo.tensor<[?], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.tile(%arg0, %repeats) : (!mo.tensor<?, f32>, !mo.tensor<[?], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @tile_dyn_dims<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[D2:[A-Za-z0-9_]*]] -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D5:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D2]]], si64> -> !mo.tensor<[[[D5]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[RMO_SH7]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D8:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[D5]]], si64>) -> !mo.tensor<[[[RMO_D8]]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.tile<() -> [[RMO_SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]], [[V3]]) : (!mo.tensor<[[RMO_SH7]], f32>, !mo.tensor<[[[RMO_D8]]], si64>) -> !mo.tensor<[[RMO_SH9]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[RMO_SH9]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V5]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @tile_params_dyn(%arg0: !mo.tensor<[?, ?, ?], f32>, %repeats: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.tile(%arg0, %repeats) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @tile_params_dyn<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[SH4:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH4]], si64>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]], [[D8:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D6]], [[D7]], [[D8]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH4]], si64> -> !mo.tensor<[[SH9]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]], [[RMO_D13:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[D6]], [[D7]], [[D8]]], f32>) -> !mo.tensor<[[[RMO_D11]], [[RMO_D12]], [[RMO_D13]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH9]], si64>) -> !mo.tensor<[3], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.tile<() -> [[RMO_D14:[A-Za-z0-9_]*]], [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]]>([[V2]], [[V3]]) : (!mo.tensor<[[[RMO_D11]], [[RMO_D12]], [[RMO_D13]]], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[RMO_D14]], [[RMO_D15]], [[RMO_D16]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D14]], [[RMO_D15]], [[RMO_D16]]], f32>) -> !mo.tensor<[[SH10]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH10]]> [[V5]] : !mo.tensor<[[SH10]], f32>

// -----

mo.graph @tile_params_dyn_static(%arg0: !mo.tensor<[?, ?, ?], f32>, %repeats: !mo.tensor<[3], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.tile(%arg0, %repeats) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @tile_params_dyn_static<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]] -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[3], si64>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D5:[A-Za-z0-9_]*]], [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D5]], [[D6]], [[D7]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]], [[RMO_D11:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[D5]], [[D6]], [[D7]]], f32>) -> !mo.tensor<[[[RMO_D9]], [[RMO_D10]], [[RMO_D11]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.tile<() -> [[RMO_D12:[A-Za-z0-9_]*]], [[RMO_D13:[A-Za-z0-9_]*]], [[RMO_D14:[A-Za-z0-9_]*]]>([[V1]], [[ARG1]]) : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]], [[RMO_D11]]], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[RMO_D12]], [[RMO_D13]], [[RMO_D14]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D12]], [[RMO_D13]], [[RMO_D14]]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V3]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @tile_params_mix(%arg0: !mo.tensor<[2, ?, ?], f32>, %repeats: !mo.tensor<[3], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.tile(%arg0, %repeats) : (!mo.tensor<[2, ?, ?], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @tile_params_mix<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]] -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, [[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[3], si64>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[2, [[D1]], [[D2]]], f32> -> !mo.tensor<[2, [[D4]], [[D5]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]], [[RMO_D9:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[2, [[D4]], [[D5]]], f32>) -> !mo.tensor<[[[RMO_D7]], [[RMO_D8]], [[RMO_D9]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.tile<() -> [[RMO_D10:[A-Za-z0-9_]*]], [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]]>([[V1]], [[ARG1]]) : (!mo.tensor<[[[RMO_D7]], [[RMO_D8]], [[RMO_D9]]], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[RMO_D10]], [[RMO_D11]], [[RMO_D12]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D10]], [[RMO_D11]], [[RMO_D12]]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V3]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @tile_dyn_out_param(%arg0: !mo.tensor<?, f32>, %repeats: !mo.tensor<?, si64>) -> !mo.tensor<[?, ?], f32> {
  %0 = rmo.mo.tile(%arg0, %repeats) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @tile_dyn_out_param<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[[D3]], [[D4]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH6]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH6]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.tile<() -> [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]]>([[V2]], [[V3]]) : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[D7:[A-Za-z0-9_]*]], [[D8:[A-Za-z0-9_]*]]>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>) -> !mo.tensor<[[[D7]], [[D8]]], f32>
//  CHECK-NEXT:   mo.output<[[D7]], [[D8]]> [[V5]] : !mo.tensor<[[[D7]], [[D8]]], f32>

// -----

mo.graph @tile_static(%arg0: !mo.tensor<[2, 4, 7], f32>, %repeats: !mo.tensor<[3], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.tile(%arg0, %repeats) : (!mo.tensor<[2, 4, 7], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @tile_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, 4, 7], f32>, [[ARG1:%.*]]: !mo.tensor<[3], si64>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[2, 4, 7], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.tile<() -> [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]]>([[V0]], [[ARG1]]) : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[RMO_D6]], [[RMO_D7]], [[RMO_D8]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D6]], [[RMO_D7]], [[RMO_D8]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V2]] : !mo.tensor<[[SH2]], f32>
