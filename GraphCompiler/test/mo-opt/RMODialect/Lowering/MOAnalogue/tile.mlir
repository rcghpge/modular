// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @tile_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %repeats: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.tile<() -> Sh3: !mosh.ape>(%arg0, %repeats) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @tile_dyn_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[RMO_SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[[RMO_D5]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.tile<() -> [[RMO_SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]], [[V1]]) : (!mo.tensor<[[RMO_SH4]], f32>, !mo.tensor<[[[RMO_D5]]], si64>) -> !mo.tensor<[[RMO_SH6]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[RMO_SH6]], f32>) -> !mo.tensor<[[SH7]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH7]]> [[V3]] : !mo.tensor<[[SH7]], f32>

// -----

mo.graph @tile_dyn_dims<Sh0: !mosh.ape, D1 -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %repeats: !mo.tensor<[D1], si64>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.tile<() -> Sh3: !mosh.ape>(%arg0, %repeats) : (!mo.tensor<Sh0, f32>, !mo.tensor<[D1], si64>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @tile_dyn_dims<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[D2:[A-Za-z0-9_]*]] -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[RMO_SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[[[RMO_D5]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.tile<() -> [[RMO_SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]], [[V1]]) : (!mo.tensor<[[RMO_SH4]], f32>, !mo.tensor<[[[RMO_D5]]], si64>) -> !mo.tensor<[[RMO_SH6]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[RMO_SH6]], f32>) -> !mo.tensor<[[SH7]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH7]]> [[V3]] : !mo.tensor<[[SH7]], f32>

// -----

mo.graph @tile_params_dyn<D0, D1, D2, Sh3: !mosh.ape -> Sh4: !mosh.ape>(%arg0: !mo.tensor<[D0, D1, D2], f32>, %repeats: !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32> {
  %0 = rmo.mo.tile<() -> Sh5: !mosh.ape>(%arg0, %repeats) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh5, f32>
  mo.output<:!mosh.ape Sh5> %0 : !mo.tensor<Sh5, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @tile_params_dyn<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[SH4:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH4]], si64>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[RMO_D6]], [[RMO_D7]], [[RMO_D8]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape([[ARG1]]) {{.*}} : (!mo.tensor<[[SH4]], si64>) -> !mo.tensor<[3], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.tile<() -> [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]], [[RMO_D11:[A-Za-z0-9_]*]]>([[V0]], [[V1]]) : (!mo.tensor<[[[RMO_D6]], [[RMO_D7]], [[RMO_D8]]], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[RMO_D9]], [[RMO_D10]], [[RMO_D11]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH12:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]], [[RMO_D11]]], f32>) -> !mo.tensor<[[SH12]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH12]]> [[V3]] : !mo.tensor<[[SH12]], f32>

// -----

mo.graph @tile_params_dyn_static<D0, D1, D2 -> Sh3: !mosh.ape>(%arg0: !mo.tensor<[D0, D1, D2], f32>, %repeats: !mo.tensor<[3], si64>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.tile<() -> Sh4: !mosh.ape>(%arg0, %repeats) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %0 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @tile_params_dyn_static<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]] -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[3], si64>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]], [[RMO_D7]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.tile<() -> [[RMO_D8:[A-Za-z0-9_]*]], [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]]>([[V0]], [[ARG1]]) : (!mo.tensor<[[[RMO_D5]], [[RMO_D6]], [[RMO_D7]]], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[RMO_D8]], [[RMO_D9]], [[RMO_D10]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH11:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D8]], [[RMO_D9]], [[RMO_D10]]], f32>) -> !mo.tensor<[[SH11]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH11]]> [[V2]] : !mo.tensor<[[SH11]], f32>

// -----

mo.graph @tile_params_mix<D0, D1 -> Sh2: !mosh.ape>(%arg0: !mo.tensor<[2, D0, D1], f32>, %repeats: !mo.tensor<[3], si64>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.tile<() -> Sh3: !mosh.ape>(%arg0, %repeats) : (!mo.tensor<[2, D0, D1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @tile_params_mix<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]] -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, [[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[3], si64>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[2, [[D1]], [[D2]]], f32>) -> !mo.tensor<[[[RMO_D4]], [[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.tile<() -> [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]], [[RMO_D9:[A-Za-z0-9_]*]]>([[V0]], [[ARG1]]) : (!mo.tensor<[[[RMO_D4]], [[RMO_D5]], [[RMO_D6]]], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[RMO_D7]], [[RMO_D8]], [[RMO_D9]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D7]], [[RMO_D8]], [[RMO_D9]]], f32>) -> !mo.tensor<[[SH10]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH10]]> [[V2]] : !mo.tensor<[[SH10]], f32>

// -----

mo.graph @tile_dyn_out_param<Sh0: !mosh.ape, Sh1: !mosh.ape -> D2, D3>(%arg0: !mo.tensor<Sh0, f32>, %repeats: !mo.tensor<Sh1, si64>) -> !mo.tensor<[D2, D3], f32> {
  %0 = rmo.mo.tile<() -> D4, D5>(%arg0, %repeats) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %0 : !mo.tensor<[D4, D5], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @tile_dyn_out_param<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[[D3]], [[D4]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape([[ARG1]]) {{.*}} : (!mo.tensor<[[SH2]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.tile<() -> [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]]>([[V0]], [[V1]]) : (!mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]]>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f32>) -> !mo.tensor<[[[D9]], [[D10]]], f32>
//  CHECK-NEXT:   mo.output<[[D9]], [[D10]]> [[V3]] : !mo.tensor<[[[D9]], [[D10]]], f32>

// -----

mo.graph @tile_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[2, 4, 7], f32>, %repeats: !mo.tensor<[3], si64>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.tile<() -> Sh1: !mosh.ape>(%arg0, %repeats) : (!mo.tensor<[2, 4, 7], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @tile_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, 4, 7], f32>, [[ARG1:%.*]]: !mo.tensor<[3], si64>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[2, 4, 7], f32>) -> !mo.tensor<[[[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.tile<() -> [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]]>([[V0]], [[ARG1]]) : (!mo.tensor<[[[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]], [[RMO_D7]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D5]], [[RMO_D6]], [[RMO_D7]]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V2]] : !mo.tensor<[[SH8]], f32>
