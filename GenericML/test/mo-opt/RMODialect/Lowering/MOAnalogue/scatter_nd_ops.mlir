// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @scatter_nd_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.scatter_nd<() -> Sh4: !mosh.ape>(%arg0, %updates, %indices) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %0 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_dyn<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[RMO_SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.scatter_nd([[ARG0]], [[V0]], [[ARG2]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[RMO_SH4]], f32>, !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH0]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V2]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @scatter_nd_dyn_params<Sh0: !mosh.ape, Sh1: !mosh.ape, D0, D1 -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<[D0, D1], si64>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.scatter_nd<() -> Sh3: !mosh.ape>(%arg0, %updates, %indices) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<[D0, D1], si64>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_dyn_params<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[D0:[A-Za-z0-9_]*]], [[D1:[A-Za-z0-9_]*]] -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG2:%.*]]: !mo.tensor<[[[D0]], [[D1]]], si64>) -> !mo.tensor<[[SH2]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[RMO_SH3]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.scatter_nd([[ARG0]], [[V0]], [[ARG2]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[RMO_SH3]], f32>, !mo.tensor<[[[D0]], [[D1]]], si64>) -> !mo.tensor<[[SH0]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> [[V2]] : !mo.tensor<[[SH3]], f32>


// -----

mo.graph @scatter_nd_params<D0, D1, D2, D3, D4, D5, D6, D7 -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[D0, D1], f32>, %updates: !mo.tensor<[D2, D3, D4], f32>, %indices: !mo.tensor<[D5, D6, D7], si64>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.scatter_nd<() -> Sh1: !mosh.ape>(%arg0, %updates, %indices) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3, D4], f32>, !mo.tensor<[D5, D6, D7], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_params<[[D0:[A-Za-z0-9_]*]], [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]], [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]] -> [[SH0:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32>, [[ARG2:%.*]]: !mo.tensor<[[[D5]], [[D6]], [[D7]]], si64>) -> !mo.tensor<[[SH0]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) {{.*}} : (!mo.tensor<[[[D2]], [[D3]], [[D4]]], f32>) -> !mo.tensor<[[RMO_SH1]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.scatter_nd([[ARG0]], [[V0]], [[ARG2]]) : (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[RMO_SH1]], f32>, !mo.tensor<[[[D5]], [[D6]], [[D7]]], si64>) -> !mo.tensor<[[[D0]], [[D1]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[D0]], [[D1]]], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V2]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @scatter_nd_mix_params<D0, Sh0: !mosh.ape, D1, D2 -> Sh1: !mosh.ape>(%arg0: !mo.tensor<[4, D0], f32>, %updates: !mo.tensor<Sh0, f32>, %indices: !mo.tensor<[D1, D2, 1], si64>) -> !mo.tensor<Sh1, f32> {
  %0 = rmo.mo.scatter_nd<() -> Sh2: !mosh.ape>(%arg0, %updates, %indices) : (!mo.tensor<[4, D0], f32>, !mo.tensor<Sh0, f32>, !mo.tensor<[D1, D2, 1], si64>) -> !mo.tensor<Sh2, f32>
  mo.output<:!mosh.ape Sh2> %0 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_mix_params<[[D0:[A-Za-z0-9_]*]], [[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]] -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[4, [[D0]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG2:%.*]]: !mo.tensor<[[[D1]], [[D2]], 1], si64>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.scatter_nd([[ARG0]], [[V0]], [[ARG2]]) : (!mo.tensor<[4, [[D0]]], f32>, !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>, !mo.tensor<[[[D1]], [[D2]], 1], si64>) -> !mo.tensor<[4, [[D0]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[4, [[D0]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V2]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @scatter_nd_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[4, 2], f32>, %updates: !mo.tensor<[1, 3, 2], f32>, %indices: !mo.tensor<[1, 3, 1], si64>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.scatter_nd<() -> Sh1: !mosh.ape>(%arg0, %updates, %indices) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_static<() -> [[SH0:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[4, 2], f32>, [[ARG1:%.*]]: !mo.tensor<[1, 3, 2], f32>, [[ARG2:%.*]]: !mo.tensor<[1, 3, 1], si64>) -> !mo.tensor<[[SH0]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[1, 3, 2], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.scatter_nd([[ARG0]], [[V0]], [[ARG2]]) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32>, !mo.tensor<[1, 3, 1], si64>) -> !mo.tensor<[4, 2], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[4, 2], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V2]] : !mo.tensor<[[SH1]], f32>


// -----

mo.graph @scatter_nd_dyn_out_param<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> D0, D1>(%arg0: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<[D0, D1], f32> {
  %0 = rmo.mo.scatter_nd<() -> D2, D3>(%arg0, %updates, %indices) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %0 : !mo.tensor<[D2, D3], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_dyn_out_param<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[D0:[A-Za-z0-9_]*]], [[D1:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[[D0]], [[D1]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[RMO_SH3]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.scatter_nd([[V0]], [[V1]], [[ARG2]]) : (!mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>, !mo.tensor<[[RMO_SH3]], f32>, !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]]>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>) -> !mo.tensor<[[[D2]], [[D3]]], f32>
//  CHECK-NEXT:   mo.output<[[D2]], [[D3]]> [[V3]] : !mo.tensor<[[[D2]], [[D3]]], f32>

// -----

mo.graph @scatter_nd_dyn_out_param_2<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> D0>(%arg0: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<[D0, 3], f32> {
  %0 = rmo.mo.scatter_nd<() -> D1>(%arg0, %updates, %indices) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>) -> !mo.tensor<[D1, 3], f32>
  mo.output<D1> %0 : !mo.tensor<[D1, 3], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_dyn_out_param_2<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[D0:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[[D0]], 3], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[RMO_SH3]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.scatter_nd([[V0]], [[V1]], [[ARG2]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[[RMO_SH3]], f32>, !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[D1:[A-Za-z0-9_]*]]>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>) -> !mo.tensor<[[[D1]], 3], f32>
//  CHECK-NEXT:   mo.output<[[D1]]> [[V3]] : !mo.tensor<[[[D1]], 3], f32>

// -----

mo.graph @scatter_nd_dyn_out_param_3<D0, Sh0: !mosh.ape, Sh1: !mosh.ape -> D1>(%arg0: !mo.tensor<[7, D0], f32>, %updates: !mo.tensor<Sh0, f32>, %indices: !mo.tensor<Sh1, si64>) -> !mo.tensor<[D1, 3], f32> {
  %0 = rmo.mo.scatter_nd<() -> D2>(%arg0, %updates, %indices) : (!mo.tensor<[7, D0], f32>, !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<[D2, 3], f32>
  mo.output<D2> %0 : !mo.tensor<[D2, 3], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_dyn_out_param_3<[[D0:[A-Za-z0-9_]*]], [[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape -> [[D1:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[7, [[D0]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[[D1]], 3], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[RMO_SH2]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.scatter_nd([[ARG0]], [[V0]], [[ARG2]]) : (!mo.tensor<[7, [[D0]]], f32>, !mo.tensor<[[RMO_SH2]], f32>, !mo.tensor<[[SH1]], si64>) -> !mo.tensor<[7, [[D0]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[D2:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[7, [[D0]]], f32>) -> !mo.tensor<[[[D2]], 3], f32>
//  CHECK-NEXT:   mo.output<[[D2]]> [[V2]] : !mo.tensor<[[[D2]], 3], f32>

// -----

mo.graph @scatter_nd_add_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.scatter_nd.add<() -> Sh4: !mosh.ape>(%arg0, %updates, %indices) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %0 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_add_dyn<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[RMO_SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.scatter_nd.add([[ARG0]], [[V0]], [[ARG2]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[RMO_SH4]], f32>, !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH0]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V2]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @scatter_nd_max_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.scatter_nd.max<() -> Sh4: !mosh.ape>(%arg0, %updates, %indices) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %0 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_max_dyn<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[RMO_SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.scatter_nd.max([[ARG0]], [[V0]], [[ARG2]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[RMO_SH4]], f32>, !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH0]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V2]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @scatter_nd_min_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.scatter_nd.min<() -> Sh4: !mosh.ape>(%arg0, %updates, %indices) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %0 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_min_dyn<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[RMO_SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.scatter_nd.min([[ARG0]], [[V0]], [[ARG2]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[RMO_SH4]], f32>, !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH0]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V2]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @scatter_nd_mul_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.scatter_nd.mul<() -> Sh4: !mosh.ape>(%arg0, %updates, %indices) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %0 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_mul_dyn<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[RMO_SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.scatter_nd.mul([[ARG0]], [[V0]], [[ARG2]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[RMO_SH4]], f32>, !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH0]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V2]] : !mo.tensor<[[SH4]], f32>
