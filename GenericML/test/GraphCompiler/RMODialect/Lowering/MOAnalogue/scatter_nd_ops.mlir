// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @scatter_nd_dyn(%arg0: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH7]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[RMO_SH9]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.scatter_nd([[V0]], [[V3]], [[V2]]) : (!mo.tensor<[[SH5]], f32>, !mo.tensor<[[RMO_SH9]], f32>, !mo.tensor<[[SH7]], si64>) -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V5]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @scatter_nd_dyn_params(%arg0: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<[?, ?], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<[?, ?], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_dyn_params<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]] -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[[D3]], [[D4]]], si64>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH7]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]]>([[ARG2]]) : !mo.tensor<[[[D3]], [[D4]]], si64> -> !mo.tensor<[[[D8]], [[D9]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH11:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH7]], f32>) -> !mo.tensor<[[RMO_SH11]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.scatter_nd([[V0]], [[V3]], [[V2]]) : (!mo.tensor<[[SH6]], f32>, !mo.tensor<[[RMO_SH11]], f32>, !mo.tensor<[[[D8]], [[D9]]], si64>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[SH10]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH10]]> [[V5]] : !mo.tensor<[[SH10]], f32>


// -----

mo.graph @scatter_nd_params(%arg0: !mo.tensor<[?, ?], f32>, %updates: !mo.tensor<[?, ?, ?], f32>, %indices: !mo.tensor<[?, ?, ?], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]], [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]], [[D8:[A-Za-z0-9_]*]] -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>, [[ARG2:%.*]]: !mo.tensor<[[[D6]], [[D7]], [[D8]]], si64>) -> !mo.tensor<[[SH9]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D10]], [[D11]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D12:[A-Za-z0-9_]*]], [[D13:[A-Za-z0-9_]*]], [[D14:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32> -> !mo.tensor<[[[D12]], [[D13]], [[D14]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[D15:[A-Za-z0-9_]*]], [[D16:[A-Za-z0-9_]*]], [[D17:[A-Za-z0-9_]*]]>([[ARG2]]) : !mo.tensor<[[[D6]], [[D7]], [[D8]]], si64> -> !mo.tensor<[[[D15]], [[D16]], [[D17]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH19:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[D12]], [[D13]], [[D14]]], f32>) -> !mo.tensor<[[RMO_SH19]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.scatter_nd([[V0]], [[V3]], [[V2]]) : (!mo.tensor<[[[D10]], [[D11]]], f32>, !mo.tensor<[[RMO_SH19]], f32>, !mo.tensor<[[[D15]], [[D16]], [[D17]]], si64>) -> !mo.tensor<[[[D10]], [[D11]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH18:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[D10]], [[D11]]], f32>) -> !mo.tensor<[[SH18]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH18]]> [[V5]] : !mo.tensor<[[SH18]], f32>

// -----

mo.graph @scatter_nd_mix_params(%arg0: !mo.tensor<[4, ?], f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<[?, ?, 1], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<[4, ?], f32>, !mo.tensor<?, f32>, !mo.tensor<[?, ?, 1], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_mix_params<[[D1:[A-Za-z0-9_]*]], [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]] -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[4, [[D1]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[[D3]], [[D4]], 1], si64>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D6:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[4, [[D1]]], f32> -> !mo.tensor<[4, [[D6]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH7]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]]>([[ARG2]]) : !mo.tensor<[[[D3]], [[D4]], 1], si64> -> !mo.tensor<[[[D8]], [[D9]], 1], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]], [[RMO_D13:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH7]], f32>) -> !mo.tensor<[[[RMO_D11]], [[RMO_D12]], [[RMO_D13]]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.scatter_nd([[V0]], [[V3]], [[V2]]) : (!mo.tensor<[4, [[D6]]], f32>, !mo.tensor<[[[RMO_D11]], [[RMO_D12]], [[RMO_D13]]], f32>, !mo.tensor<[[[D8]], [[D9]], 1], si64>) -> !mo.tensor<[4, [[D6]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[4, [[D6]]], f32>) -> !mo.tensor<[[SH10]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH10]]> [[V5]] : !mo.tensor<[[SH10]], f32>

// -----

mo.graph @scatter_nd_static(%arg0: !mo.tensor<[4, 2], f32>, %updates: !mo.tensor<[1, 3, 2], f32>, %indices: !mo.tensor<[1, 3, 1], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[4, 2], f32>, [[ARG1:%.*]]: !mo.tensor<[1, 3, 2], f32>, [[ARG2:%.*]]: !mo.tensor<[1, 3, 1], si64>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[1, 3, 2], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.scatter_nd([[ARG0]], [[V0]], [[ARG2]]) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>, !mo.tensor<[1, 3, 1], si64>) -> !mo.tensor<[4, 2], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[4, 2], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V2]] : !mo.tensor<[[SH2]], f32>


// -----

mo.graph @scatter_nd_dyn_out_param(%arg0: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<[?, ?], f32> {
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_dyn_out_param<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape -> [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[[D4]], [[D5]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH7]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH8]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH13:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH7]], f32>) -> !mo.tensor<[[RMO_SH13]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = mo.scatter_nd([[V3]], [[V4]], [[V2]]) : (!mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>, !mo.tensor<[[RMO_SH13]], f32>, !mo.tensor<[[SH8]], si64>) -> !mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape<() -> [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]]>([[V5]]) {{.*}} : (!mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>) -> !mo.tensor<[[[D9]], [[D10]]], f32>
//  CHECK-NEXT:   mo.output<[[D9]], [[D10]]> [[V6]] : !mo.tensor<[[[D9]], [[D10]]], f32>

// -----

mo.graph @scatter_nd_dyn_out_param_2(%arg0: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<[?, 3], f32> {
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<[?, 3], f32>
  mo.output %0 : !mo.tensor<[?, 3], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_dyn_out_param_2<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape -> [[D4:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[[D4]], 3], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH7]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH11:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[RMO_SH11]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = mo.scatter_nd([[V3]], [[V4]], [[V2]]) : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>, !mo.tensor<[[RMO_SH11]], f32>, !mo.tensor<[[SH7]], si64>) -> !mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape<() -> [[D8:[A-Za-z0-9_]*]]>([[V5]]) {{.*}} : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>) -> !mo.tensor<[[[D8]], 3], f32>
//  CHECK-NEXT:   mo.output<[[D8]]> [[V6]] : !mo.tensor<[[[D8]], 3], f32>

// -----

mo.graph @scatter_nd_dyn_out_param_3(%arg0: !mo.tensor<[7, ?], f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<[?, 3], f32> {
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<[7, ?], f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<[?, 3], f32>
  mo.output %0 : !mo.tensor<[?, 3], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_dyn_out_param_3<[[D1:[A-Za-z0-9_]*]], [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape -> [[D4:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[7, [[D1]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[[D4]], 3], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D5:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[7, [[D1]]], f32> -> !mo.tensor<[7, [[D5]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH7]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[RMO_SH9]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.scatter_nd([[V0]], [[V3]], [[V2]]) : (!mo.tensor<[7, [[D5]]], f32>, !mo.tensor<[[RMO_SH9]], f32>, !mo.tensor<[[SH7]], si64>) -> !mo.tensor<[7, [[D5]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[D8:[A-Za-z0-9_]*]]>([[V4]]) {{.*}} : (!mo.tensor<[7, [[D5]]], f32>) -> !mo.tensor<[[[D8]], 3], f32>
//  CHECK-NEXT:   mo.output<[[D8]]> [[V5]] : !mo.tensor<[[[D8]], 3], f32>

// -----

mo.graph @scatter_nd_add_dyn(%arg0: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.scatter_nd.add(%arg0, %updates, %indices) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_add_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH7]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[RMO_SH9]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.scatter_nd.add([[V0]], [[V3]], [[V2]]) : (!mo.tensor<[[SH5]], f32>, !mo.tensor<[[RMO_SH9]], f32>, !mo.tensor<[[SH7]], si64>) -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V5]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @scatter_nd_max_dyn(%arg0: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.scatter_nd.max(%arg0, %updates, %indices) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_max_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH7]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[RMO_SH9]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.scatter_nd.max([[V0]], [[V3]], [[V2]]) : (!mo.tensor<[[SH5]], f32>, !mo.tensor<[[RMO_SH9]], f32>, !mo.tensor<[[SH7]], si64>) -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V5]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @scatter_nd_min_dyn(%arg0: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.scatter_nd.min(%arg0, %updates, %indices) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_min_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH7]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[RMO_SH9]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.scatter_nd.min([[V0]], [[V3]], [[V2]]) : (!mo.tensor<[[SH5]], f32>, !mo.tensor<[[RMO_SH9]], f32>, !mo.tensor<[[SH7]], si64>) -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V5]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @scatter_nd_mul_dyn(%arg0: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.scatter_nd.mul(%arg0, %updates, %indices) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @scatter_nd_mul_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH7]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[RMO_SH9]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.scatter_nd.mul([[V0]], [[V3]], [[V2]]) : (!mo.tensor<[[SH5]], f32>, !mo.tensor<[[RMO_SH9]], f32>, !mo.tensor<[[SH7]], si64>) -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V5]] : !mo.tensor<[[SH8]], f32>
