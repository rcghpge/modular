// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @mean_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.mo.mean<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @mean_dyn<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], si32>) -> !mo.tensor<[[SH2]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], si32>) -> !mo.tensor<[], si32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.mean<() -> [[RMO_SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]], [[V0]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[RMO_SH1]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[RMO_SH1]], f32>) -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> [[V2]] : !mo.tensor<[[SH3]], f32>

// -----

mo.graph @mean_dyn_dims<D0, D1, D2, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<[D4, D5, D6, D7], f32> {
  %1 = rmo.mo.mean<() -> D8, D9, D10, D11>(%arg0, %arg1) : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[D8, D9, D10, D11], f32>
  mo.output<D8, D9, D10, D11> %1 : !mo.tensor<[D8, D9, D10, D11], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @mean_dyn_dims<[[D0:[A-Z0-9]*]], [[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]] -> [[D4:[A-Z0-9]*]], [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.mean<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//  CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> [[V1]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>

// -----

mo.graph @mean_params_mix<D0, D1, D2 -> Sh3: !mosh.ape>(%arg0: !mo.tensor<[D0, 4, D1, D2], f32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<Sh3, f32> {
  %1 = rmo.mo.mean<() -> Sh4: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[D0, 4, D1, D2], f32>, !mo.tensor<[], si32>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %1 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @mean_params_mix<[[D0:[A-Z0-9]*]], [[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]] -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D0]], 4, [[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.mean<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) : (!mo.tensor<[[[D0]], 4, [[D1]], [[D2]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V1]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @mean_dyn_const_axis<D0, D1, D2, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[D4, D5, D6, D7], f32> {
  %0 = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %1 = rmo.mo.mean<() -> D8, D9, D10, D11>(%arg0, %0) : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[D8, D9, D10, D11], f32>
  mo.output<D8, D9, D10, D11> %1 : !mo.tensor<[D8, D9, D10, D11], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @mean_dyn_const_axis<[[D0:[A-Z0-9]*]], [[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]] -> [[D4:[A-Z0-9]*]], [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32> {
//   CHECK-DAG:   [[V0:%.*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[], si32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.mean([[ARG0]], [[V0]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[D0]], 1, [[D2]], [[D3]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[D0]], 1, [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//  CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> [[V2]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>

// -----

mo.graph @mean_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[5, 4, 8, 6], f32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.mean<() -> Sh1: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[5, 4, 8, 6], f32>, !mo.tensor<[], si32>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @mean_static<() -> [[SH0:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[5, 4, 8, 6], f32>, [[ARG1:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[SH0]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.mean<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) : (!mo.tensor<[5, 4, 8, 6], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V1]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @mean_params_mix_const_axis<D0, D1, D2 -> Sh3: !mosh.ape>(%arg0: !mo.tensor<[D0, 4, D1, D2], f32>) -> !mo.tensor<Sh3, f32> {
  %0 = mo.constant {value = #M.dense_array<2> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %1 = rmo.mo.mean<() -> Sh4: !mosh.ape>(%arg0, %0) : (!mo.tensor<[D0, 4, D1, D2], f32>, !mo.tensor<[], si32>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %1 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @mean_params_mix_const_axis<[[D0:[A-Z0-9]*]], [[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]] -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D0]], 4, [[D1]], [[D2]]], f32>) -> !mo.tensor<[[SH3]], f32> {
//   CHECK-DAG:   [[V0:%.*]] = mo.constant {value = #M.dense_array<2> : tensor<1xsi32>} : !mo.tensor<[], si32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.mean([[ARG0]], [[V0]]) : (!mo.tensor<[[[D0]], 4, [[D1]], [[D2]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[D0]], 4, 1, [[D2]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[D0]], 4, 1, [[D2]]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V2]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @mean_static_const_axis<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[5, 4, 8, 6], f32>) -> !mo.tensor<Sh0, f32> {
  %0 = mo.constant {value = #M.dense_array<2> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %1 = rmo.mo.mean<() -> Sh1: !mosh.ape>(%arg0, %0) : (!mo.tensor<[5, 4, 8, 6], f32>, !mo.tensor<[], si32>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @mean_static_const_axis<() -> [[SH0:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[5, 4, 8, 6], f32>) -> !mo.tensor<[[SH0]], f32> {
//   CHECK-DAG:   [[V0:%.*]] = mo.constant {value = #M.dense_array<2> : tensor<1xsi32>} : !mo.tensor<[], si32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.mean([[ARG0]], [[V0]]) : (!mo.tensor<[5, 4, 8, 6], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[5, 4, 1, 6], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[5, 4, 1, 6], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V2]] : !mo.tensor<[[SH1]], f32>

// -----


// Testing all reduce ops

mo.graph @reduce_mul_dyn_dims<D0, D1, D2, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<[D4, D5, D6, D7], f32> {
  %1 = rmo.mo.reduce.mul<() -> D8, D9, D10, D11>(%arg0, %arg1) : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[D8, D9, D10, D11], f32>
  mo.output<D8, D9, D10, D11> %1 : !mo.tensor<[D8, D9, D10, D11], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @reduce_mul_dyn_dims<[[D0:[A-Z0-9]*]], [[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]] -> [[D4:[A-Z0-9]*]], [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.reduce.mul<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//  CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> [[V1]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>

// -----

mo.graph @reduce_add_dyn_dims<D0, D1, D2, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<[D4, D5, D6, D7], f32> {
  %1 = rmo.mo.reduce.add<() -> D8, D9, D10, D11>(%arg0, %arg1) : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[D8, D9, D10, D11], f32>
  mo.output<D8, D9, D10, D11> %1 : !mo.tensor<[D8, D9, D10, D11], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @reduce_add_dyn_dims<[[D0:[A-Z0-9]*]], [[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]] -> [[D4:[A-Z0-9]*]], [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.reduce.add<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//  CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> [[V1]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>

// -----

mo.graph @reduce_min_dyn_dims<D0, D1, D2, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<[D4, D5, D6, D7], f32> {
  %1 = rmo.mo.reduce.min<() -> D8, D9, D10, D11>(%arg0, %arg1) : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[D8, D9, D10, D11], f32>
  mo.output<D8, D9, D10, D11> %1 : !mo.tensor<[D8, D9, D10, D11], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @reduce_min_dyn_dims<[[D0:[A-Z0-9]*]], [[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]] -> [[D4:[A-Z0-9]*]], [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.reduce.min<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//  CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> [[V1]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>

// -----

mo.graph @reduce_max_dyn_dims<D0, D1, D2, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<[D4, D5, D6, D7], f32> {
  %1 = rmo.mo.reduce.max<() -> D8, D9, D10, D11>(%arg0, %arg1) : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[D8, D9, D10, D11], f32>
  mo.output<D8, D9, D10, D11> %1 : !mo.tensor<[D8, D9, D10, D11], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @reduce_max_dyn_dims<[[D0:[A-Z0-9]*]], [[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]] -> [[D4:[A-Z0-9]*]], [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.reduce.max<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//  CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> [[V1]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>

// -----

mo.graph @argmin_dyn_dims<D0, D1, D2, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<[D4, D5, D6, D7], si32> {
  %1 = rmo.mo.arg_min<() -> D8, D9, D10, D11>(%arg0, %arg1) : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[D8, D9, D10, D11], si32>
  mo.output<D8, D9, D10, D11> %1 : !mo.tensor<[D8, D9, D10, D11], si32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @argmin_dyn_dims<[[D0:[A-Z0-9]*]], [[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]] -> [[D4:[A-Z0-9]*]], [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], si32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.arg_min<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], si32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], si32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], si32>
//  CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> [[V1]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], si32>

// -----

mo.graph @argmax_dyn_dims<D0, D1, D2, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<[D4, D5, D6, D7], si32> {
  %1 = rmo.mo.arg_max<() -> D8, D9, D10, D11>(%arg0, %arg1) : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[D8, D9, D10, D11], si32>
  mo.output<D8, D9, D10, D11> %1 : !mo.tensor<[D8, D9, D10, D11], si32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @argmax_dyn_dims<[[D0:[A-Z0-9]*]], [[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]] -> [[D4:[A-Z0-9]*]], [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], si32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.arg_max<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], si32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], si32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], si32>
//  CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> [[V1]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], si32>
