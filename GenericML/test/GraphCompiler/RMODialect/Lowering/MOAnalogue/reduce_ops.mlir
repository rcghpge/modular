// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @mean_dyn(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, si32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.mean(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, si32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @mean_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si32>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si32> -> !mo.tensor<[[SH5]], si32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], si32>) -> !mo.tensor<[], si32>
//  CHECK-NEXT:   [[V3:%.*]] = mo.mean<() -> [[RMO_SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]], [[V2]]) : (!mo.tensor<[[SH4]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[RMO_SH7]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[RMO_SH7]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @mean_dyn_dims(%arg0: !mo.tensor<[?, ?, ?, ?], f32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %1 = rmo.mo.mean(%arg0, %arg1) : (!mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %1 : !mo.tensor<[?, ?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @mean_dyn_dims<[[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]], [[D4:[A-Z0-9]*]] -> [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]], [[D8:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[[D5]], [[D6]], [[D7]], [[D8]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]], [[D12:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D9]], [[D10]], [[D11]], [[D12]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.mean<() -> [[RMO_D17:[A-Za-z0-9_]*]], [[RMO_D18:[A-Za-z0-9_]*]], [[RMO_D19:[A-Za-z0-9_]*]], [[RMO_D20:[A-Za-z0-9_]*]]>([[V0]], [[ARG1]]) : (!mo.tensor<[[[D9]], [[D10]], [[D11]], [[D12]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D17]], [[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[D13:[A-Za-z0-9_]*]], [[D14:[A-Za-z0-9_]*]], [[D15:[A-Za-z0-9_]*]], [[D16:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D17]], [[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], f32>) -> !mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], f32>
//  CHECK-NEXT:   mo.output<[[D13]], [[D14]], [[D15]], [[D16]]> [[V2]] : !mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], f32>

// -----

mo.graph @mean_params_mix(%arg0: !mo.tensor<[?, 4, ?, ?], f32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.mean(%arg0, %arg1) : (!mo.tensor<[?, 4, ?, ?], f32>, !mo.tensor<[], si32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @mean_params_mix<[[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]] -> Sh4: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], 4, [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[SH4:[A-Za-z0-9_]*]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D5:[A-Za-z0-9_]*]], [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], 4, [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D5]], 4, [[D6]], [[D7]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.mean<() -> [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]], [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]]>([[V0]], [[ARG1]]) : (!mo.tensor<[[[D5]], 4, [[D6]], [[D7]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D9]], [[RMO_D10]], [[RMO_D11]], [[RMO_D12]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]], [[RMO_D11]], [[RMO_D12]]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V2]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @mean_dyn_const_axis(%arg0: !mo.tensor<[?, ?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %0 = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %1 = rmo.mo.mean(%arg0, %0) : (!mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %1 : !mo.tensor<[?, ?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @mean_dyn_const_axis<[[D9:[A-Z0-9]*]], [[D10:[A-Z0-9]*]], [[D11:[A-Z0-9]*]], [[D12:[A-Z0-9]*]] -> [[D13:[A-Z0-9]*]], [[D14:[A-Z0-9]*]], [[D15:[A-Z0-9]*]], [[D16:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D9]], [[D10]], [[D11]], [[D12]]], f32>) -> !mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], f32> {
//   CHECK-DAG:   [[V0:%.*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[], si32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D9]], [[D10]], [[D11]], [[D12]]], f32> -> !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.mean([[V1]], [[V0]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[D1]], 1, [[D3]], [[D4]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[D5:[A-Za-z0-9_]*]], [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]], [[D8:[A-Za-z0-9_]*]]>([[V2]]) {{.*}} : (!mo.tensor<[[[D1]], 1, [[D3]], [[D4]]], f32>) -> !mo.tensor<[[[D5]], [[D6]], [[D7]], [[D8]]], f32>
//  CHECK-NEXT:   mo.output<[[D5]], [[D6]], [[D7]], [[D8]]> [[V3]] : !mo.tensor<[[[D5]], [[D6]], [[D7]], [[D8]]], f32>

// -----

mo.graph @mean_static(%arg0: !mo.tensor<[5, 4, 8, 6], f32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.mean(%arg0, %arg1) : (!mo.tensor<[5, 4, 8, 6], f32>, !mo.tensor<[], si32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @mean_static<() -> Sh1: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[5, 4, 8, 6], f32>, [[ARG1:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[SH1:[A-Za-z0-9_]*]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.mean<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) : (!mo.tensor<[5, 4, 8, 6], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]], [[RMO_D6]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @mean_params_mix_const_axis(%arg0: !mo.tensor<[?, 4, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %0 = mo.constant {value = #M.dense_array<2> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %1 = rmo.mo.mean(%arg0, %0) : (!mo.tensor<[?, 4, ?, ?], f32>, !mo.tensor<[], si32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @mean_params_mix_const_axis<[[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]] -> Sh8: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D5]], 4, [[D6]], [[D7]]], f32>) -> !mo.tensor<[[SH8:[A-Za-z0-9_]*]], f32> {
//   CHECK-DAG:   [[V0:%.*]] = mo.constant {value = #M.dense_array<2> : tensor<1xsi32>} : !mo.tensor<[], si32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D5]], 4, [[D6]], [[D7]]], f32> -> !mo.tensor<[[[D1]], 4, [[D2]], [[D3]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.mean([[V1]], [[V0]]) : (!mo.tensor<[[[D1]], 4, [[D2]], [[D3]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[D1]], 4, 1, [[D3]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[D1]], 4, 1, [[D3]]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V3]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @mean_static_const_axis(%arg0: !mo.tensor<[5, 4, 8, 6], f32>) -> !mo.tensor<?, f32> {
  %0 = mo.constant {value = #M.dense_array<2> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %1 = rmo.mo.mean(%arg0, %0) : (!mo.tensor<[5, 4, 8, 6], f32>, !mo.tensor<[], si32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @mean_static_const_axis<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[5, 4, 8, 6], f32>) -> !mo.tensor<[[SH2]], f32> {
//   CHECK-DAG:   [[V0:%.*]] = mo.constant {value = #M.dense_array<2> : tensor<1xsi32>} : !mo.tensor<[], si32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.mean([[ARG0]], [[V0]]) : (!mo.tensor<[5, 4, 8, 6], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[5, 4, 1, 6], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[5, 4, 1, 6], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V2]] : !mo.tensor<[[SH1]], f32>

// -----


// Testing all reduce ops

mo.graph @reduce_mul_dyn_dims(%arg0: !mo.tensor<[?, ?, ?, ?], f32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %1 = rmo.mo.reduce.mul(%arg0, %arg1) : (!mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %1 : !mo.tensor<[?, ?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @reduce_mul_dyn_dims<[[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]], [[D4:[A-Z0-9]*]] -> [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]], [[D8:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[[D5]], [[D6]], [[D7]], [[D8]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]], [[D12:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D9]], [[D10]], [[D11]], [[D12]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.reduce.mul<() -> [[RMO_D17:[A-Za-z0-9_]*]], [[RMO_D18:[A-Za-z0-9_]*]], [[RMO_D19:[A-Za-z0-9_]*]], [[RMO_D20:[A-Za-z0-9_]*]]>([[V0]], [[ARG1]]) : (!mo.tensor<[[[D9]], [[D10]], [[D11]], [[D12]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D17]], [[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[D13:[A-Za-z0-9_]*]], [[D14:[A-Za-z0-9_]*]], [[D15:[A-Za-z0-9_]*]], [[D16:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D17]], [[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], f32>) -> !mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], f32>
//  CHECK-NEXT:   mo.output<[[D13]], [[D14]], [[D15]], [[D16]]> [[V2]] : !mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], f32>

// -----

mo.graph @reduce_add_dyn_dims(%arg0: !mo.tensor<[?, ?, ?, ?], f32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %1 = rmo.mo.reduce.add(%arg0, %arg1) : (!mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %1 : !mo.tensor<[?, ?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @reduce_add_dyn_dims<[[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]], [[D4:[A-Z0-9]*]] -> [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]], [[D8:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[[D5]], [[D6]], [[D7]], [[D8]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]], [[D12:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D9]], [[D10]], [[D11]], [[D12]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.reduce.add<() -> [[RMO_D17:[A-Za-z0-9_]*]], [[RMO_D18:[A-Za-z0-9_]*]], [[RMO_D19:[A-Za-z0-9_]*]], [[RMO_D20:[A-Za-z0-9_]*]]>([[V0]], [[ARG1]]) : (!mo.tensor<[[[D9]], [[D10]], [[D11]], [[D12]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D17]], [[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[D13:[A-Za-z0-9_]*]], [[D14:[A-Za-z0-9_]*]], [[D15:[A-Za-z0-9_]*]], [[D16:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D17]], [[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], f32>) -> !mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], f32>
//  CHECK-NEXT:   mo.output<[[D13]], [[D14]], [[D15]], [[D16]]> [[V2]] : !mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], f32>

// -----

mo.graph @reduce_min_dyn_dims(%arg0: !mo.tensor<[?, ?, ?, ?], f32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %1 = rmo.mo.reduce.min(%arg0, %arg1) : (!mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %1 : !mo.tensor<[?, ?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @reduce_min_dyn_dims<[[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]], [[D4:[A-Z0-9]*]] -> [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]], [[D8:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[[D5]], [[D6]], [[D7]], [[D8]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]], [[D12:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D9]], [[D10]], [[D11]], [[D12]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.reduce.min<() -> [[RMO_D17:[A-Za-z0-9_]*]], [[RMO_D18:[A-Za-z0-9_]*]], [[RMO_D19:[A-Za-z0-9_]*]], [[RMO_D20:[A-Za-z0-9_]*]]>([[V0]], [[ARG1]]) : (!mo.tensor<[[[D9]], [[D10]], [[D11]], [[D12]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D17]], [[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[D13:[A-Za-z0-9_]*]], [[D14:[A-Za-z0-9_]*]], [[D15:[A-Za-z0-9_]*]], [[D16:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D17]], [[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], f32>) -> !mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], f32>
//  CHECK-NEXT:   mo.output<[[D13]], [[D14]], [[D15]], [[D16]]> [[V2]] : !mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], f32>

// -----

mo.graph @reduce_max_dyn_dims(%arg0: !mo.tensor<[?, ?, ?, ?], f32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %1 = rmo.mo.reduce.max(%arg0, %arg1) : (!mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %1 : !mo.tensor<[?, ?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @reduce_max_dyn_dims<[[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]], [[D4:[A-Z0-9]*]] -> [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]], [[D8:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[[D5]], [[D6]], [[D7]], [[D8]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]], [[D12:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D9]], [[D10]], [[D11]], [[D12]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.reduce.max<() -> [[RMO_D17:[A-Za-z0-9_]*]], [[RMO_D18:[A-Za-z0-9_]*]], [[RMO_D19:[A-Za-z0-9_]*]], [[RMO_D20:[A-Za-z0-9_]*]]>([[V0]], [[ARG1]]) : (!mo.tensor<[[[D9]], [[D10]], [[D11]], [[D12]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D17]], [[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[D13:[A-Za-z0-9_]*]], [[D14:[A-Za-z0-9_]*]], [[D15:[A-Za-z0-9_]*]], [[D16:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D17]], [[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], f32>) -> !mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], f32>
//  CHECK-NEXT:   mo.output<[[D13]], [[D14]], [[D15]], [[D16]]> [[V2]] : !mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], f32>

// -----

mo.graph @argmin_dyn_dims(%arg0: !mo.tensor<[?, ?, ?, ?], f32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?, ?], si32> {
  %1 = rmo.mo.arg_min(%arg0, %arg1) : (!mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?, ?], si32>
  mo.output %1 : !mo.tensor<[?, ?, ?, ?], si32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @argmin_dyn_dims<[[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]], [[D4:[A-Z0-9]*]] -> [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]], [[D8:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[[D5]], [[D6]], [[D7]], [[D8]]], si32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]], [[D12:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D9]], [[D10]], [[D11]], [[D12]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.arg_min<() -> [[RMO_D17:[A-Za-z0-9_]*]], [[RMO_D18:[A-Za-z0-9_]*]], [[RMO_D19:[A-Za-z0-9_]*]], [[RMO_D20:[A-Za-z0-9_]*]]>([[V0]], [[ARG1]]) : (!mo.tensor<[[[D9]], [[D10]], [[D11]], [[D12]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D17]], [[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], si32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[D13:[A-Za-z0-9_]*]], [[D14:[A-Za-z0-9_]*]], [[D15:[A-Za-z0-9_]*]], [[D16:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D17]], [[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], si32>) -> !mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], si32>
//  CHECK-NEXT:   mo.output<[[D13]], [[D14]], [[D15]], [[D16]]> [[V2]] : !mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], si32>

// -----

mo.graph @argmax_dyn_dims(%arg0: !mo.tensor<[?, ?, ?, ?], f32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?, ?], si32> {
  %1 = rmo.mo.arg_max(%arg0, %arg1) : (!mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?, ?], si32>
  mo.output %1 : !mo.tensor<[?, ?, ?, ?], si32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @argmax_dyn_dims<[[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]], [[D4:[A-Z0-9]*]] -> [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]], [[D8:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[[D5]], [[D6]], [[D7]], [[D8]]], si32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]], [[D12:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D9]], [[D10]], [[D11]], [[D12]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.arg_max<() -> [[RMO_D17:[A-Za-z0-9_]*]], [[RMO_D18:[A-Za-z0-9_]*]], [[RMO_D19:[A-Za-z0-9_]*]], [[RMO_D20:[A-Za-z0-9_]*]]>([[V0]], [[ARG1]]) : (!mo.tensor<[[[D9]], [[D10]], [[D11]], [[D12]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D17]], [[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], si32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[D13:[A-Za-z0-9_]*]], [[D14:[A-Za-z0-9_]*]], [[D15:[A-Za-z0-9_]*]], [[D16:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D17]], [[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], si32>) -> !mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], si32>
//  CHECK-NEXT:   mo.output<[[D13]], [[D14]], [[D15]], [[D16]]> [[V2]] : !mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], si32>
