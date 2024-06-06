// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @concat_dyn(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>, %axis: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.concat[%axis: !mo.tensor<?, si64>](%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> (!mo.tensor<?, f32>)
  mo.output %0: !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @concat_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH5]], si64>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH7]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[SH5]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[RMO_SH9]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[SH7]], f32>) -> !mo.tensor<[[RMO_SH10]], f32>
//  CHECK-NEXT:   [[V6:%.*]] = mo.concat<() -> [[RMO_SH11:[A-Za-z0-9_]*]]: !mosh.ape>[[[V3]] : !mo.tensor<[], si64>] ([[V4]], [[V5]]) : (!mo.tensor<[[RMO_SH9]], f32>, !mo.tensor<[[RMO_SH10]], f32>) -> !mo.tensor<[[RMO_SH11]], f32>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V6]]) {{.*}} : (!mo.tensor<[[RMO_SH11]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V7]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @concat_dyn_params(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<[?, ?], f32>, %axis: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.concat[%axis: !mo.tensor<?, si64>](%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<[?, ?], f32>) -> (!mo.tensor<?, f32>)
  mo.output %0: !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @concat_dyn_params<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[SH4:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D2]], [[D3]]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH4]], si64>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH4]], si64> -> !mo.tensor<[[SH6]], si64>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH7]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D2]], [[D3]]], f32> -> !mo.tensor<[[[D8]], [[D9]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[SH6]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH7]], f32>) -> !mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D13:[A-Za-z0-9_]*]], [[RMO_D14:[A-Za-z0-9_]*]]>([[V2]]) {{.*}} : (!mo.tensor<[[[D8]], [[D9]]], f32>) -> !mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>
//  CHECK-NEXT:   [[V6:%.*]] = mo.concat<() -> [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]]>[[[V3]] : !mo.tensor<[], si64>] ([[V4]], [[V5]]) : (!mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>, !mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>) -> !mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V6]]) {{.*}} : (!mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>) -> !mo.tensor<[[SH10]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH10]]> [[V7]] : !mo.tensor<[[SH10]], f32>

// -----

mo.graph @concat_params_dyn(%arg0: !mo.tensor<[?, ?], f32>, %arg1: !mo.tensor<?, f32>, %axis: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.concat[%axis: !mo.tensor<?, si64>](%arg0, %arg1) : (!mo.tensor<[?, ?], f32>, !mo.tensor<?, f32>) -> (!mo.tensor<?, f32>)
  mo.output %0: !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @concat_params_dyn<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH3]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH4]], si64>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH4]], si64> -> !mo.tensor<[[SH6]], si64>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D7:[A-Za-z0-9_]*]], [[D8:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D7]], [[D8]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH3]], f32> -> !mo.tensor<[[SH9]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[SH6]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]]], f32>) -> !mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D13:[A-Za-z0-9_]*]], [[RMO_D14:[A-Za-z0-9_]*]]>([[V2]]) {{.*}} : (!mo.tensor<[[SH9]], f32>) -> !mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>
//  CHECK-NEXT:   [[V6:%.*]] = mo.concat<() -> [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]]>[[[V3]] : !mo.tensor<[], si64>] ([[V4]], [[V5]]) : (!mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>, !mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>) -> !mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V6]]) {{.*}} : (!mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>) -> !mo.tensor<[[SH10]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH10]]> [[V7]] : !mo.tensor<[[SH10]], f32>

// -----

mo.graph @concat_params_mix(%arg0: !mo.tensor<[2, ?], f32>, %arg1: !mo.tensor<?, f32>, %arg2: !mo.tensor<[2, ?], f32>, %axis: !mo.tensor<[], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.concat[%axis: !mo.tensor<[], si64>](%arg0, %arg1, %arg2) : (!mo.tensor<[2, ?], f32>, !mo.tensor<?, f32>, !mo.tensor<[2, ?], f32>) -> (!mo.tensor<?, f32>)
  mo.output %0: !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @concat_params_mix<[[D1:[A-Za-z0-9_]*]], [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[D3:[A-Za-z0-9_]*]] -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, [[D1]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[2, [[D3]]], f32>, [[ARG3:%.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D5:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[2, [[D1]]], f32> -> !mo.tensor<[2, [[D5]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[D7:[A-Za-z0-9_]*]]>([[ARG2]]) : !mo.tensor<[2, [[D3]]], f32> -> !mo.tensor<[2, [[D7]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[2, [[D5]]], f32>) -> !mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D13:[A-Za-z0-9_]*]], [[RMO_D14:[A-Za-z0-9_]*]]>([[V2]]) {{.*}} : (!mo.tensor<[2, [[D7]]], f32>) -> !mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>
//  CHECK-NEXT:   [[V6:%.*]] = mo.concat<() -> [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]]>[[[ARG3]] : !mo.tensor<[], si64>] ([[V3]], [[V4]], [[V5]]) : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>, !mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>, !mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>) -> !mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V6]]) {{.*}} : (!mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V7]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @concat_dyn_out_params(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>, %axis: !mo.tensor<?, si64>) -> !mo.tensor<[?, ?], f32> {
  %0 = rmo.mo.concat[%axis: !mo.tensor<?, si64>](%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> (!mo.tensor<[?, ?], f32>)
  mo.output %0: !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @concat_dyn_out_params<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape -> [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[[D4]], [[D5]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH6]], si64>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH7]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[SH6]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH7]], f32>) -> !mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D13:[A-Za-z0-9_]*]], [[RMO_D14:[A-Za-z0-9_]*]]>([[V2]]) {{.*}} : (!mo.tensor<[[SH8]], f32>) -> !mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>
//  CHECK-NEXT:   [[V6:%.*]] = mo.concat<() -> [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]]>[[[V3]] : !mo.tensor<[], si64>] ([[V4]], [[V5]]) : (!mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>, !mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>) -> !mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape<() -> [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]]>([[V6]]) {{.*}} : (!mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>) -> !mo.tensor<[[[D9]], [[D10]]], f32>
//  CHECK-NEXT:   mo.output<[[D9]], [[D10]]> [[V7]] : !mo.tensor<[[[D9]], [[D10]]], f32>

// -----

mo.graph @concat_static(%arg0: !mo.tensor<[2, 7], f32>, %arg1: !mo.tensor<[2, 3], f32>, %axis: !mo.tensor<[], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.concat[%axis: !mo.tensor<[], si64>](%arg0, %arg1) : (!mo.tensor<[2, 7], f32>, !mo.tensor<[2, 3], f32>) -> (!mo.tensor<?, f32>)
  mo.output %0: !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @concat_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, 7], f32>, [[ARG1:%.*]]: !mo.tensor<[2, 3], f32>, [[ARG2:%.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[2, 7], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.concat<() -> [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]]>[[[ARG2]] : !mo.tensor<[], si64>] ([[V0]], [[V1]]) : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>, !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>) -> !mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V3]] : !mo.tensor<[[SH2]], f32>
