// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @pad_reflect_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %pads: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.pad.reflect<() -> Sh3: !mosh.ape>(%arg0, %pads) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @pad_reflect_dyn_dyn<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[SH2]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[RMO_SH3]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[[RMO_D4]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.pad.reflect<() -> [[RMO_SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]], [[V1]]) : (!mo.tensor<[[RMO_SH3]], f32>, !mo.tensor<[[[RMO_D4]]], si64>) -> !mo.tensor<[[RMO_SH5]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2b:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[RMO_SH5]], f32>) -> !mo.tensor<[[SH2b]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2b]]> [[V3]] : !mo.tensor<[[SH2b]], f32>

// -----

mo.graph @pad_reflect_dyn_params<Sh0: !mosh.ape, D0 -> Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %pads: !mo.tensor<[D0], si64>) -> !mo.tensor<Sh1, f32> {
  %0 = rmo.mo.pad.reflect<() -> Sh2: !mosh.ape>(%arg0, %pads) : (!mo.tensor<Sh0, f32>, !mo.tensor<[D0], si64>) -> !mo.tensor<Sh2, f32>
  mo.output<:!mosh.ape Sh2> %0 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @pad_reflect_dyn_params<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[D0:[A-Za-z0-9_]*]] -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D0]]], si64>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[RMO_SH2]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[[D0]]], si64>) -> !mo.tensor<[[[RMO_D3]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.pad.reflect<() -> [[RMO_SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]], [[V1]]) : (!mo.tensor<[[RMO_SH2]], f32>, !mo.tensor<[[[RMO_D3]]], si64>) -> !mo.tensor<[[RMO_SH4]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1b:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[RMO_SH4]], f32>) -> !mo.tensor<[[SH1b]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1b]]> [[V3]] : !mo.tensor<[[SH1b]], f32>

// -----

mo.graph @pad_reflect_params_dyn<D0, D1, D2, Sh0: !mosh.ape -> Sh1: !mosh.ape>(%arg0: !mo.tensor<[D0, D1, D2], f32>, %pads: !mo.tensor<Sh0, si64>) -> !mo.tensor<Sh1, f32> {
  %0 = rmo.mo.pad.reflect<() -> Sh2: !mosh.ape>(%arg0, %pads) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<Sh0, si64>) -> !mo.tensor<Sh2, f32>
  mo.output<:!mosh.ape Sh2> %0 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @pad_reflect_params_dyn<[[D0:[A-Za-z0-9_]*]], [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[SH0:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH0]], si64>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape([[ARG1]]) {{.*}} : (!mo.tensor<[[SH0]], si64>) -> !mo.tensor<[6], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.pad.reflect<() -> [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]]>([[V0]], [[V1]]) : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>, !mo.tensor<[6], si64>) -> !mo.tensor<[[[RMO_D6]], [[RMO_D7]], [[RMO_D8]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1b:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D6]], [[RMO_D7]], [[RMO_D8]]], f32>) -> !mo.tensor<[[SH1b]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1b]]> [[V3]] : !mo.tensor<[[SH1b]], f32>

// -----

mo.graph @pad_reflect_dyn_params_rank<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %pads: !mo.tensor<[6], si64>) -> !mo.tensor<Sh1, f32> {
  %0 = rmo.mo.pad.reflect<() -> Sh2: !mosh.ape>(%arg0, %pads) : (!mo.tensor<Sh0, f32>, !mo.tensor<[6], si64>) -> !mo.tensor<Sh2, f32>
  mo.output<:!mosh.ape Sh2> %0 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @pad_reflect_dyn_params_rank<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[6], si64>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.pad.reflect<() -> [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]]>([[V0]], [[ARG1]]) : (!mo.tensor<[[[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>, !mo.tensor<[6], si64>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]], [[RMO_D7]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1b:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D5]], [[RMO_D6]], [[RMO_D7]]], f32>) -> !mo.tensor<[[SH1b]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1b]]> [[V2]] : !mo.tensor<[[SH1b]], f32>

// -----

mo.graph @pad_reflect_params_mix<D0, D1, D2 -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[4, D0, D1], f32>, %pads: !mo.tensor<[D2], si64>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.pad.reflect<() -> Sh1: !mosh.ape>(%arg0, %pads) : (!mo.tensor<[4, D0, D1], f32>, !mo.tensor<[D2], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}

// CHECK-LABEL: mo.graph
//  CHECK-SAME: @pad_reflect_params_mix<[[D0:[A-Za-z0-9_]*]], [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]] -> [[SH0:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[4, [[D0]], [[D1]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[[SH0]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[4, [[D0]], [[D1]]], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape([[ARG1]]) {{.*}} : {{.*}} -> !mo.tensor<[6], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.pad.reflect<() -> [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]]>([[V0]], [[V1]]) : {{.*}} -> !mo.tensor<[[[RMO_D6]], [[RMO_D7]], [[RMO_D8]]], f32>

//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH0b:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D6]], [[RMO_D7]], [[RMO_D8]]], f32>) -> !mo.tensor<[[SH0b]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH0b]]> [[V3]] : !mo.tensor<[[SH0b]], f32>

// -----

mo.graph @pad_reflect_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[4, 5, 6], f32>, %pads: !mo.tensor<[6], si64>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.pad.reflect<() -> Sh1: !mosh.ape>(%arg0, %pads) : (!mo.tensor<[4, 5, 6], f32>, !mo.tensor<[6], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @pad_reflect_static<() -> [[SH0:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[4, 5, 6], f32>, [[ARG1:%.*]]: !mo.tensor<[6], si64>) -> !mo.tensor<[[SH0]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[4, 5, 6], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.pad.reflect<() -> [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]]>([[V0]], [[ARG1]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32>, !mo.tensor<[6], si64>) -> !mo.tensor<[[[RMO_D4]], [[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH0b:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D4]], [[RMO_D5]], [[RMO_D6]]], f32>) -> !mo.tensor<[[SH0b]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH0b]]> [[V2]] : !mo.tensor<[[SH0b]], f32>

// -----

mo.graph @pad_repeat_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %pads: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.pad.repeat<() -> Sh3: !mosh.ape>(%arg0, %pads) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @pad_repeat_dyn<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[SH2]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[RMO_SH3]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[[RMO_D4]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.pad.repeat<() -> [[RMO_SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]], [[V1]]) : {{.*}} -> !mo.tensor<[[RMO_SH5]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2b:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[RMO_SH5]], f32>) -> !mo.tensor<[[SH2b]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2b]]> [[V3]] : !mo.tensor<[[SH2b]], f32>

// -----

mo.graph @pad_repeat_dyn_out_params<Sh0: !mosh.ape, Sh1: !mosh.ape -> D0, D1, D2>(%arg0: !mo.tensor<Sh0, f32>, %pads: !mo.tensor<Sh1, si64>) -> !mo.tensor<[D0, D1, D2], f32> {
  %0 = rmo.mo.pad.repeat<() -> D3, D4, D5>(%arg0, %pads) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<[D3, D4, D5], f32>
  mo.output<D3, D4, D5> %0 : !mo.tensor<[D3, D4, D5], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @pad_repeat_dyn_out_params<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape -> [[D0:[A-Za-z0-9_]*]], [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], si64>) -> !mo.tensor<[6], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.pad.repeat<() -> [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]]>([[V0]], [[V1]]) : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>, !mo.tensor<[6], si64>) -> !mo.tensor<[[[RMO_D6]], [[RMO_D7]], [[RMO_D8]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[D0b:[A-Za-z0-9_]*]], [[D1b:[A-Za-z0-9_]*]], [[D2b:[A-Za-z0-9_]*]]>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D6]], [[RMO_D7]], [[RMO_D8]]], f32>) -> !mo.tensor<[[[D0b]], [[D1b]], [[D2b]]], f32>
//  CHECK-NEXT:   mo.output<[[D0b]], [[D1b]], [[D2b]]> [[V3]] : !mo.tensor<[[[D0b]], [[D1b]], [[D2b]]], f32>

// -----

mo.graph @pad_constant_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %pads: !mo.tensor<Sh1, si64>, %cst: !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.pad.constant<() -> Sh4: !mosh.ape>(%arg0, %pads, %cst) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %0 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @pad_constant_dyn<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], si64>, [[ARG2:%.*]]: !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[RMO_SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[[RMO_D5]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[ARG2]]) {{.*}} : (!mo.tensor<[[SH2]], f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   [[V3:%.*]] = mo.pad.constant<() -> [[RMO_SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]], [[V1]], [[V2]]) : {{.*}} -> !mo.tensor<[[RMO_SH6]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH3b:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[RMO_SH6]], f32>) -> !mo.tensor<[[SH3b]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3b]]> [[V4]] : !mo.tensor<[[SH3b]], f32>
