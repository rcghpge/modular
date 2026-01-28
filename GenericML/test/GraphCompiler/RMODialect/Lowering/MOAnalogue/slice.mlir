// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @slice_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape -> Sh4: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %start: !mo.tensor<Sh1, si64>, %stop: !mo.tensor<Sh2, si32>, %step: !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32> {
  %0 = rmo.mo.slice<() -> Sh5: !mosh.ape>(%arg0, %start, %stop, %step) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si32>, !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh5, f32>
  mo.output<:!mosh.ape Sh5> %0 : !mo.tensor<Sh5, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @slice_dyn<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], si64>, [[ARG2:%.*]]: !mo.tensor<[[SH2]], si32>, [[ARG3:%.*]]: !mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[RMO_SH1]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[[RMO_D2]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[ARG2]]) {{.*}} : (!mo.tensor<[[SH2]], si32>) -> !mo.tensor<[[[RMO_D2]]], si32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[ARG3]]) {{.*}} : (!mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[[RMO_D2]]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.slice<() -> [[RMO_SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]], [[V1]], [[V2]], [[V3]]) : (!mo.tensor<[[RMO_SH1]], f32>, !mo.tensor<[[[RMO_D2]]], si64>, !mo.tensor<[[[RMO_D2]]], si32>, !mo.tensor<[[[RMO_D2]]], si64>) -> !mo.tensor<[[RMO_SH3]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[RMO_SH3]], f32>) -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH5]]> [[V5]] : !mo.tensor<[[SH5]], f32>

// -----

mo.graph @slice_dyn_params<D0, D1, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape -> Sh5: !mosh.ape>(%arg0: !mo.tensor<[D0, D1], f32>, %start: !mo.tensor<Sh2, si64>, %stop: !mo.tensor<Sh3, si32>, %step: !mo.tensor<Sh4, si64>) -> !mo.tensor<Sh5, f32> {
  %0 = rmo.mo.slice<() -> Sh6: !mosh.ape>(%arg0, %start, %stop, %step) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, si32>, !mo.tensor<Sh4, si64>) -> !mo.tensor<Sh6, f32>
  mo.output<:!mosh.ape Sh6> %0 : !mo.tensor<Sh6, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @slice_dyn_params<[[D0:[A-Za-z0-9_]*]], [[D1:[A-Za-z0-9_]*]], [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si32>, [[ARG3:%.*]]: !mo.tensor<[[SH4]], si64>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[[D0]], [[D1]]], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape([[ARG1]]) {{.*}} : (!mo.tensor<[[SH2]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[ARG2]]) {{.*}} : (!mo.tensor<[[SH3]], si32>) -> !mo.tensor<[2], si32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[ARG3]]) {{.*}} : (!mo.tensor<[[SH4]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.slice<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[V0]], [[V1]], [[V2]], [[V3]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V5]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @slice_params<D0, D1, D2, D3, D4 -> Sh5: !mosh.ape>(%arg0: !mo.tensor<[D0, D1], f32>, %start: !mo.tensor<[D2], si64>, %stop: !mo.tensor<[D3], si32>, %step: !mo.tensor<[D4], si64>) -> !mo.tensor<Sh5, f32> {
  %0 = rmo.mo.slice<() -> Sh6: !mosh.ape>(%arg0, %start, %stop, %step) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2], si64>, !mo.tensor<[D3], si32>, !mo.tensor<[D4], si64>) -> !mo.tensor<Sh6, f32>
  mo.output<:!mosh.ape Sh6> %0 : !mo.tensor<Sh6, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @slice_params<[[D0:[A-Za-z0-9_]*]], [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]] -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D2]]], si64>, [[ARG2:%.*]]: !mo.tensor<[[[D3]]], si32>, [[ARG3:%.*]]: !mo.tensor<[[[D4]]], si64>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[[D0]], [[D1]]], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape([[ARG1]]) {{.*}} : (!mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[ARG2]]) {{.*}} : (!mo.tensor<[[[D3]]], si32>) -> !mo.tensor<[2], si32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[ARG3]]) {{.*}} : (!mo.tensor<[[[D4]]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.slice<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[V0]], [[V1]], [[V2]], [[V3]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V5]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @slice_params_mix<D0, D1, D2 -> Sh3: !mosh.ape>(%arg0: !mo.tensor<[3, D0], f32>, %start: !mo.tensor<[D1], si64>, %stop: !mo.tensor<[2], si32>, %step: !mo.tensor<[D2], si64>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.slice<() -> Sh4: !mosh.ape>(%arg0, %start, %stop, %step) : (!mo.tensor<[3, D0], f32>, !mo.tensor<[D1], si64>, !mo.tensor<[2], si32>, !mo.tensor<[D2], si64>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %0 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @slice_params_mix<[[D0:[A-Za-z0-9_]*]], [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]] -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[3, [[D0]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D1]]], si64>, [[ARG2:%.*]]: !mo.tensor<[2], si32>, [[ARG3:%.*]]: !mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[3, [[D0]]], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape([[ARG1]]) {{.*}} : (!mo.tensor<[[[D1]]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[ARG3]]) {{.*}} : (!mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.slice<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[V0]], [[V1]], [[ARG2]], [[V2]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V4]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @slice_dyn_mix<Sh0: !mosh.ape, D1, D2 -> Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %start: !mo.tensor<[D1], si64>, %stop: !mo.tensor<[2], si32>, %step: !mo.tensor<[D2], si64>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.slice<() -> Sh4: !mosh.ape>(%arg0, %start, %stop, %step) : (!mo.tensor<Sh0, f32>, !mo.tensor<[D1], si64>, !mo.tensor<[2], si32>, !mo.tensor<[D2], si64>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %0 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @slice_dyn_mix<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]] -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D1]]], si64>, [[ARG2:%.*]]: !mo.tensor<[2], si32>, [[ARG3:%.*]]: !mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape([[ARG1]]) {{.*}} : (!mo.tensor<[[[D1]]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[ARG3]]) {{.*}} : (!mo.tensor<[[[D2]]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.slice<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[V0]], [[V1]], [[ARG2]], [[V2]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V4]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @slice_dyn_out_params<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape -> D4, D5>(%arg0: !mo.tensor<Sh0, f32>, %start: !mo.tensor<Sh1, si64>, %stop: !mo.tensor<Sh2, si32>, %step: !mo.tensor<Sh3, si64>) -> !mo.tensor<[D4, D5], f32> {
  %0 = rmo.mo.slice<() -> D6, D7>(%arg0, %start, %stop, %step) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si32>, !mo.tensor<Sh3, si64>) -> !mo.tensor<[D6, D7], f32>
  mo.output<D6, D7> %0 : !mo.tensor<[D6, D7], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @slice_dyn_out_params<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape -> [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], si64>, [[ARG2:%.*]]: !mo.tensor<[[SH2]], si32>, [[ARG3:%.*]]: !mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[[D4]], [[D5]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[ARG2]]) {{.*}} : (!mo.tensor<[[SH2]], si32>) -> !mo.tensor<[2], si32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[ARG3]]) {{.*}} : (!mo.tensor<[[SH3]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.slice<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[V0]], [[V1]], [[V2]], [[V3]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]]>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[[D6]], [[D7]]], f32>
//  CHECK-NEXT:   mo.output<[[D6]], [[D7]]> [[V5]] : !mo.tensor<[[[D6]], [[D7]]], f32>

// -----

mo.graph @slice_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[3, 4], f32>, %start: !mo.tensor<[2], si64>, %stop: !mo.tensor<[2], si32>, %step: !mo.tensor<[2], si64>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.slice<() -> Sh1: !mosh.ape>(%arg0, %start, %stop, %step) : (!mo.tensor<[3, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @slice_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[3, 4], f32>, [[ARG1:%.*]]: !mo.tensor<[2], si64>, [[ARG2:%.*]]: !mo.tensor<[2], si32>, [[ARG3:%.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[3, 4], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.slice<() -> [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]]>([[V0]], [[ARG1]], [[ARG2]], [[ARG3]]) : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V2]] : !mo.tensor<[[SH2]], f32>
