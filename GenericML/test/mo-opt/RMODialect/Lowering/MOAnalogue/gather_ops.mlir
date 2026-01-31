// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @gather_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %indices: !mo.tensor<Sh1, si32>, %axis: !mo.tensor<Sh2, si32>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.gather<() -> Sh4: !mosh.ape>(%input, %indices, %axis) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si32>, !mo.tensor<Sh2, si32>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %0 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @gather_dyn<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], si32>, [[ARG2:%.*]]: !mo.tensor<[[SH2]], si32>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape([[ARG2]]) {{.*}} : (!mo.tensor<[[SH2]], si32>) -> !mo.tensor<[], si32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.gather<() -> [[RMO_SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]], [[ARG1]], [[V0]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH1]], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[[RMO_SH1]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[RMO_SH1]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V2]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @gather_dyn_dims<D0, D1, D2, D3, D4 -> Sh5: !mosh.ape>(%input: !mo.tensor<[D0, D1, D2], f32>, %indices: !mo.tensor<[D3, D4], si32>, %axis: !mo.tensor<[], si32>) -> !mo.tensor<Sh5, f32> {
  %0 = rmo.mo.gather<() -> Sh6: !mosh.ape>(%input, %indices, %axis) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[D3, D4], si32>, !mo.tensor<[], si32>) -> !mo.tensor<Sh6, f32>
  mo.output<:!mosh.ape Sh6> %0 : !mo.tensor<Sh6, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @gather_dyn_dims<[[D0:[A-Za-z0-9_]*]], [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]] -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D3]], [[D4]]], si32>, [[ARG2:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.gather<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]], [[ARG2]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, !mo.tensor<[[[D3]], [[D4]]], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V1]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @gather_mix_dyn_dims<D0, D1, D2 -> Sh3: !mosh.ape>(%input: !mo.tensor<[8, D0, D1], f32>, %indices: !mo.tensor<[D2, 3], si32>, %axis: !mo.tensor<[], si32>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.gather<() -> Sh4: !mosh.ape>(%input, %indices, %axis) : (!mo.tensor<[8, D0, D1], f32>, !mo.tensor<[D2, 3], si32>, !mo.tensor<[], si32>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %0 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @gather_mix_dyn_dims<[[D0:[A-Za-z0-9_]*]], [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]] -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[8, [[D0]], [[D1]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D2]], 3], si32>, [[ARG2:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.gather<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]], [[ARG2]]) : (!mo.tensor<[8, [[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], 3], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V1]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @gather_static<() -> Sh0: !mosh.ape>(%input: !mo.tensor<[12, 2], f32>, %indices: !mo.tensor<[128], si32>, %axis: !mo.tensor<[], si32>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.gather<() -> Sh1: !mosh.ape>(%input, %indices, %axis) : (!mo.tensor<[12, 2], f32>, !mo.tensor<[128], si32>, !mo.tensor<[], si32>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @gather_static<() -> [[SH0:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[12, 2], f32>, [[ARG1:%.*]]: !mo.tensor<[128], si32>, [[ARG2:%.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[[SH0]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.gather<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]], [[ARG2]]) : (!mo.tensor<[12, 2], f32>, !mo.tensor<[128], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V1]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @gather_nd_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %indices: !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.gather_nd<() -> Sh3: !mosh.ape>(%input, %indices) {batchDims = 1 : index} : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @gather_nd_dyn<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], si32>) -> !mo.tensor<[[SH2]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.gather_nd<() -> [[RMO_SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]], [[ARG1]]) {[[BATCHDIMS:[A-Za-z0-9_]*]] = 1 : index} : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH1]], si32>) -> !mo.tensor<[[RMO_SH1]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[RMO_SH1]], f32>) -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> [[V1]] : !mo.tensor<[[SH3]], f32>

// -----

mo.graph @gather_nd_dyn_dims<D0, D1, D2, D3 -> Sh4: !mosh.ape>(%input: !mo.tensor<[D0, D1], f32>, %indices: !mo.tensor<[D2, D3], si32>) -> !mo.tensor<Sh4, f32> {
  %0 = rmo.mo.gather_nd<() -> Sh5: !mosh.ape>(%input, %indices) {batchDims = 1 : index} : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], si32>) -> !mo.tensor<Sh5, f32>
  mo.output<:!mosh.ape Sh5> %0 : !mo.tensor<Sh5, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @gather_nd_dyn_dims<[[D0:[A-Za-z0-9_]*]], [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]] -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D2]], [[D3]]], si32>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.gather_nd<() -> [[RMO_SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]], [[ARG1]]) {[[BATCHDIMS:[A-Za-z0-9_]*]] = 1 : index} : (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], si32>) -> !mo.tensor<[[RMO_SH1]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[RMO_SH1]], f32>) -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH5]]> [[V1]] : !mo.tensor<[[SH5]], f32>

// -----

mo.graph @gather_nd_static<() -> Sh0: !mosh.ape>(%input: !mo.tensor<[2, 2, 4], f32>, %indices: !mo.tensor<[3], si32>) -> !mo.tensor<Sh0, f32> {
   %0 = rmo.mo.gather_nd<() -> Sh1: !mosh.ape>(%input, %indices) {batchDims = 1 : index} : (!mo.tensor<[2, 2, 4], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<Sh1, f32>
   mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// COM: output is rank 0
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @gather_nd_static<() -> [[SH0:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, 2, 4], f32>, [[ARG1:%.*]]: !mo.tensor<[3], si32>) -> !mo.tensor<[[SH0]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.gather_nd([[ARG0]], [[ARG1]]) {[[BATCHDIMS:[A-Za-z0-9_]*]] = 1 : index} : (!mo.tensor<[2, 2, 4], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V1]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @gather_nd_static_2<() -> Sh0: !mosh.ape>(%input: !mo.tensor<[2, 2, 4], f32>, %indices: !mo.tensor<[4, 3], si32>) -> !mo.tensor<Sh0, f32> {
   %0 = rmo.mo.gather_nd<() -> Sh1: !mosh.ape>(%input, %indices) {batchDims = 0 : index} : (!mo.tensor<[2, 2, 4], f32>, !mo.tensor<[4, 3], si32>) -> !mo.tensor<Sh1, f32>
   mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// COM: output is rank 1
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @gather_nd_static_2<() -> [[SH0:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, 2, 4], f32>, [[ARG1:%.*]]: !mo.tensor<[4, 3], si32>) -> !mo.tensor<[[SH0]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.gather_nd<() -> [[RMO_D1:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) {[[BATCHDIMS:[A-Za-z0-9_]*]] = 0 : index} : (!mo.tensor<[2, 2, 4], f32>, !mo.tensor<[4, 3], si32>) -> !mo.tensor<[[[RMO_D1]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]]], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V1]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @gather_nd_static_3<() -> Sh0: !mosh.ape>(%input: !mo.tensor<[2, 2, 1], f32>, %indices: !mo.tensor<[2, 1], si32>) -> !mo.tensor<Sh0, f32> {
   %0 = rmo.mo.gather_nd<() -> Sh1: !mosh.ape>(%input, %indices) {batchDims = 1 : index} : (!mo.tensor<[2, 2, 1], f32>, !mo.tensor<[2, 1], si32>) -> !mo.tensor<Sh1, f32>
   mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// COM: output is rank 2
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @gather_nd_static_3<() -> [[SH0:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, 2, 1], f32>, [[ARG1:%.*]]: !mo.tensor<[2, 1], si32>) -> !mo.tensor<[[SH0]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.gather_nd<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) {[[BATCHDIMS:[A-Za-z0-9_]*]] = 1 : index} : (!mo.tensor<[2, 2, 1], f32>, !mo.tensor<[2, 1], si32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V1]] : !mo.tensor<[[SH1]], f32>
