// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @conv_transpose_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape -> Sh6: !mosh.ape>(%input : !mo.tensor<Sh0, f32>, %filter : !mo.tensor<Sh1, f32>, %st: !mo.tensor<Sh2, si64>, %di: !mo.tensor<Sh3, si64>, %pa: !mo.tensor<Sh4, si64>, %op: !mo.tensor<Sh5, si64>) -> (!mo.tensor<Sh6, f32>) {
  %0 = rmo.mo.conv_transpose<() -> Sh7: !mosh.ape>(%input, %filter) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, si64>, !mo.tensor<Sh4, si64>, !mo.tensor<Sh5, si64>
  ) -> !mo.tensor<Sh7, f32>
  mo.output<:!mosh.ape Sh7> %0 : !mo.tensor<Sh7, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @conv_transpose_dyn<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape, [[SH5:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH2]], si64>, [[ARG3:%.*]]: !mo.tensor<[[SH3]], si64>, [[ARG4:%.*]]: !mo.tensor<[[SH4]], si64>, [[ARG5:%.*]]: !mo.tensor<[[SH5]], si64>) -> !mo.tensor<[[SH6]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]]>([[ARG2]]) {{.*}} : (!mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape([[ARG3]]) {{.*}} : (!mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG4]]) {{.*}} : (!mo.tensor<[[SH4]], si64>) -> !mo.tensor<[[[RMO_D2]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[ARG5]]) {{.*}} : (!mo.tensor<[[SH5]], si64>) -> !mo.tensor<[[[RMO_D1]]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.conv_transpose<() -> [[RMO_SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]], [[ARG1]]) {filter_layout = ""} [strides = [[V0]], dilations = [[V1]], paddings = [[V2]], output_paddings = [[V3]]] : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH1]], f32>, !mo.tensor<[[[RMO_D1]]], si64>, !mo.tensor<[[[RMO_D1]]], si64>, !mo.tensor<[[[RMO_D2]]], si64>, !mo.tensor<[[[RMO_D1]]], si64>) -> !mo.tensor<[[RMO_SH3]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[RMO_SH3]], f32>) -> !mo.tensor<[[SH7]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH7]]> [[V5]] : !mo.tensor<[[SH7]], f32>

// -----

mo.graph @conv_transpose_dyn_params<D0, D1, D2, D3, D4, D5, D6, D7, D8, D9 -> Sh10: !mosh.ape>(%input : !mo.tensor<[D0, D1, D2], f32>, %filter : !mo.tensor<[D3, D4, D5], f32>, %st: !mo.tensor<[D6], si64>, %di: !mo.tensor<[D7], si64>, %pa: !mo.tensor<[D8], si64>, %op: !mo.tensor<[D9], si64>) -> (!mo.tensor<Sh10, f32>) {
  %0 = rmo.mo.conv_transpose<() -> Sh11: !mosh.ape>(%input, %filter) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[D3, D4, D5], f32>, !mo.tensor<[D6], si64>, !mo.tensor<[D7], si64>, !mo.tensor<[D8], si64>, !mo.tensor<[D9], si64>
  ) -> !mo.tensor<Sh11, f32>
  mo.output<:!mosh.ape Sh11> %0 : !mo.tensor<Sh11, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @conv_transpose_dyn_params<D0, D1, D2, D3, D4, D5, D6, D7, D8, D9 -> [[SH:[A-Za-z0-9_]*]]: !mosh.ape>
//  CHECK-SAME: ([[ARG0:%.*]]: !mo.tensor<[D0, D1, D2], f32>, [[ARG1:%.*]]: !mo.tensor<[D3, D4, D5], f32>, [[ARG2:%.*]]: !mo.tensor<[D6], si64>, [[ARG3:%.*]]: !mo.tensor<[D7], si64>, [[ARG4:%.*]]: !mo.tensor<[D8], si64>, [[ARG5:%.*]]: !mo.tensor<[D9], si64>) -> !mo.tensor<[[SH]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape([[ARG3]]) {{.*}} : (!mo.tensor<[D7], si64>) -> !mo.tensor<[D6], si64>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape([[ARG5]]) {{.*}} : (!mo.tensor<[D9], si64>) -> !mo.tensor<[D6], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.conv_transpose<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) {filter_layout = ""} [strides = [[ARG2]], dilations = [[V0]], paddings = [[ARG4]], output_paddings = [[V1]]] : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[D3, D4, D5], f32>, !mo.tensor<[D6], si64>, !mo.tensor<[D6], si64>, !mo.tensor<[D8], si64>, !mo.tensor<[D6], si64>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V3]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @conv_transpose_static<() -> Sh0: !mosh.ape>(%input : !mo.tensor<[3, 12, 17, 19, 16], f32>, %filter : !mo.tensor<[3, 4, 5, 8, 16], f32>, %st: !mo.tensor<[3], si64>, %di: !mo.tensor<[3], si64>, %pa: !mo.tensor<[6], si64>, %op: !mo.tensor<[3], si64>) -> (!mo.tensor<Sh0, f32>) {
  %0 = rmo.mo.conv_transpose<() -> Sh1: !mosh.ape>(%input, %filter) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<[3, 12, 17, 19, 16], f32>, !mo.tensor<[3, 4, 5, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[6], si64>, !mo.tensor<[3], si64>
  ) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @conv_transpose_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[3, 12, 17, 19, 16], f32>, [[ARG1:%.*]]: !mo.tensor<[3, 4, 5, 8, 16], f32>, [[ARG2:%.*]]: !mo.tensor<[3], si64>, [[ARG3:%.*]]: !mo.tensor<[3], si64>, [[ARG4:%.*]]: !mo.tensor<[6], si64>, [[ARG5:%.*]]: !mo.tensor<[3], si64>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.conv_transpose<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) {filter_layout = ""} [strides = [[ARG2]], dilations = [[ARG3]], paddings = [[ARG4]], output_paddings = [[ARG5]]] : (!mo.tensor<[3, 12, 17, 19, 16], f32>, !mo.tensor<[3, 4, 5, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[6], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]], [[RMO_D6]], [[RMO_D7]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]], [[RMO_D6]], [[RMO_D7]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @conv_transpose_static_const_params<() -> Sh0: !mosh.ape>(%input : !mo.tensor<[3, 12, 17, 19, 16], f32>, %filter : !mo.tensor<[3, 4, 5, 8, 16], f32>) -> (!mo.tensor<Sh0, f32>) {
  %st = mo.constant {value = #M.dense_array<2, 3, 2> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %di = mo.constant {value = #M.dense_array<1, 2, 3> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %pa = mo.constant {value = #M.dense_array<1, 1, 3, 3, 2, 2> : tensor<6xsi64>} : !mo.tensor<[6], si64>
  %op = mo.constant {value = #M.dense_array<0, 2, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %0 = rmo.mo.conv_transpose<() -> Sh1: !mosh.ape>(%input, %filter) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<[3, 12, 17, 19, 16], f32>, !mo.tensor<[3, 4, 5, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[6], si64>, !mo.tensor<[3], si64>
  ) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @conv_transpose_static_const_params<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[3, 12, 17, 19, 16], f32>, [[ARG1:%.*]]: !mo.tensor<[3, 4, 5, 8, 16], f32>) -> !mo.tensor<[[SH2]], f32> {
//   CHECK-DAG:   [[V0:%.*]] = mo.constant {value = #M.dense_array<2, 3, 2> : tensor<3xsi64>} : !mo.tensor<[3], si64>
//   CHECK-DAG:   [[V1:%.*]] = mo.constant {value = #M.dense_array<1, 2, 3> : tensor<3xsi64>} : !mo.tensor<[3], si64>
//   CHECK-DAG:   [[V2:%.*]] = mo.constant {value = #M.dense_array<1, 1, 3, 3, 2, 2> : tensor<6xsi64>} : !mo.tensor<[6], si64>
//   CHECK-DAG:   [[V3:%.*]] = mo.constant {value = #M.dense_array<0, 2, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.conv_transpose<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) {filter_layout = ""} [strides = [[V0]], dilations = [[V1]], paddings = [[V2]], output_paddings = [[V3]]] : (!mo.tensor<[3, 12, 17, 19, 16], f32>, !mo.tensor<[3, 4, 5, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[6], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]], [[RMO_D6]], [[RMO_D7]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]], [[RMO_D6]], [[RMO_D7]]], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V5]] : !mo.tensor<[[SH1]], f32>

// -----

// COM: strides / dilations / outputPadding must have the same size (rebound to `[2]`)
mo.graph @conv_transpose_dyn_window_diff<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, D4 -> Sh5: !mosh.ape>(%input : !mo.tensor<Sh0, f32>, %filter : !mo.tensor<Sh1, f32>, %st: !mo.tensor<Sh2, si64>, %di: !mo.tensor<[2], si64>, %pa: !mo.tensor<Sh3, si64>, %op: !mo.tensor<[D4], si64>) -> (!mo.tensor<Sh5, f32>) {
  %0 = rmo.mo.conv_transpose<() -> Sh6: !mosh.ape>(%input, %filter) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<[2], si64>, !mo.tensor<Sh3, si64>, !mo.tensor<[D4], si64>
  ) -> !mo.tensor<Sh6, f32>
  mo.output<:!mosh.ape Sh6> %0 : !mo.tensor<Sh6, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @conv_transpose_dyn_window_diff<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[D4:[A-Za-z0-9_]*]] -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH2]], si64>, [[ARG3:%.*]]: !mo.tensor<[2], si64>, [[ARG4:%.*]]: !mo.tensor<[[SH3]], si64>, [[ARG5:%.*]]: !mo.tensor<{{\[}}[[D4]]{{\]}}, si64>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape([[ARG2]]) {{.*}} : (!mo.tensor<[[SH2]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]]>([[ARG4]]) {{.*}} : (!mo.tensor<[[SH3]], si64>) -> !mo.tensor<{{\[}}[[RMO_D1]]{{\]}}, si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[ARG5]]) {{.*}} : (!mo.tensor<{{\[}}[[D4]]{{\]}}, si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.conv_transpose<() -> [[RMO_SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]], [[ARG1]]) {filter_layout = ""} [strides = [[V0]], dilations = [[ARG3]], paddings = [[V1]], output_paddings = [[V2]]] : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH1]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<{{\[}}[[RMO_D1]]{{\]}}, si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[[RMO_SH2]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[RMO_SH2]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], f32>
