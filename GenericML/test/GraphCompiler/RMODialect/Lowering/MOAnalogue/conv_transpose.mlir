// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @conv_transpose_dyn(%input : !mo.tensor<?, f32>, %filter : !mo.tensor<?, f32>, %st: !mo.tensor<?, si64>, %di: !mo.tensor<?, si64>, %pa: !mo.tensor<?, si64>, %op: !mo.tensor<?, si64>) -> (!mo.tensor<?, f32>) {
  %0 = rmo.mo.conv_transpose(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>
  ) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @conv_transpose_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape, [[SH5:[A-Za-z0-9_]*]]: !mosh.ape, [[SH6:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>, [[ARG3:%.*]]: !mo.tensor<[[SH4]], si64>, [[ARG4:%.*]]: !mo.tensor<[[SH5]], si64>, [[ARG5:%.*]]: !mo.tensor<[[SH6]], si64>) -> !mo.tensor<[[SH7]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH9]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH10]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[SH11:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG3]]) : !mo.tensor<[[SH4]], si64> -> !mo.tensor<[[SH11]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.rebind<() -> [[SH12:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG4]]) : !mo.tensor<[[SH5]], si64> -> !mo.tensor<[[SH12]], si64>
//  CHECK-NEXT:   [[V5:%.*]] = mo.rebind<() -> [[SH13:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG5]]) : !mo.tensor<[[SH6]], si64> -> !mo.tensor<[[SH13]], si64>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D15:[A-Za-z0-9_]*]]>([[V2]]) {{.*}} : (!mo.tensor<[[SH10]], si64>) -> !mo.tensor<[[[RMO_D15]]], si64>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH11]], si64>) -> !mo.tensor<[[[RMO_D15]]], si64>
//  CHECK-NEXT:   [[V8:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D16:[A-Za-z0-9_]*]]>([[V4]]) {{.*}} : (!mo.tensor<[[SH12]], si64>) -> !mo.tensor<[[[RMO_D16]]], si64>
//  CHECK-NEXT:   [[V9:%.*]] = rmo.rebind_tensor_shape([[V5]]) {{.*}} : (!mo.tensor<[[SH13]], si64>) -> !mo.tensor<[[[RMO_D15]]], si64>
//  CHECK-NEXT:   [[V10:%.*]] = mo.conv_transpose<() -> [[RMO_SH17:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]], [[V1]]) [strides = [[V6]], dilations = [[V7]], paddings = [[V8]], output_paddings = [[V9]]] : (!mo.tensor<[[SH8]], f32>, !mo.tensor<[[SH9]], f32>, !mo.tensor<[[[RMO_D15]]], si64>, !mo.tensor<[[[RMO_D15]]], si64>, !mo.tensor<[[[RMO_D16]]], si64>, !mo.tensor<[[[RMO_D15]]], si64>) -> !mo.tensor<[[RMO_SH17]], f32>
//  CHECK-NEXT:   [[V11:%.*]] = rmo.rebind_tensor_shape<() -> [[SH14:[A-Za-z0-9_]*]]: !mosh.ape>([[V10]]) {{.*}} : (!mo.tensor<[[RMO_SH17]], f32>) -> !mo.tensor<[[SH14]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH14]]> [[V11]] : !mo.tensor<[[SH14]], f32>

// -----

mo.graph @conv_transpose_dyn_params(%input : !mo.tensor<[?, ?, ?], f32>, %filter : !mo.tensor<[?, ?, ?], f32>, %st: !mo.tensor<[?], si64>, %di: !mo.tensor<[?], si64>, %pa: !mo.tensor<[?], si64>, %op: !mo.tensor<[?], si64>) -> (!mo.tensor<?, f32>) {
  %0 = rmo.mo.conv_transpose(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?], si64>, !mo.tensor<[?], si64>, !mo.tensor<[?], si64>, !mo.tensor<[?], si64>
  ) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @conv_transpose_dyn_params<D1, D2, D3, D4, D5, D6, D7, D8, D9, D10 -> [[SH11:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[D1, D2, D3], f32>, [[ARG1:%.*]]: !mo.tensor<[D4, D5, D6], f32>, [[ARG2:%.*]]: !mo.tensor<[D7], si64>, [[ARG3:%.*]]: !mo.tensor<[D8], si64>, [[ARG4:%.*]]: !mo.tensor<[D9], si64>, [[ARG5:%.*]]: !mo.tensor<[D10], si64>) -> !mo.tensor<[[SH11]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D12:[A-Za-z0-9_]*]], [[D13:[A-Za-z0-9_]*]], [[D14:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[D1, D2, D3], f32> -> !mo.tensor<[[[D12]], [[D13]], [[D14]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D15:[A-Za-z0-9_]*]], [[D16:[A-Za-z0-9_]*]], [[D17:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[D4, D5, D6], f32> -> !mo.tensor<[[[D15]], [[D16]], [[D17]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[D18:[A-Za-z0-9_]*]]>([[ARG2]]) : !mo.tensor<[D7], si64> -> !mo.tensor<[[[D18]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[D19:[A-Za-z0-9_]*]]>([[ARG3]]) : !mo.tensor<[D8], si64> -> !mo.tensor<[[[D19]]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.rebind<() -> [[D20:[A-Za-z0-9_]*]]>([[ARG4]]) : !mo.tensor<[D9], si64> -> !mo.tensor<[[[D20]]], si64>
//  CHECK-NEXT:   [[V5:%.*]] = mo.rebind<() -> [[D21:[A-Za-z0-9_]*]]>([[ARG5]]) : !mo.tensor<[D10], si64> -> !mo.tensor<[[[D21]]], si64>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D19]]], si64>) -> !mo.tensor<[[[D18]]], si64>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape([[V5]]) {{.*}} : (!mo.tensor<[[[D21]]], si64>) -> !mo.tensor<[[[D18]]], si64>
//  CHECK-NEXT:   [[V8:%.*]] = mo.conv_transpose<() -> [[RMO_D23:[A-Za-z0-9_]*]], [[RMO_D24:[A-Za-z0-9_]*]], [[RMO_D25:[A-Za-z0-9_]*]]>([[V0]], [[V1]]) [strides = [[V2]], dilations = [[V6]], paddings = [[V4]], output_paddings = [[V7]]] : (!mo.tensor<[[[D12]], [[D13]], [[D14]]], f32>, !mo.tensor<[[[D15]], [[D16]], [[D17]]], f32>, !mo.tensor<[[[D18]]], si64>, !mo.tensor<[[[D18]]], si64>, !mo.tensor<[[[D20]]], si64>, !mo.tensor<[[[D18]]], si64>) -> !mo.tensor<[[[RMO_D23]], [[RMO_D24]], [[RMO_D25]]], f32>
//  CHECK-NEXT:   [[V9:%.*]] = rmo.rebind_tensor_shape<() -> [[SH22:[A-Za-z0-9_]*]]: !mosh.ape>([[V8]]) {{.*}} : (!mo.tensor<[[[RMO_D23]], [[RMO_D24]], [[RMO_D25]]], f32>) -> !mo.tensor<[[SH22]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH22]]> [[V9]] : !mo.tensor<[[SH22]], f32>

// -----

mo.graph @conv_transpose_static(%input : !mo.tensor<[3, 12, 17, 19, 16], f32>, %filter : !mo.tensor<[3, 4, 5, 8, 16], f32>, %st: !mo.tensor<[3], si64>, %di: !mo.tensor<[3], si64>, %pa: !mo.tensor<[6], si64>, %op: !mo.tensor<[3], si64>) -> (!mo.tensor<?, f32>) {
  %0 = rmo.mo.conv_transpose(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<[3, 12, 17, 19, 16], f32>, !mo.tensor<[3, 4, 5, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[6], si64>, !mo.tensor<[3], si64>
  ) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @conv_transpose_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[3, 12, 17, 19, 16], f32>, [[ARG1:%.*]]: !mo.tensor<[3, 4, 5, 8, 16], f32>, [[ARG2:%.*]]: !mo.tensor<[3], si64>, [[ARG3:%.*]]: !mo.tensor<[3], si64>, [[ARG4:%.*]]: !mo.tensor<[6], si64>, [[ARG5:%.*]]: !mo.tensor<[3], si64>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.conv_transpose<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) [strides = [[ARG2]], dilations = [[ARG3]], paddings = [[ARG4]], output_paddings = [[ARG5]]] : (!mo.tensor<[3, 12, 17, 19, 16], f32>, !mo.tensor<[3, 4, 5, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[6], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]], [[RMO_D6]], [[RMO_D7]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]], [[RMO_D6]], [[RMO_D7]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @conv_transpose_static_const_params(%input : !mo.tensor<[3, 12, 17, 19, 16], f32>, %filter : !mo.tensor<[3, 4, 5, 8, 16], f32>) -> (!mo.tensor<?, f32>) {
  %st = mo.constant {value = #M.dense_array<2, 3, 2> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %di = mo.constant {value = #M.dense_array<1, 2, 3> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %pa = mo.constant {value = #M.dense_array<1, 1, 3, 3, 2, 2> : tensor<6xsi64>} : !mo.tensor<[6], si64>
  %op = mo.constant {value = #M.dense_array<0, 2, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %0 = rmo.mo.conv_transpose(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<[3, 12, 17, 19, 16], f32>, !mo.tensor<[3, 4, 5, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[6], si64>, !mo.tensor<[3], si64>
  ) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @conv_transpose_static_const_params<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[3, 12, 17, 19, 16], f32>, [[ARG1:%.*]]: !mo.tensor<[3, 4, 5, 8, 16], f32>) -> !mo.tensor<[[SH2]], f32> {
//   CHECK-DAG:   [[V0:%.*]] = mo.constant {value = #M.dense_array<2, 3, 2> : tensor<3xsi64>} : !mo.tensor<[3], si64>
//   CHECK-DAG:   [[V1:%.*]] = mo.constant {value = #M.dense_array<1, 2, 3> : tensor<3xsi64>} : !mo.tensor<[3], si64>
//   CHECK-DAG:   [[V2:%.*]] = mo.constant {value = #M.dense_array<1, 1, 3, 3, 2, 2> : tensor<6xsi64>} : !mo.tensor<[6], si64>
//   CHECK-DAG:   [[V3:%.*]] = mo.constant {value = #M.dense_array<0, 2, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.conv_transpose<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) [strides = [[V0]], dilations = [[V1]], paddings = [[V2]], output_paddings = [[V3]]] : (!mo.tensor<[3, 12, 17, 19, 16], f32>, !mo.tensor<[3, 4, 5, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[6], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]], [[RMO_D6]], [[RMO_D7]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]], [[RMO_D6]], [[RMO_D7]]], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V5]] : !mo.tensor<[[SH1]], f32>

// -----

// COM: strides / dilations / outputPadding must have the same size (rebound to `[2]`)
mo.graph @conv_transpose_dyn_window_diff(%input : !mo.tensor<?, f32>, %filter : !mo.tensor<?, f32>, %st: !mo.tensor<?, si64>, %di: !mo.tensor<[2], si64>, %pa: !mo.tensor<?, si64>, %op: !mo.tensor<[?], si64>) -> (!mo.tensor<?, f32>) {
  %0 = rmo.mo.conv_transpose(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<[2], si64>, !mo.tensor<?, si64>, !mo.tensor<[?], si64>
  ) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @conv_transpose_dyn_window_diff<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape, D5 -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>, [[ARG3:%.*]]: !mo.tensor<[2], si64>, [[ARG4:%.*]]: !mo.tensor<[[SH4]], si64>, [[ARG5:%.*]]: !mo.tensor<[D5], si64>) -> !mo.tensor<[[SH6]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH7]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH9]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG4]]) : !mo.tensor<[[SH4]], si64> -> !mo.tensor<[[SH10]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.rebind<() -> [[D11:[A-Za-z0-9_]*]]>([[ARG5]]) : !mo.tensor<[D5], si64> -> !mo.tensor<[[[D11]]], si64>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH9]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D13:[A-Za-z0-9_]*]]>([[V3]]) {{.*}} : (!mo.tensor<[[SH10]], si64>) -> !mo.tensor<[[[RMO_D13]]], si64>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape([[V4]]) {{.*}} : (!mo.tensor<[[[D11]]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V8:%.*]] = mo.conv_transpose<() -> [[RMO_SH14:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]], [[V1]]) [strides = [[V5]], dilations = [[ARG3]], paddings = [[V6]], output_paddings = [[V7]]] : (!mo.tensor<[[SH7]], f32>, !mo.tensor<[[SH8]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[[[RMO_D13]]], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[[RMO_SH14]], f32>
//  CHECK-NEXT:   [[V9:%.*]] = rmo.rebind_tensor_shape<() -> [[SH12:[A-Za-z0-9_]*]]: !mosh.ape>([[V8]]) {{.*}} : (!mo.tensor<[[RMO_SH14]], f32>) -> !mo.tensor<[[SH12]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH12]]> [[V9]] : !mo.tensor<[[SH12]], f32>
