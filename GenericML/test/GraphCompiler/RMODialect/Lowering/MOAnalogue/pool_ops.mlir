// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @max_pool_dyn(%arg0: !mo.tensor<?, f32>, %fs: !mo.tensor<?, si64>, %st: !mo.tensor<?, si64>, %di: !mo.tensor<?, si64>, %pa: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @max_pool_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape, [[SH5:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>, [[ARG3:%.*]]: !mo.tensor<[[SH4]], si64>, [[ARG4:%.*]]: !mo.tensor<[[SH5]], si64>) -> !mo.tensor<[[SH6]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH7]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH8]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH9]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG3]]) : !mo.tensor<[[SH4]], si64> -> !mo.tensor<[[SH10]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.rebind<() -> [[SH11:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG4]]) : !mo.tensor<[[SH5]], si64> -> !mo.tensor<[[SH11]], si64>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D13:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH8]], si64>) -> !mo.tensor<[[[RMO_D13]]], si64>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH9]], si64>) -> !mo.tensor<[[[RMO_D13]]], si64>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH10]], si64>) -> !mo.tensor<[[[RMO_D13]]], si64>
//  CHECK-NEXT:   [[V8:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D14:[A-Za-z0-9_]*]]>([[V4]]) {{.*}} : (!mo.tensor<[[SH11]], si64>) -> !mo.tensor<[[[RMO_D14]]], si64>
//  CHECK-NEXT:   [[V9:%.*]] = mo.max_pool<() -> [[RMO_SH15:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) [filter_shape = [[V5]], strides = [[V6]], dilations = [[V7]], paddings = [[V8]]] : (!mo.tensor<[[SH7]], f32>, !mo.tensor<[[[RMO_D13]]], si64>, !mo.tensor<[[[RMO_D13]]], si64>, !mo.tensor<[[[RMO_D13]]], si64>, !mo.tensor<[[[RMO_D14]]], si64>) -> !mo.tensor<[[RMO_SH15]], f32>
//  CHECK-NEXT:   [[V10:%.*]] = rmo.rebind_tensor_shape<() -> [[SH12:[A-Za-z0-9_]*]]: !mosh.ape>([[V9]]) {{.*}} : (!mo.tensor<[[RMO_SH15]], f32>) -> !mo.tensor<[[SH12]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH12]]> [[V10]] : !mo.tensor<[[SH12]], f32>

// -----

mo.graph @max_pool_dyn_dims(%arg0: !mo.tensor<[?, ?, ?, ?], f32>, %fs: !mo.tensor<[?], si64>, %st: !mo.tensor<[?], si64>, %di: !mo.tensor<[?], si64>, %pa: !mo.tensor<[?], si64>) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %0 = rmo.mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[?], si64>, !mo.tensor<[?], si64>, !mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @max_pool_dyn_dims<D1, D2, D3, D4, D5, D6, D7, D8 -> D9, D10, D11, D12>([[ARG0:%.*]]: !mo.tensor<[D1, D2, D3, D4], f32>, [[ARG1:%.*]]: !mo.tensor<[D5], si64>, [[ARG2:%.*]]: !mo.tensor<[D6], si64>, [[ARG3:%.*]]: !mo.tensor<[D7], si64>, [[ARG4:%.*]]: !mo.tensor<[D8], si64>) -> !mo.tensor<[D9, D10, D11, D12], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D13:[A-Za-z0-9_]*]], [[D14:[A-Za-z0-9_]*]], [[D15:[A-Za-z0-9_]*]], [[D16:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[D1, D2, D3, D4], f32> -> !mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D17:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[D5], si64> -> !mo.tensor<[[[D17]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[D18:[A-Za-z0-9_]*]]>([[ARG2]]) : !mo.tensor<[D6], si64> -> !mo.tensor<[[[D18]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[D19:[A-Za-z0-9_]*]]>([[ARG3]]) : !mo.tensor<[D7], si64> -> !mo.tensor<[[[D19]]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = mo.rebind<() -> [[D20:[A-Za-z0-9_]*]]>([[ARG4]]) : !mo.tensor<[D8], si64> -> !mo.tensor<[[[D20]]], si64>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[[D18]]], si64>) -> !mo.tensor<[[[D17]]], si64>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D19]]], si64>) -> !mo.tensor<[[[D17]]], si64>
//  CHECK-NEXT:   [[V7:%.*]] = mo.max_pool<() -> [[RMO_D25:[A-Za-z0-9_]*]], [[RMO_D26:[A-Za-z0-9_]*]], [[RMO_D27:[A-Za-z0-9_]*]], [[RMO_D28:[A-Za-z0-9_]*]]>([[V0]]) [filter_shape = [[V1]], strides = [[V5]], dilations = [[V6]], paddings = [[V4]]] : (!mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], f32>, !mo.tensor<[[[D17]]], si64>, !mo.tensor<[[[D17]]], si64>, !mo.tensor<[[[D17]]], si64>, !mo.tensor<[[[D20]]], si64>) -> !mo.tensor<[[[RMO_D25]], [[RMO_D26]], [[RMO_D27]], [[RMO_D28]]], f32>
//  CHECK-NEXT:   [[V8:%.*]] = rmo.rebind_tensor_shape<() -> [[D21:[A-Za-z0-9_]*]], [[D22:[A-Za-z0-9_]*]], [[D23:[A-Za-z0-9_]*]], [[D24:[A-Za-z0-9_]*]]>([[V7]]) {{.*}} : (!mo.tensor<[[[RMO_D25]], [[RMO_D26]], [[RMO_D27]], [[RMO_D28]]], f32>) -> !mo.tensor<[[[D21]], [[D22]], [[D23]], [[D24]]], f32>
//  CHECK-NEXT:   mo.output<[[D21]], [[D22]], [[D23]], [[D24]]> [[V8]] : !mo.tensor<[[[D21]], [[D22]], [[D23]], [[D24]]], f32>

// -----

mo.graph @max_pool_static(%arg0: !mo.tensor<[3, 8, 43, 49], f32>, %fs: !mo.tensor<[2], si64>, %st: !mo.tensor<[2], si64>, %di: !mo.tensor<[2], si64>, %pa: !mo.tensor<[4], si64>) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %0 = rmo.mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<[3, 8, 43, 49], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @max_pool_static<() -> [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[3, 8, 43, 49], f32>, [[ARG1:%.*]]: !mo.tensor<[2], si64>, [[ARG2:%.*]]: !mo.tensor<[2], si64>, [[ARG3:%.*]]: !mo.tensor<[2], si64>, [[ARG4:%.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.max_pool<() -> [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]], [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]]>([[ARG0]]) [filter_shape = [[ARG1]], strides = [[ARG2]], dilations = [[ARG3]], paddings = [[ARG4]]] : (!mo.tensor<[3, 8, 43, 49], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[[[RMO_D9]], [[RMO_D10]], [[RMO_D11]], [[RMO_D12]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[D5:[A-Za-z0-9_]*]], [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]], [[D8:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]], [[RMO_D11]], [[RMO_D12]]], f32>) -> !mo.tensor<[[[D5]], [[D6]], [[D7]], [[D8]]], f32>
//  CHECK-NEXT:   mo.output<[[D5]], [[D6]], [[D7]], [[D8]]> [[V1]] : !mo.tensor<[[[D5]], [[D6]], [[D7]], [[D8]]], f32>

// -----

mo.graph @max_pool_static_dyn(%arg0: !mo.tensor<[3, ?, 43, ?], f32>, %fs: !mo.tensor<[?], si64>, %st: !mo.tensor<[?], si64>, %di: !mo.tensor<[?], si64>, %pa: !mo.tensor<[4], si64>) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %0 = rmo.mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<[3, ?, 43, ?], f32>, !mo.tensor<[?], si64>, !mo.tensor<[?], si64>, !mo.tensor<[?], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @max_pool_static_dyn<D1, D2, D3, D4, D5 -> D6, D7, D8, D9>([[ARG0:%.*]]: !mo.tensor<[3, D1, 43, D2], f32>, [[ARG1:%.*]]: !mo.tensor<[D3], si64>, [[ARG2:%.*]]: !mo.tensor<[D4], si64>, [[ARG3:%.*]]: !mo.tensor<[D5], si64>, [[ARG4:%.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<[D6, D7, D8, D9], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[3, D1, 43, D2], f32> -> !mo.tensor<[3, [[D10]], 43, [[D11]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D12:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[D3], si64> -> !mo.tensor<[[[D12]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[D13:[A-Za-z0-9_]*]]>([[ARG2]]) : !mo.tensor<[D4], si64> -> !mo.tensor<[[[D13]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[D14:[A-Za-z0-9_]*]]>([[ARG3]]) : !mo.tensor<[D5], si64> -> !mo.tensor<[[[D14]]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[[D13]]], si64>) -> !mo.tensor<[[[D12]]], si64>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D14]]], si64>) -> !mo.tensor<[[[D12]]], si64>
//  CHECK-NEXT:   [[V6:%.*]] = mo.max_pool<() -> [[RMO_D19:[A-Za-z0-9_]*]], [[RMO_D20:[A-Za-z0-9_]*]], [[RMO_D21:[A-Za-z0-9_]*]], [[RMO_D22:[A-Za-z0-9_]*]]>([[V0]]) [filter_shape = [[V1]], strides = [[V4]], dilations = [[V5]], paddings = [[ARG4]]] : (!mo.tensor<[3, [[D10]], 43, [[D11]]], f32>, !mo.tensor<[[[D12]]], si64>, !mo.tensor<[[[D12]]], si64>, !mo.tensor<[[[D12]]], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[[[RMO_D19]], [[RMO_D20]], [[RMO_D21]], [[RMO_D22]]], f32>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape<() -> [[D15:[A-Za-z0-9_]*]], [[D16:[A-Za-z0-9_]*]], [[D17:[A-Za-z0-9_]*]], [[D18:[A-Za-z0-9_]*]]>([[V6]]) {{.*}} : (!mo.tensor<[[[RMO_D19]], [[RMO_D20]], [[RMO_D21]], [[RMO_D22]]], f32>) -> !mo.tensor<[[[D15]], [[D16]], [[D17]], [[D18]]], f32>
//  CHECK-NEXT:   mo.output<[[D15]], [[D16]], [[D17]], [[D18]]> [[V7]] : !mo.tensor<[[[D15]], [[D16]], [[D17]], [[D18]]], f32>

// -----

mo.graph @max_pool_static_dyn_view(%arg0: !mo.tensor<[3, ?, 43, ?], f32>, %fs: !mo.tensor<?, si64>, %st: !mo.tensor<[?], si64>, %di: !mo.tensor<[2], si64>, %pa: !mo.tensor<[4], si64>) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %0 = rmo.mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<[3, ?, 43, ?], f32>, !mo.tensor<?, si64>, !mo.tensor<[?], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @max_pool_static_dyn_view<D1, D2, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, D4 -> D5, D6, D7, D8>([[ARG0:%.*]]: !mo.tensor<[3, D1, 43, D2], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH3]], si64>, [[ARG2:%.*]]: !mo.tensor<[D4], si64>, [[ARG3:%.*]]: !mo.tensor<[2], si64>, [[ARG4:%.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<[D5, D6, D7, D8], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[3, D1, 43, D2], f32> -> !mo.tensor<[3, [[D9]], 43, [[D10]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH11:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH11]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[D12:[A-Za-z0-9_]*]]>([[ARG2]]) : !mo.tensor<[D4], si64> -> !mo.tensor<[[[D12]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH11]], si64>) -> !mo.tensor<[[[D12]]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape([[ARG3]]) {{.*}} : (!mo.tensor<[2], si64>) -> !mo.tensor<[[[D12]]], si64>
//  CHECK-NEXT:   [[V5:%.*]] = mo.max_pool<() -> [[RMO_D17:[A-Za-z0-9_]*]], [[RMO_D18:[A-Za-z0-9_]*]], [[RMO_D19:[A-Za-z0-9_]*]], [[RMO_D20:[A-Za-z0-9_]*]]>([[V0]]) [filter_shape = [[V3]], strides = [[V2]], dilations = [[V4]], paddings = [[ARG4]]] : (!mo.tensor<[3, [[D9]], 43, [[D10]]], f32>, !mo.tensor<[[[D12]]], si64>, !mo.tensor<[[[D12]]], si64>, !mo.tensor<[[[D12]]], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[[[RMO_D17]], [[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], f32>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape<() -> [[D13:[A-Za-z0-9_]*]], [[D14:[A-Za-z0-9_]*]], [[D15:[A-Za-z0-9_]*]], [[D16:[A-Za-z0-9_]*]]>([[V5]]) {{.*}} : (!mo.tensor<[[[RMO_D17]], [[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], f32>) -> !mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], f32>
//  CHECK-NEXT:   mo.output<[[D13]], [[D14]], [[D15]], [[D16]]> [[V6]] : !mo.tensor<[[[D13]], [[D14]], [[D15]], [[D16]]], f32>
