// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @max_pool_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape -> Sh5: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %fs: !mo.tensor<Sh1, si64>, %st: !mo.tensor<Sh2, si64>, %di: !mo.tensor<Sh3, si64>, %pa: !mo.tensor<Sh4, si64>) -> !mo.tensor<Sh5, f32> {
  %0 = rmo.mo.max_pool<() -> Sh6: !mosh.ape>(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, si64>, !mo.tensor<Sh4, si64>) -> !mo.tensor<Sh6, f32>
  mo.output<:!mosh.ape Sh6> %0 : !mo.tensor<Sh6, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @max_pool_dyn
//  CHECK-SAME: ([[ARG0:%[A-Za-z0-9_]*]]: !mo.tensor<{{[A-Za-z0-9_]+}}, f32>
//  CHECK-SAME: [[ARG1:%[A-Za-z0-9_]*]]: !mo.tensor<{{[A-Za-z0-9_]+}}, si64>
//  CHECK-SAME: [[ARG2:%[A-Za-z0-9_]*]]: !mo.tensor<{{[A-Za-z0-9_]+}}, si64>
//  CHECK-SAME: [[ARG3:%[A-Za-z0-9_]*]]: !mo.tensor<{{[A-Za-z0-9_]+}}, si64>
//  CHECK-SAME: [[ARG4:%[A-Za-z0-9_]*]]: !mo.tensor<{{[A-Za-z0-9_]+}}, si64>)
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape{{.*}}([[ARG1]])
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape([[ARG2]])
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[ARG3]])
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape{{.*}}([[ARG4]])
//  CHECK-NEXT:   [[V4:%.*]] = mo.max_pool{{.*}}([[ARG0]]) [filter_shape = [[V0]], strides = [[V1]], dilations = [[V2]], paddings = [[V3]]]
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape{{.*}}([[V4]])
//  CHECK-NEXT:   mo.output{{.*}} [[V5]]

// -----

mo.graph @max_pool_dyn_dims<D0, D1, D2, D3, D4, D5, D6, D7, D8 -> D9, D10, D11, D12>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>, %fs: !mo.tensor<[D4], si64>, %st: !mo.tensor<[D5], si64>, %di: !mo.tensor<[D6], si64>, %pa: !mo.tensor<[D7], si64>) -> !mo.tensor<[D9, D10, D11, D12], f32> {
  %0 = rmo.mo.max_pool<() -> D13, D14, D15, D16>(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[D4], si64>, !mo.tensor<[D5], si64>, !mo.tensor<[D6], si64>, !mo.tensor<[D7], si64>) -> !mo.tensor<[D13, D14, D15, D16], f32>
  mo.output<D13, D14, D15, D16> %0 : !mo.tensor<[D13, D14, D15, D16], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @max_pool_dyn_dims
//  CHECK-SAME: ([[ARG0:%[A-Za-z0-9_]*]]: !mo.tensor<[{{[A-Za-z0-9_]+}}, {{[A-Za-z0-9_]+}}, {{[A-Za-z0-9_]+}}, {{[A-Za-z0-9_]+}}], f32>
//  CHECK-SAME: [[ARG1:%[A-Za-z0-9_]*]]: !mo.tensor<[{{[A-Za-z0-9_]+}}], si64>
//  CHECK-SAME: [[ARG2:%[A-Za-z0-9_]*]]: !mo.tensor<[{{[A-Za-z0-9_]+}}], si64>
//  CHECK-SAME: [[ARG3:%[A-Za-z0-9_]*]]: !mo.tensor<[{{[A-Za-z0-9_]+}}], si64>
//  CHECK-SAME: [[ARG4:%[A-Za-z0-9_]*]]: !mo.tensor<[{{[A-Za-z0-9_]+}}], si64>)
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape([[ARG2]])
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape([[ARG3]])
//  CHECK-NEXT:   [[V2:%.*]] = mo.max_pool{{.*}}([[ARG0]]) [filter_shape = [[ARG1]], strides = [[V0]], dilations = [[V1]], paddings = [[ARG4]]]
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape{{.*}}([[V2]])
//  CHECK-NEXT:   mo.output{{.*}} [[V3]]

// -----

mo.graph @max_pool_static<() -> D0, D1, D2, D3>(%arg0: !mo.tensor<[3, 8, 43, 49], f32>, %fs: !mo.tensor<[2], si64>, %st: !mo.tensor<[2], si64>, %di: !mo.tensor<[2], si64>, %pa: !mo.tensor<[4], si64>) -> !mo.tensor<[D0, D1, D2, D3], f32> {
  %0 = rmo.mo.max_pool<() -> D4, D5, D6, D7>(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<[3, 8, 43, 49], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[D4, D5, D6, D7], f32>
  mo.output<D4, D5, D6, D7> %0 : !mo.tensor<[D4, D5, D6, D7], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @max_pool_static
//  CHECK-SAME: ([[ARG0:%[A-Za-z0-9_]*]]: !mo.tensor<[3, 8, 43, 49], f32>
//  CHECK-SAME: [[ARG1:%[A-Za-z0-9_]*]]: !mo.tensor<[2], si64>
//  CHECK-SAME: [[ARG2:%[A-Za-z0-9_]*]]: !mo.tensor<[2], si64>
//  CHECK-SAME: [[ARG3:%[A-Za-z0-9_]*]]: !mo.tensor<[2], si64>
//  CHECK-SAME: [[ARG4:%[A-Za-z0-9_]*]]: !mo.tensor<[4], si64>)
//  CHECK-NEXT:   [[V0:%.*]] = mo.max_pool{{.*}}([[ARG0]]) [filter_shape = [[ARG1]], strides = [[ARG2]], dilations = [[ARG3]], paddings = [[ARG4]]]
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape{{.*}}([[V0]])
//  CHECK-NEXT:   mo.output{{.*}} [[V1]]

// -----

mo.graph @max_pool_static_dyn<D0, D1, D2, D3, D4 -> D5, D6, D7, D8>(%arg0: !mo.tensor<[3, D0, 43, D1], f32>, %fs: !mo.tensor<[D2], si64>, %st: !mo.tensor<[D3], si64>, %di: !mo.tensor<[D4], si64>, %pa: !mo.tensor<[4], si64>) -> !mo.tensor<[D5, D6, D7, D8], f32> {
  %0 = rmo.mo.max_pool<() -> D9, D10, D11, D12>(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<[3, D0, 43, D1], f32>, !mo.tensor<[D2], si64>, !mo.tensor<[D3], si64>, !mo.tensor<[D4], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[D9, D10, D11, D12], f32>
  mo.output<D9, D10, D11, D12> %0 : !mo.tensor<[D9, D10, D11, D12], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @max_pool_static_dyn
//  CHECK-SAME: ([[ARG0:%[A-Za-z0-9_]*]]: !mo.tensor<[3, {{[A-Za-z0-9_]+}}, 43, {{[A-Za-z0-9_]+}}], f32>
//  CHECK-SAME: [[ARG1:%[A-Za-z0-9_]*]]: !mo.tensor<[{{[A-Za-z0-9_]+}}], si64>
//  CHECK-SAME: [[ARG2:%[A-Za-z0-9_]*]]: !mo.tensor<[{{[A-Za-z0-9_]+}}], si64>
//  CHECK-SAME: [[ARG3:%[A-Za-z0-9_]*]]: !mo.tensor<[{{[A-Za-z0-9_]+}}], si64>
//  CHECK-SAME: [[ARG4:%[A-Za-z0-9_]*]]: !mo.tensor<[4], si64>)
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape([[ARG2]])
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape([[ARG3]])
//  CHECK-NEXT:   [[V2:%.*]] = mo.max_pool{{.*}}([[ARG0]]) [filter_shape = [[ARG1]], strides = [[V0]], dilations = [[V1]], paddings = [[ARG4]]]
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape{{.*}}([[V2]])
//  CHECK-NEXT:   mo.output{{.*}} [[V3]]

// -----

mo.graph @max_pool_static_dyn_view<D0, D1, Sh2: !mosh.ape, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[3, D0, 43, D1], f32>, %fs: !mo.tensor<Sh2, si64>, %st: !mo.tensor<[D3], si64>, %di: !mo.tensor<[2], si64>, %pa: !mo.tensor<[4], si64>) -> !mo.tensor<[D4, D5, D6, D7], f32> {
  %0 = rmo.mo.max_pool<() -> D8, D9, D10, D11>(%arg0) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (!mo.tensor<[3, D0, 43, D1], f32>, !mo.tensor<Sh2, si64>, !mo.tensor<[D3], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[D8, D9, D10, D11], f32>
  mo.output<D8, D9, D10, D11> %0 : !mo.tensor<[D8, D9, D10, D11], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @max_pool_static_dyn_view
//  CHECK-SAME: ([[ARG0:%[A-Za-z0-9_]*]]: !mo.tensor<[3, {{[A-Za-z0-9_]+}}, 43, {{[A-Za-z0-9_]+}}], f32>
//  CHECK-SAME: [[ARG1:%[A-Za-z0-9_]*]]: !mo.tensor<{{[A-Za-z0-9_]+}}, si64>
//  CHECK-SAME: [[ARG2:%[A-Za-z0-9_]*]]: !mo.tensor<[{{[A-Za-z0-9_]+}}], si64>
//  CHECK-SAME: [[ARG3:%[A-Za-z0-9_]*]]: !mo.tensor<[2], si64>
//  CHECK-SAME: [[ARG4:%[A-Za-z0-9_]*]]: !mo.tensor<[4], si64>)
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape([[ARG1]])
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape([[ARG3]])
//  CHECK-NEXT:   [[V2:%.*]] = mo.max_pool{{.*}}([[ARG0]]) [filter_shape = [[V0]], strides = [[ARG2]], dilations = [[V1]], paddings = [[ARG4]]]
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape{{.*}}([[V2]])
//  CHECK-NEXT:   mo.output{{.*}} [[V3]]
