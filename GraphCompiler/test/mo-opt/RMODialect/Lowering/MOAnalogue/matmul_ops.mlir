// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope


mo.graph @matmul_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.matmul<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @matmul_dyn_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.matmul([[V0]], [[V1]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D3]]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V3]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @matmul_dyn_dyn_mixed_types<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f16>, %arg1: !mo.tensor<Sh1, bf16>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.matmul<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f16>, !mo.tensor<Sh1, bf16>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @matmul_dyn_dyn_mixed_types<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f16>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], bf16>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH1]], f16>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f16>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH2]], bf16>) -> !mo.tensor<[[[RMO_D2]], [[RMO_D3]]], bf16>
//  CHECK-NEXT:   [[V2:%.*]] = mo.matmul([[V0]], [[V1]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f16>, !mo.tensor<[[[RMO_D2]], [[RMO_D3]]], bf16>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D3]]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V3]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @matmul_dyn_param<Sh0: !mosh.ape, D0, D1 -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.matmul<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<[D0, D1], f32>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @matmul_dyn_param<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]] -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D2]], [[D3]]], f32>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[[D2]], [[D3]]], f32>) -> !mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.matmul([[V0]], [[V1]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D3]]], f32>) -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH5]]> [[V3]] : !mo.tensor<[[SH5]], f32>

// -----

mo.graph @matmul_mixed_param<D0, D1 -> Sh2: !mosh.ape>(%arg0: !mo.tensor<[D0, 4], f32>, %arg1: !mo.tensor<[4, D1], f32>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.matmul<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[D0, 4], f32>, !mo.tensor<[4, D1], f32>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @matmul_mixed_param<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]] -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], 4], f32>, [[ARG1:%.*]]: !mo.tensor<[4, [[D2]]], f32>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[[D1]], 4], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[4, [[D2]]], f32>) -> !mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.matmul([[V0]], [[V1]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D3]]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V3]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @matmul_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[7, 4], f32>, %arg1: !mo.tensor<[4, 9], f32>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.matmul<() -> Sh1: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[7, 4], f32>, !mo.tensor<[4, 9], f32>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @matmul_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[7, 4], f32>, [[ARG1:%.*]]: !mo.tensor<[4, 9], f32>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[7, 4], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[4, 9], f32>) -> !mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.matmul([[V0]], [[V1]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D3]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V3]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @batch_matmul_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.batch_matmul<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @batch_matmul_dyn_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.batch_matmul<() -> [[RMO_SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]], [[ARG1]]) : (!mo.tensor<[[SH1]], f32>, !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[RMO_SH1]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[RMO_SH1]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V1]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @batch_matmul_dyn_dyn_mixed_types<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f16>, %arg1: !mo.tensor<Sh1, bf16>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.batch_matmul<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f16>, !mo.tensor<Sh1, bf16>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @batch_matmul_dyn_dyn_mixed_types<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f16>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], bf16>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.batch_matmul<() -> [[RMO_SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]], [[ARG1]]) : (!mo.tensor<[[SH1]], f16>, !mo.tensor<[[SH2]], bf16>) -> !mo.tensor<[[RMO_SH1]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[RMO_SH1]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V1]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @batch_matmul_dyn_params<Sh0: !mosh.ape, D0, D1, D2 -> Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.batch_matmul<() -> Sh4: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %0 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @batch_matmul_dyn_params<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]] -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.batch_matmul<() -> [[RMO_SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]], [[ARG1]]) : (!mo.tensor<[[SH1]], f32>, !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32>) -> !mo.tensor<[[RMO_SH1]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[RMO_SH1]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V1]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @batch_matmul_params_params<D0, D1, D2, D3, D4, D5 -> Sh6: !mosh.ape>(%arg0: !mo.tensor<[D0, D1, D2], f32>, %arg1: !mo.tensor<[D3, D4, D5], f32>) -> !mo.tensor<Sh6, f32> {
  %0 = rmo.mo.batch_matmul<() -> Sh7: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[D3, D4, D5], f32>) -> !mo.tensor<Sh7, f32>
  mo.output<:!mosh.ape Sh7> %0 : !mo.tensor<Sh7, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @batch_matmul_params_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]], [[D6:[A-Za-z0-9_]*]] -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>) -> !mo.tensor<[[SH7]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.batch_matmul<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V1]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @batch_matmul_mixed_params<D0, D1, D2 -> Sh3: !mosh.ape>(%arg0: !mo.tensor<[2, D0, 4], f32>, %arg1: !mo.tensor<[D1, 4, D2], f32>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.batch_matmul<() -> Sh4: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[2, D0, 4], f32>, !mo.tensor<[D1, 4, D2], f32>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %0 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @batch_matmul_mixed_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]] -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, [[D1]], 4], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D2]], 4, [[D3]]], f32>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.batch_matmul<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) : (!mo.tensor<[2, [[D1]], 4], f32>, !mo.tensor<[[[D2]], 4, [[D3]]], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32>) -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH5]]> [[V1]] : !mo.tensor<[[SH5]], f32>

// -----

mo.graph @batch_matmul_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[2, 7, 4], f32>, %arg1: !mo.tensor<[2, 4, 9], f32>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.batch_matmul<() -> Sh1: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[2, 7, 4], f32>, !mo.tensor<[2, 4, 9], f32>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @batch_matmul_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, 7, 4], f32>, [[ARG1:%.*]]: !mo.tensor<[2, 4, 9], f32>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.batch_matmul<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) : (!mo.tensor<[2, 7, 4], f32>, !mo.tensor<[2, 4, 9], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>
