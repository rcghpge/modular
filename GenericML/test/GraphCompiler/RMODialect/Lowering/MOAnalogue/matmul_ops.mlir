// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope


mo.graph @matmul_dyn_dyn(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.matmul(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @matmul_dyn_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D9:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[[RMO_D8]], [[RMO_D9]]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.matmul([[V2]], [[V3]]) : (!mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f32>, !mo.tensor<[[[RMO_D8]], [[RMO_D9]]], f32>) -> !mo.tensor<[[[RMO_D7]], [[RMO_D9]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D7]], [[RMO_D9]]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V5]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @matmul_dyn_dyn_mixed_types(%arg0: !mo.tensor<?, f16>, %arg1: !mo.tensor<?, bf16>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.matmul(%arg0, %arg1) : (!mo.tensor<?, f16>, !mo.tensor<?, bf16>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @matmul_dyn_dyn_mixed_types<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f16>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], bf16>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f16> -> !mo.tensor<[[SH4]], f16>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], bf16> -> !mo.tensor<[[SH5]], bf16>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH4]], f16>) -> !mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f16>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D9:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], bf16>) -> !mo.tensor<[[[RMO_D8]], [[RMO_D9]]], bf16>
//  CHECK-NEXT:   [[V4:%.*]] = mo.matmul([[V2]], [[V3]]) : (!mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f16>, !mo.tensor<[[[RMO_D8]], [[RMO_D9]]], bf16>) -> !mo.tensor<[[[RMO_D7]], [[RMO_D9]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D7]], [[RMO_D9]]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V5]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @matmul_dyn_param(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<[?, ?], f32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.matmul(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<[?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @matmul_dyn_param<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]] -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D2]], [[D3]]], f32>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D2]], [[D3]]], f32> -> !mo.tensor<[[[D6]], [[D7]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D11:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[D6]], [[D7]]], f32>) -> !mo.tensor<[[[RMO_D10]], [[RMO_D11]]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.matmul([[V2]], [[V3]]) : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>, !mo.tensor<[[[RMO_D10]], [[RMO_D11]]], f32>) -> !mo.tensor<[[[RMO_D9]], [[RMO_D11]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D9]], [[RMO_D11]]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V5]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @matmul_mixed_param(%arg0: !mo.tensor<[?, 4], f32>, %arg1: !mo.tensor<[4, ?], f32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.matmul(%arg0, %arg1) : (!mo.tensor<[?, 4], f32>, !mo.tensor<[4, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @matmul_mixed_param<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]] -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], 4], f32>, [[ARG1:%.*]]: !mo.tensor<[4, [[D2]]], f32>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D4:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], 4], f32> -> !mo.tensor<[[[D4]], 4], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D5:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[4, [[D2]]], f32> -> !mo.tensor<[4, [[D5]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[D4]], 4], f32>) -> !mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D9:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[4, [[D5]]], f32>) -> !mo.tensor<[[[RMO_D8]], [[RMO_D9]]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.matmul([[V2]], [[V3]]) : (!mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f32>, !mo.tensor<[[[RMO_D8]], [[RMO_D9]]], f32>) -> !mo.tensor<[[[RMO_D7]], [[RMO_D9]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D7]], [[RMO_D9]]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V5]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @matmul_static(%arg0: !mo.tensor<[7, 4], f32>, %arg1: !mo.tensor<[4, 9], f32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.matmul(%arg0, %arg1) : (!mo.tensor<[7, 4], f32>, !mo.tensor<[4, 9], f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @matmul_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[7, 4], f32>, [[ARG1:%.*]]: !mo.tensor<[4, 9], f32>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[7, 4], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[4, 9], f32>) -> !mo.tensor<[[[RMO_D4]], [[RMO_D5]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.matmul([[V0]], [[V1]]) : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>, !mo.tensor<[[[RMO_D4]], [[RMO_D5]]], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D5]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D5]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V3]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @batch_matmul_dyn_dyn(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @batch_matmul_dyn_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.batch_matmul<() -> [[RMO_SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]], [[V1]]) : (!mo.tensor<[[SH4]], f32>, !mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[RMO_SH7]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[RMO_SH7]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V3]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @batch_matmul_dyn_dyn_mixed_types(%arg0: !mo.tensor<?, f16>, %arg1: !mo.tensor<?, bf16>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<?, f16>, !mo.tensor<?, bf16>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @batch_matmul_dyn_dyn_mixed_types<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f16>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], bf16>) -> !mo.tensor<[[SH3]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f16> -> !mo.tensor<[[SH4]], f16>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], bf16> -> !mo.tensor<[[SH5]], bf16>
//  CHECK-NEXT:   [[V2:%.*]] = mo.batch_matmul<() -> [[RMO_SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]], [[V1]]) : (!mo.tensor<[[SH4]], f16>, !mo.tensor<[[SH5]], bf16>) -> !mo.tensor<[[RMO_SH7]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[RMO_SH7]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V3]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @batch_matmul_dyn_params(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @batch_matmul_dyn_params<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]] -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D7:[A-Za-z0-9_]*]], [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.batch_matmul<() -> [[RMO_SH11:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]], [[V1]]) : (!mo.tensor<[[SH6]], f32>, !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[RMO_SH11]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[RMO_SH11]], f32>) -> !mo.tensor<[[SH10]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH10]]> [[V3]] : !mo.tensor<[[SH10]], f32>

// -----

mo.graph @batch_matmul_params_params(%arg0: !mo.tensor<[?, ?, ?], f32>, %arg1: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @batch_matmul_params_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]], [[D6:[A-Za-z0-9_]*]] -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>) -> !mo.tensor<[[SH7]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D11:[A-Za-z0-9_]*]], [[D12:[A-Za-z0-9_]*]], [[D13:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32> -> !mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.batch_matmul<() -> [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]], [[RMO_D17:[A-Za-z0-9_]*]]>([[V0]], [[V1]]) : (!mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>, !mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>) -> !mo.tensor<[[[RMO_D15]], [[RMO_D16]], [[RMO_D17]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH14:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D15]], [[RMO_D16]], [[RMO_D17]]], f32>) -> !mo.tensor<[[SH14]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH14]]> [[V3]] : !mo.tensor<[[SH14]], f32>

// -----

mo.graph @batch_matmul_mixed_params(%arg0: !mo.tensor<[2, ?, 4], f32>, %arg1: !mo.tensor<[?, 4, ?], f32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[2, ?, 4], f32>, !mo.tensor<[?, 4, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @batch_matmul_mixed_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]] -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, [[D1]], 4], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D2]], 4, [[D3]]], f32>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D5:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[2, [[D1]], 4], f32> -> !mo.tensor<[2, [[D5]], 4], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D2]], 4, [[D3]]], f32> -> !mo.tensor<[[[D6]], 4, [[D7]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.batch_matmul<() -> [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]], [[RMO_D11:[A-Za-z0-9_]*]]>([[V0]], [[V1]]) : (!mo.tensor<[2, [[D5]], 4], f32>, !mo.tensor<[[[D6]], 4, [[D7]]], f32>) -> !mo.tensor<[[[RMO_D9]], [[RMO_D10]], [[RMO_D11]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]], [[RMO_D11]]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V3]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @batch_matmul_static(%arg0: !mo.tensor<[2, 7, 4], f32>, %arg1: !mo.tensor<[2, 4, 9], f32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[2, 7, 4], f32>, !mo.tensor<[2, 4, 9], f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @batch_matmul_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, 7, 4], f32>, [[ARG1:%.*]]: !mo.tensor<[2, 4, 9], f32>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.batch_matmul<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) : (!mo.tensor<[2, 7, 4], f32>, !mo.tensor<[2, 4, 9], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>
