// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @split_dyn(%arg0: !mo.tensor<?, f32>, %axis: !mo.tensor<?, si64>, %sizes: !mo.tensor<?, si64>) -> (!mo.tensor<?, f32>, !mo.tensor<?, f32>) {
  %0:2 = rmo.mo.split[%axis: !mo.tensor<?, si64>](%arg0, %sizes) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> (!mo.tensor<?, f32>, !mo.tensor<?, f32>)
  mo.output %0#0, %0#1 : !mo.tensor<?, f32>, !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @split_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape, [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>) -> (!mo.tensor<[[SH4]], f32>, !mo.tensor<[[SH5]], f32>) {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH7]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH8]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH7]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH8]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V5:%.*]]:2 = mo.split<() -> [[RMO_SH11:[A-Za-z0-9_]*]]: !mosh.ape, [[RMO_SH12:[A-Za-z0-9_]*]]: !mosh.ape>[[[V4]] : !mo.tensor<[], si64>] ([[V0]], [[V3]]) : (!mo.tensor<[[SH6]], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[[RMO_SH11]], f32>, !mo.tensor<[[RMO_SH12]], f32>)
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[V5]]#0) {{.*}} : (!mo.tensor<[[RMO_SH11]], f32>) -> !mo.tensor<[[SH9]], f32>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V5]]#1) {{.*}} : (!mo.tensor<[[RMO_SH12]], f32>) -> !mo.tensor<[[SH10]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH9]], :!mosh.ape [[SH10]]> [[V6]], [[V7]] : !mo.tensor<[[SH9]], f32>, !mo.tensor<[[SH10]], f32>

// -----

mo.graph @split_params(%arg0: !mo.tensor<[?, ?], f32>, %axis: !mo.tensor<[], si64>, %sizes: !mo.tensor<[?], si64>) -> (!mo.tensor<?, f32>, !mo.tensor<?, f32>) {
  %0:2 = rmo.mo.split[%axis: !mo.tensor<[], si64>](%arg0, %sizes) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?], si64>) -> (!mo.tensor<?, f32>, !mo.tensor<?, f32>)
  mo.output %0#0, %0#1 : !mo.tensor<?, f32>, !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @split_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]] -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape, [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[], si64>, [[ARG2:%.*]]: !mo.tensor<[[[D3]]], si64>) -> (!mo.tensor<[[SH4]], f32>, !mo.tensor<[[SH5]], f32>) {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D6]], [[D7]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D8:[A-Za-z0-9_]*]]>([[ARG2]]) : !mo.tensor<[[[D3]]], si64> -> !mo.tensor<[[[D8]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D8]]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V3:%.*]]:2 = mo.split<() -> [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]], [[RMO_D13:[A-Za-z0-9_]*]], [[RMO_D14:[A-Za-z0-9_]*]]>[[[ARG1]] : !mo.tensor<[], si64>] ([[V0]], [[V2]]) : (!mo.tensor<[[[D6]], [[D7]]], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>, !mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>)
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]#0) {{.*}} : (!mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>) -> !mo.tensor<[[SH9]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]#1) {{.*}} : (!mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>) -> !mo.tensor<[[SH10]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH9]], :!mosh.ape [[SH10]]> [[V4]], [[V5]] : !mo.tensor<[[SH9]], f32>, !mo.tensor<[[SH10]], f32>

// -----

mo.graph @split_mix(%arg0: !mo.tensor<[?, 4], f32>, %axis: !mo.tensor<[], si64>, %sizes: !mo.tensor<[?], si64>) -> (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, f32>) {
  %0:3 = rmo.mo.split[%axis: !mo.tensor<[], si64>](%arg0, %sizes) : (!mo.tensor<[?, 4], f32>, !mo.tensor<[?], si64>) -> (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, f32>)
  mo.output %0#0, %0#1, %0#2 : !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @split_mix<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]] -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape, [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], 4], f32>, [[ARG1:%.*]]: !mo.tensor<[], si64>, [[ARG2:%.*]]: !mo.tensor<[[[D2]]], si64>) -> (!mo.tensor<[[SH3]], f32>, !mo.tensor<[[SH4]], f32>, !mo.tensor<[[SH5]], f32>) {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D6:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], 4], f32> -> !mo.tensor<[[[D6]], 4], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D7:[A-Za-z0-9_]*]]>([[ARG2]]) : !mo.tensor<[[[D2]]], si64> -> !mo.tensor<[[[D7]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D7]]], si64>) -> !mo.tensor<[3], si64>
//  CHECK-NEXT:   [[V3:%.*]]:3 = mo.split<() -> [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]], [[RMO_D13:[A-Za-z0-9_]*]], [[RMO_D14:[A-Za-z0-9_]*]], [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]]>[[[ARG1]] : !mo.tensor<[], si64>] ([[V0]], [[V2]]) : (!mo.tensor<[[[D6]], 4], f32>, !mo.tensor<[3], si64>) -> (!mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>, !mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>, !mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>)
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]#0) {{.*}} : (!mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]#1) {{.*}} : (!mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>) -> !mo.tensor<[[SH9]], f32>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V3]]#2) {{.*}} : (!mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>) -> !mo.tensor<[[SH10]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]], :!mosh.ape [[SH9]], :!mosh.ape [[SH10]]> [[V4]], [[V5]], [[V6]] : !mo.tensor<[[SH8]], f32>, !mo.tensor<[[SH9]], f32>, !mo.tensor<[[SH10]], f32>

// -----

mo.graph @split_static(%arg0: !mo.tensor<[8, 9], f32>, %axis: !mo.tensor<[], si64>, %sizes: !mo.tensor<[2], si64>) -> (!mo.tensor<?, f32>, !mo.tensor<?, f32>) {
  %0:2 = rmo.mo.split[%axis: !mo.tensor<[], si64>](%arg0, %sizes) : (!mo.tensor<[8, 9], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<?, f32>, !mo.tensor<?, f32>)
  mo.output %0#0, %0#1 : !mo.tensor<?, f32>, !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @split_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[8, 9], f32>, [[ARG1:%.*]]: !mo.tensor<[], si64>, [[ARG2:%.*]]: !mo.tensor<[2], si64>) -> (!mo.tensor<[[SH1]], f32>, !mo.tensor<[[SH2]], f32>) {
//  CHECK-NEXT:   [[V0:%.*]]:2 = mo.split<() -> [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]]>[[[ARG1]] : !mo.tensor<[], si64>] ([[ARG0]], [[ARG2]]) : (!mo.tensor<[8, 9], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>, !mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f32>)
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]#0) {{.*}} : (!mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>) -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]#1) {{.*}} : (!mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]], :!mosh.ape [[SH4]]> [[V1]], [[V2]] : !mo.tensor<[[SH3]], f32>, !mo.tensor<[[SH4]], f32>
