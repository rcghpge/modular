// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @split_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape, Sh4: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %axis: !mo.tensor<Sh1, si64>, %sizes: !mo.tensor<Sh2, si64>) -> (!mo.tensor<Sh3, f32>, !mo.tensor<Sh4, f32>) {
  %0:2 = rmo.mo.split<() -> Sh5: !mosh.ape, Sh6: !mosh.ape>[%axis: !mo.tensor<Sh1, si64>](%arg0, %sizes) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh2, si64>) -> (!mo.tensor<Sh5, f32>, !mo.tensor<Sh6, f32>)
  mo.output<:!mosh.ape Sh5, :!mosh.ape Sh6> %0#0, %0#1 : !mo.tensor<Sh5, f32>, !mo.tensor<Sh6, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @split_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape, Sh4: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, si64>, %[[ARG2:.*]]: !mo.tensor<Sh2, si64>) -> (!mo.tensor<Sh3, f32>, !mo.tensor<Sh4, f32>) {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<Sh2, si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V2:.*]]:2 = mo.split<() -> [[RMO_SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[RMO_SH2:[A-Za-z0-9_]*]]: !mosh.ape>[%[[V1]] : !mo.tensor<[], si64>] (%[[ARG0]], %[[V0]]) : (!mo.tensor<Sh0, f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[[RMO_SH1]], f32>, !mo.tensor<[[RMO_SH2]], f32>)
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V2]]#0) {{.*}} : (!mo.tensor<[[RMO_SH1]], f32>) -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   %[[V4:.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V2]]#1) {{.*}} : (!mo.tensor<[[RMO_SH2]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH5]], :!mosh.ape [[SH6]]> %[[V3]], %[[V4]] : !mo.tensor<[[SH5]], f32>, !mo.tensor<[[SH6]], f32>

// -----

mo.graph @split_params<D0, D1, D2 -> Sh0: !mosh.ape, Sh1: !mosh.ape>(%arg0: !mo.tensor<[D0, D1], f32>, %axis: !mo.tensor<[], si64>, %sizes: !mo.tensor<[D2], si64>) -> (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) {
  %0:2 = rmo.mo.split<() -> Sh2: !mosh.ape, Sh3: !mosh.ape>[%axis: !mo.tensor<[], si64>](%arg0, %sizes) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2], si64>) -> (!mo.tensor<Sh2, f32>, !mo.tensor<Sh3, f32>)
  mo.output<:!mosh.ape Sh2, :!mosh.ape Sh3> %0#0, %0#1 : !mo.tensor<Sh2, f32>, !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @split_params<D0, D1, D2 -> Sh0: !mosh.ape, Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, D1], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>, %[[ARG2:.*]]: !mo.tensor<[D2], si64>) -> (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<[D2], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   %[[V1:.*]]:2 = mo.split<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>[%[[ARG1]] : !mo.tensor<[], si64>] (%[[ARG0]], %[[V0]]) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>)
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V1]]#0) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V1]]#1) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]], :!mosh.ape [[SH3]]> %[[V2]], %[[V3]] : !mo.tensor<[[SH2]], f32>, !mo.tensor<[[SH3]], f32>

// -----

mo.graph @split_mix<D0, D1 -> Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<[D0, 4], f32>, %axis: !mo.tensor<[], si64>, %sizes: !mo.tensor<[D1], si64>) -> (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f32>) {
  %0:3 = rmo.mo.split<() -> Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape>[%axis: !mo.tensor<[], si64>](%arg0, %sizes) : (!mo.tensor<[D0, 4], f32>, !mo.tensor<[D1], si64>) -> (!mo.tensor<Sh3, f32>, !mo.tensor<Sh4, f32>, !mo.tensor<Sh5, f32>)
  mo.output<:!mosh.ape Sh3, :!mosh.ape Sh4, :!mosh.ape Sh5> %0#0, %0#1, %0#2 : !mo.tensor<Sh3, f32>, !mo.tensor<Sh4, f32>, !mo.tensor<Sh5, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @split_mix<D0, D1 -> Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, 4], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>, %[[ARG2:.*]]: !mo.tensor<[D1], si64>) -> (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f32>) {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<[D1], si64>) -> !mo.tensor<[3], si64>
//  CHECK-NEXT:   %[[V1:.*]]:3 = mo.split<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]]>[%[[ARG1]] : !mo.tensor<[], si64>] (%[[ARG0]], %[[V0]]) : (!mo.tensor<[D0, 4], f32>, !mo.tensor<[3], si64>) -> (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>, !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>)
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V1]]#0) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>) -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V1]]#1) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   %[[V4:.*]] = rmo.rebind_tensor_shape<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V1]]#2) {{.*}} : (!mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>) -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]], :!mosh.ape [[SH4]], :!mosh.ape [[SH5]]> %[[V2]], %[[V3]], %[[V4]] : !mo.tensor<[[SH3]], f32>, !mo.tensor<[[SH4]], f32>, !mo.tensor<[[SH5]], f32>

// -----

mo.graph @split_static<() -> Sh0: !mosh.ape, Sh1: !mosh.ape>(%arg0: !mo.tensor<[8, 9], f32>, %axis: !mo.tensor<[], si64>, %sizes: !mo.tensor<[2], si64>) -> (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) {
  %0:2 = rmo.mo.split<() -> Sh2: !mosh.ape, Sh3: !mosh.ape>[%axis: !mo.tensor<[], si64>](%arg0, %sizes) : (!mo.tensor<[8, 9], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<Sh2, f32>, !mo.tensor<Sh3, f32>)
  mo.output<:!mosh.ape Sh2, :!mosh.ape Sh3> %0#0, %0#1 : !mo.tensor<Sh2, f32>, !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @split_static<() -> Sh0: !mosh.ape, Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[8, 9], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>, %[[ARG2:.*]]: !mo.tensor<[2], si64>) -> (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) {
//  CHECK-NEXT:   %[[V0:.*]]:2 = mo.split<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>[%[[ARG1]] : !mo.tensor<[], si64>] (%[[ARG0]], %[[ARG2]]) : (!mo.tensor<[8, 9], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>)
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V0]]#0) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V0]]#1) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]], :!mosh.ape [[SH3]]> %[[V1]], %[[V2]] : !mo.tensor<[[SH2]], f32>, !mo.tensor<[[SH3]], f32>
