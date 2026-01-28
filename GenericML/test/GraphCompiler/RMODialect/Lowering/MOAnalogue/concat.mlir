// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @concat_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>, %axis: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.concat<() -> Sh4: !mosh.ape>[%axis: !mo.tensor<Sh2, si64>](%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> (!mo.tensor<Sh4, f32>)
  mo.output<:!mosh.ape Sh4> %0: !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @concat_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, f32>, %[[ARG2:.*]]: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<Sh2, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH1:.*]]: !mosh.ape>(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[RMO_SH1]], f32>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH2:.*]]: !mosh.ape>(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, f32>) -> !mo.tensor<[[RMO_SH2]], f32>
//  CHECK-NEXT:   %[[V3:.*]] = mo.concat<() -> [[RMO_SH3:.*]]: !mosh.ape>[%[[V0]] : !mo.tensor<[], si64>] (%[[V1]], %[[V2]]) : (!mo.tensor<[[RMO_SH1]], f32>, !mo.tensor<[[RMO_SH2]], f32>) -> !mo.tensor<[[RMO_SH3]], f32>
//  CHECK-NEXT:   %[[V4:.*]] = rmo.rebind_tensor_shape<() -> [[SH4:.*]]: !mosh.ape>(%[[V3]]) {{.*}} : (!mo.tensor<[[RMO_SH3]], f32>) -> !mo.tensor<[[SH4]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> %[[V4]] : !mo.tensor<[[SH4]], f32>

// -----

mo.graph @concat_dyn_params<Sh0: !mosh.ape, D0, D1, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<[D0, D1], f32>, %axis: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.concat<() -> Sh3: !mosh.ape>[%axis: !mo.tensor<Sh1, si64>](%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<[D0, D1], f32>) -> (!mo.tensor<Sh3, f32>)
  mo.output<:!mosh.ape Sh3> %0: !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @concat_dyn_params<Sh0: !mosh.ape, D0, D1, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<[D0, D1], f32>, %[[ARG2:.*]]: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<Sh1, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]], [[RMO_D2:.*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:.*]], [[RMO_D4:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   %[[V3:.*]] = mo.concat<() -> [[RMO_D5:.*]], [[RMO_D6:.*]]>[%[[V0]] : !mo.tensor<[], si64>] (%[[V1]], %[[V2]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   %[[V4:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V3]]) {{.*}} : (!mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>) -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> %[[V4]] : !mo.tensor<[[SH3]], f32>

// -----

mo.graph @concat_params_dyn<D0, D1, Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<Sh0, f32>, %axis: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.concat<() -> Sh3: !mosh.ape>[%axis: !mo.tensor<Sh1, si64>](%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<Sh0, f32>) -> (!mo.tensor<Sh3, f32>)
  mo.output<:!mosh.ape Sh3> %0: !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @concat_params_dyn<D0, D1, Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, D1], f32>, %[[ARG1:.*]]: !mo.tensor<Sh0, f32>, %[[ARG2:.*]]: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<Sh1, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]], [[RMO_D2:.*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:.*]], [[RMO_D4:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   %[[V3:.*]] = mo.concat<() -> [[RMO_D5:.*]], [[RMO_D6:.*]]>[%[[V0]] : !mo.tensor<[], si64>] (%[[V1]], %[[V2]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   %[[V4:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:.*]]: !mosh.ape>(%[[V3]]) {{.*}} : (!mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>) -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> %[[V4]] : !mo.tensor<[[SH3]], f32>

// -----

mo.graph @concat_params_mix<D0, Sh0: !mosh.ape, D1 -> Sh1: !mosh.ape>(%arg0: !mo.tensor<[2, D0], f32>, %arg1: !mo.tensor<Sh0, f32>, %arg2: !mo.tensor<[2, D1], f32>, %axis: !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f32> {
  %0 = rmo.mo.concat<() -> Sh2: !mosh.ape>[%axis: !mo.tensor<[], si64>](%arg0, %arg1, %arg2) : (!mo.tensor<[2, D0], f32>, !mo.tensor<Sh0, f32>, !mo.tensor<[2, D1], f32>) -> (!mo.tensor<Sh2, f32>)
  mo.output<:!mosh.ape Sh2> %0: !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @concat_params_mix<D0, Sh0: !mosh.ape, D1 -> Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[2, D0], f32>, %[[ARG1:.*]]: !mo.tensor<Sh0, f32>, %[[ARG2:.*]]: !mo.tensor<[2, D1], f32>, %[[ARG3:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]], [[RMO_D2:.*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<[2, D0], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:.*]], [[RMO_D4:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D5:.*]], [[RMO_D6:.*]]>(%[[ARG2]]) {{.*}} : (!mo.tensor<[2, D1], f32>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   %[[V3:.*]] = mo.concat<() -> [[RMO_D7:.*]], [[RMO_D8:.*]]>[%[[ARG3]] : !mo.tensor<[], si64>] (%[[V0]], %[[V1]], %[[V2]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>, !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>) -> !mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f32>
//  CHECK-NEXT:   %[[V4:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:.*]]: !mosh.ape>(%[[V3]]) {{.*}} : (!mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> %[[V4]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @concat_dyn_out_params<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> D0, D1>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>, %axis: !mo.tensor<Sh2, si64>) -> !mo.tensor<[D0, D1], f32> {
  %0 = rmo.mo.concat<() -> D2, D3>[%axis: !mo.tensor<Sh2, si64>](%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> (!mo.tensor<[D2, D3], f32>)
  mo.output<D2, D3> %0: !mo.tensor<[D2, D3], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @concat_dyn_out_params<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> D0, D1>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, f32>, %[[ARG2:.*]]: !mo.tensor<Sh2, si64>) -> !mo.tensor<[D0, D1], f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<Sh2, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]], [[RMO_D2:.*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:.*]], [[RMO_D4:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   %[[V3:.*]] = mo.concat<() -> [[RMO_D5:.*]], [[RMO_D6:.*]]>[%[[V0]] : !mo.tensor<[], si64>] (%[[V1]], %[[V2]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   %[[V4:.*]] = rmo.rebind_tensor_shape<() -> [[D2:.*]], [[D3:.*]]>(%[[V3]]) {{.*}} : (!mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>) -> !mo.tensor<[[[D2]], [[D3]]], f32>
//  CHECK-NEXT:   mo.output<[[D2]], [[D3]]> %[[V4]] : !mo.tensor<[[[D2]], [[D3]]], f32>

// -----

mo.graph @concat_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[2, 7], f32>, %arg1: !mo.tensor<[2, 3], f32>, %axis: !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.concat<() -> Sh1: !mosh.ape>[%axis: !mo.tensor<[], si64>](%arg0, %arg1) : (!mo.tensor<[2, 7], f32>, !mo.tensor<[2, 3], f32>) -> (!mo.tensor<Sh1, f32>)
  mo.output<:!mosh.ape Sh1> %0: !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @concat_static<() -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[2, 7], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG2:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:.*]], [[RMO_D2:.*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<[2, 7], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:.*]], [[RMO_D4:.*]]>(%[[ARG1]]) {{.*}} : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   %[[V2:.*]] = mo.concat<() -> [[RMO_D5:.*]], [[RMO_D6:.*]]>[%[[ARG2]] : !mo.tensor<[], si64>] (%[[V0]], %[[V1]]) : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V3]] : !mo.tensor<[[SH1]], f32>
