// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @band_part_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape -> Sh4: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %num_lower: !mo.tensor<Sh1, si64>, %num_upper: !mo.tensor<Sh2, si64>, %exclude: !mo.tensor<Sh3, bool>) -> !mo.tensor<Sh4, f32> {
  %0 = rmo.mo.linalg.band_part<() -> Sh5: !mosh.ape>(%arg0, %num_lower, %num_upper, %exclude) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, bool>) -> !mo.tensor<Sh5, f32>
  mo.output<:!mosh.ape Sh5> %0 : !mo.tensor<Sh5, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @band_part_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape -> Sh4: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, si64>, %[[ARG2:.*]]: !mo.tensor<Sh2, si64>, %[[ARG3:.*]]: !mo.tensor<Sh3, bool>) -> !mo.tensor<Sh4, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<Sh2, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape(%[[ARG3]]) {{.*}} : (!mo.tensor<Sh3, bool>) -> !mo.tensor<[], bool>
//  CHECK-NEXT:   %[[V3:.*]] = mo.linalg.band_part(%[[ARG0]], %[[V0]], %[[V1]], %[[V2]]) : (!mo.tensor<Sh0, f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> !mo.tensor<Sh0, f32>
//  CHECK-NEXT:   %[[V4:.*]] = rmo.rebind_tensor_shape<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V3]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH5]]> %[[V4]] : !mo.tensor<[[SH5]], f32>

// -----

mo.graph @band_part_dyn_params<D0, D1, Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<[D0, D1], f32>, %num_lower: !mo.tensor<Sh0, si64>, %num_upper: !mo.tensor<[], si64>, %exclude: !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.linalg.band_part<() -> Sh3: !mosh.ape>(%arg0, %num_lower, %num_upper, %exclude) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<Sh0, si64>, !mo.tensor<[], si64>, !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @band_part_dyn_params<D0, D1, Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, D1], f32>, %[[ARG1:.*]]: !mo.tensor<Sh0, si64>, %[[ARG2:.*]]: !mo.tensor<[], si64>, %[[ARG3:.*]]: !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh2, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh0, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG3]]) {{.*}} : (!mo.tensor<Sh1, bool>) -> !mo.tensor<[], bool>
//  CHECK-NEXT:   %[[V2:.*]] = mo.linalg.band_part(%[[ARG0]], %[[V0]], %[[ARG2]], %[[V1]]) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> !mo.tensor<[D0, D1], f32>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V2]]) {{.*}} : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]]> %[[V3]] : !mo.tensor<[[SH3]], f32>

// -----

mo.graph @band_part_mix_params<D0, Sh0: !mosh.ape -> Sh1: !mosh.ape>(%arg0: !mo.tensor<[D0, 3], f32>, %num_lower: !mo.tensor<[], si64>, %num_upper: !mo.tensor<[], si64>, %exclude: !mo.tensor<Sh0, bool>) -> !mo.tensor<Sh1, f32> {
  %0 = rmo.mo.linalg.band_part<() -> Sh2: !mosh.ape>(%arg0, %num_lower, %num_upper, %exclude) : (!mo.tensor<[D0, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<Sh0, bool>) -> !mo.tensor<Sh2, f32>
  mo.output<:!mosh.ape Sh2> %0 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @band_part_mix_params<D0, Sh0: !mosh.ape -> Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>, %[[ARG2:.*]]: !mo.tensor<[], si64>, %[[ARG3:.*]]: !mo.tensor<Sh0, bool>) -> !mo.tensor<Sh1, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape(%[[ARG3]]) {{.*}} : (!mo.tensor<Sh0, bool>) -> !mo.tensor<[], bool>
//  CHECK-NEXT:   %[[V1:.*]] = mo.linalg.band_part(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[V0]]) : (!mo.tensor<[D0, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> !mo.tensor<[D0, 3], f32>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V1]]) {{.*}} : (!mo.tensor<[D0, 3], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> %[[V2]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @band_part_static<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[7, 3], f32>, %num_lower: !mo.tensor<[], si64>, %num_upper: !mo.tensor<[], si64>, %exclude: !mo.tensor<[], bool>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.linalg.band_part<() -> Sh1: !mosh.ape>(%arg0, %num_lower, %num_upper, %exclude) : (!mo.tensor<[7, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @band_part_static<() -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[7, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>, %[[ARG2:.*]]: !mo.tensor<[], si64>, %[[ARG3:.*]]: !mo.tensor<[], bool>) -> !mo.tensor<Sh0, f32> {
//  CHECK-NEXT:   %[[V0:.*]] = mo.linalg.band_part(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]) : (!mo.tensor<[7, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> !mo.tensor<[7, 3], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[7, 3], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V1]] : !mo.tensor<[[SH1]], f32>
