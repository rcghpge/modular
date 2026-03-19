// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @roi_align_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape -> Sh6: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %rois: !mo.tensor<Sh1, f32>, %output_height: !mo.tensor<Sh2, si64>, %output_width: !mo.tensor<Sh3, si32>, %spatial_scale: !mo.tensor<Sh4, f32>, %sampling_ratio: !mo.tensor<Sh5, f32>) -> !mo.tensor<Sh6, f32> {
  %0 = rmo.mo.roi_align<() -> Sh7: !mosh.ape>(%input, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio) {aligned = false,  mode = "AVG"} : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, si32>, !mo.tensor<Sh4, f32>, !mo.tensor<Sh5, f32>) -> !mo.tensor<Sh7, f32>
  mo.output<:!mosh.ape Sh7> %0 : !mo.tensor<Sh7, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @roi_align_dyn<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape, [[SH5:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH2]], si64>, [[ARG3:%.*]]: !mo.tensor<[[SH3]], si32>, [[ARG4:%.*]]: !mo.tensor<[[SH4]], f32>, [[ARG5:%.*]]: !mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH6]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[ARG2]]) {{.*}} : (!mo.tensor<[[SH2]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[ARG3]]) {{.*}} : (!mo.tensor<[[SH3]], si32>) -> !mo.tensor<[], si32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape([[ARG4]]) {{.*}} : (!mo.tensor<[[SH4]], f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape([[ARG5]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   [[V6:%.*]] = mo.roi_align<() -> [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]], [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]]>([[V0]], [[V1]], [[V2]], [[V3]], [[V4]], [[V5]]) {aligned = false, mode = "[[AVG:[A-Za-z0-9_]*]]"} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>, !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[RMO_D7]], [[RMO_D8]], [[RMO_D9]], [[RMO_D10]]], f32>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[V6]]) {{.*}} : (!mo.tensor<[[[RMO_D7]], [[RMO_D8]], [[RMO_D9]], [[RMO_D10]]], f32>) -> !mo.tensor<[[SH7]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH7]]> [[V7]] : !mo.tensor<[[SH7]], f32>

// -----

mo.graph @roi_align_params<D0, D1, D2, D3, D4, D5 -> Sh0: !mosh.ape>(%input: !mo.tensor<[D0, D1, D2, D3], f32>, %rois: !mo.tensor<[D4, D5], f32>, %output_height: !mo.tensor<[], si64>, %output_width: !mo.tensor<[], si32>, %spatial_scale: !mo.tensor<[], f32>, %sampling_ratio: !mo.tensor<[], f32>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.roi_align<() -> Sh1: !mosh.ape>(%input, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio) {aligned = false,  mode = "AVG"} : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[D4, D5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @roi_align_params<[[D0:[A-Za-z0-9_]*]], [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]] -> [[SH0:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D4]], [[D5]]], f32>, [[ARG2:%.*]]: !mo.tensor<[], si64>, [[ARG3:%.*]]: !mo.tensor<[], si32>, [[ARG4:%.*]]: !mo.tensor<[], f32>, [[ARG5:%.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[[SH0]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[[D4]], [[D5]]], f32>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.roi_align<() -> [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]], [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]]>([[V0]], [[V1]], [[ARG2]], [[ARG3]], [[ARG4]], [[ARG5]]) {aligned = false, mode = "[[AVG:[A-Za-z0-9_]*]]"} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>, !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[RMO_D7]], [[RMO_D8]], [[RMO_D9]], [[RMO_D10]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D7]], [[RMO_D8]], [[RMO_D9]], [[RMO_D10]]], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V3]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @roi_align_params_mix<D0, D1, D2, D3 -> Sh0: !mosh.ape>(%input: !mo.tensor<[2, D0, D1, D2], f32>, %rois: !mo.tensor<[D3, 5], f32>, %output_height: !mo.tensor<[], si64>, %output_width: !mo.tensor<[], si32>, %spatial_scale: !mo.tensor<[], f32>, %sampling_ratio: !mo.tensor<[], f32>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.roi_align<() -> Sh1: !mosh.ape>(%input, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio) {aligned = false,  mode = "AVG"} : (!mo.tensor<[2, D0, D1, D2], f32>, !mo.tensor<[D3, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @roi_align_params_mix<[[D0:[A-Za-z0-9_]*]], [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]] -> [[SH0:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, [[D0]], [[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D3]], 5], f32>, [[ARG2:%.*]]: !mo.tensor<[], si64>, [[ARG3:%.*]]: !mo.tensor<[], si32>, [[ARG4:%.*]]: !mo.tensor<[], f32>, [[ARG5:%.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[[SH0]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[2, [[D0]], [[D1]], [[D2]]], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[[D3]], 5], f32>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.roi_align<() -> [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]], [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]]>([[V0]], [[V1]], [[ARG2]], [[ARG3]], [[ARG4]], [[ARG5]]) {aligned = false, mode = "[[AVG:[A-Za-z0-9_]*]]"} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>, !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[RMO_D7]], [[RMO_D8]], [[RMO_D9]], [[RMO_D10]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D7]], [[RMO_D8]], [[RMO_D9]], [[RMO_D10]]], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V3]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @roi_align_static<() -> Sh0: !mosh.ape>(%input: !mo.tensor<[1, 10, 10, 1], f32>, %rois: !mo.tensor<[1, 5], f32>, %output_height: !mo.tensor<[], si64>, %output_width: !mo.tensor<[], si32>, %spatial_scale: !mo.tensor<[], f32>, %sampling_ratio: !mo.tensor<[], f32>) -> !mo.tensor<Sh0, f32> {
  %0 = rmo.mo.roi_align<() -> Sh1: !mosh.ape>(%input, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio) {aligned = false,  mode = "AVG"} : (!mo.tensor<[1, 10, 10, 1], f32>, !mo.tensor<[1, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @roi_align_static<() -> [[SH0:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[1, 10, 10, 1], f32>, [[ARG1:%.*]]: !mo.tensor<[1, 5], f32>, [[ARG2:%.*]]: !mo.tensor<[], si64>, [[ARG3:%.*]]: !mo.tensor<[], si32>, [[ARG4:%.*]]: !mo.tensor<[], f32>, [[ARG5:%.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[[SH0]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[1, 10, 10, 1], f32>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[1, 5], f32>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.roi_align<() -> [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]], [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]]>([[V0]], [[V1]], [[ARG2]], [[ARG3]], [[ARG4]], [[ARG5]]) {aligned = false, mode = "[[AVG:[A-Za-z0-9_]*]]"} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]]], f32>, !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[RMO_D7]], [[RMO_D8]], [[RMO_D9]], [[RMO_D10]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D7]], [[RMO_D8]], [[RMO_D9]], [[RMO_D10]]], f32>) -> !mo.tensor<[[SH1]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V3]] : !mo.tensor<[[SH1]], f32>
