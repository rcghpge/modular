// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @roi_align_dyn(%input: !mo.tensor<?, f32>, %rois: !mo.tensor<?, f32>, %output_height: !mo.tensor<?, si64>, %output_width: !mo.tensor<?, si32>, %spatial_scale: !mo.tensor<?, f32>, %sampling_ratio: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.roi_align(%input, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio) {aligned = false,  mode = "AVG"} : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si32>, !mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @roi_align_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape, [[SH5:[A-Za-z0-9_]*]]: !mosh.ape, [[SH6:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>, [[ARG3:%.*]]: !mo.tensor<[[SH4]], si32>, [[ARG4:%.*]]: !mo.tensor<[[SH5]], f32>, [[ARG5:%.*]]: !mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[SH7]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH9]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH10]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[SH11:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG3]]) : !mo.tensor<[[SH4]], si32> -> !mo.tensor<[[SH11]], si32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.rebind<() -> [[SH12:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG4]]) : !mo.tensor<[[SH5]], f32> -> !mo.tensor<[[SH12]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = mo.rebind<() -> [[SH13:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG5]]) : !mo.tensor<[[SH6]], f32> -> !mo.tensor<[[SH13]], f32>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]], [[RMO_D17:[A-Za-z0-9_]*]], [[RMO_D18:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH8]], f32>) -> !mo.tensor<[[[RMO_D15]], [[RMO_D16]], [[RMO_D17]], [[RMO_D18]]], f32>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D19:[A-Za-z0-9_]*]], [[RMO_D20:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH9]], f32>) -> !mo.tensor<[[[RMO_D19]], [[RMO_D20]]], f32>
//  CHECK-NEXT:   [[V8:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH10]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V9:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH11]], si32>) -> !mo.tensor<[], si32>
//  CHECK-NEXT:   [[V10:%.*]] = rmo.rebind_tensor_shape([[V4]]) {{.*}} : (!mo.tensor<[[SH12]], f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   [[V11:%.*]] = rmo.rebind_tensor_shape([[V5]]) {{.*}} : (!mo.tensor<[[SH13]], f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   [[V12:%.*]] = mo.roi_align<() -> [[RMO_D21:[A-Za-z0-9_]*]], [[RMO_D22:[A-Za-z0-9_]*]], [[RMO_D23:[A-Za-z0-9_]*]], [[RMO_D24:[A-Za-z0-9_]*]]>([[V6]], [[V7]], [[V8]], [[V9]], [[V10]], [[V11]]) {aligned = false, mode = "[[AVG:[A-Za-z0-9_]*]]"} : (!mo.tensor<[[[RMO_D15]], [[RMO_D16]], [[RMO_D17]], [[RMO_D18]]], f32>, !mo.tensor<[[[RMO_D19]], [[RMO_D20]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[RMO_D21]], [[RMO_D22]], [[RMO_D23]], [[RMO_D24]]], f32>
//  CHECK-NEXT:   [[V13:%.*]] = rmo.rebind_tensor_shape<() -> [[SH14:[A-Za-z0-9_]*]]: !mosh.ape>([[V12]]) {{.*}} : (!mo.tensor<[[[RMO_D21]], [[RMO_D22]], [[RMO_D23]], [[RMO_D24]]], f32>) -> !mo.tensor<[[SH14]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH14]]> [[V13]] : !mo.tensor<[[SH14]], f32>

// -----

mo.graph @roi_align_params(%input: !mo.tensor<[?, ?, ?, ?], f32>, %rois: !mo.tensor<[?, ?], f32>, %output_height: !mo.tensor<[], si64>, %output_width: !mo.tensor<[], si32>, %spatial_scale: !mo.tensor<[], f32>, %sampling_ratio: !mo.tensor<[], f32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.roi_align(%input, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio) {aligned = false,  mode = "AVG"} : (!mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[?, ?], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @roi_align_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]], [[D6:[A-Za-z0-9_]*]] -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D5]], [[D6]]], f32>, [[ARG2:%.*]]: !mo.tensor<[], si64>, [[ARG3:%.*]]: !mo.tensor<[], si32>, [[ARG4:%.*]]: !mo.tensor<[], f32>, [[ARG5:%.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[[SH7]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D12:[A-Za-z0-9_]*]], [[D13:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D5]], [[D6]]], f32> -> !mo.tensor<[[[D12]], [[D13]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]], [[RMO_D17:[A-Za-z0-9_]*]], [[RMO_D18:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>) -> !mo.tensor<[[[RMO_D15]], [[RMO_D16]], [[RMO_D17]], [[RMO_D18]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D19:[A-Za-z0-9_]*]], [[RMO_D20:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[D12]], [[D13]]], f32>) -> !mo.tensor<[[[RMO_D19]], [[RMO_D20]]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.roi_align<() -> [[RMO_D21:[A-Za-z0-9_]*]], [[RMO_D22:[A-Za-z0-9_]*]], [[RMO_D23:[A-Za-z0-9_]*]], [[RMO_D24:[A-Za-z0-9_]*]]>([[V2]], [[V3]], [[ARG2]], [[ARG3]], [[ARG4]], [[ARG5]]) {aligned = false, mode = "[[AVG:[A-Za-z0-9_]*]]"} : (!mo.tensor<[[[RMO_D15]], [[RMO_D16]], [[RMO_D17]], [[RMO_D18]]], f32>, !mo.tensor<[[[RMO_D19]], [[RMO_D20]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[RMO_D21]], [[RMO_D22]], [[RMO_D23]], [[RMO_D24]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH14:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D21]], [[RMO_D22]], [[RMO_D23]], [[RMO_D24]]], f32>) -> !mo.tensor<[[SH14]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH14]]> [[V5]] : !mo.tensor<[[SH14]], f32>

// -----

mo.graph @roi_align_params_mix(%input: !mo.tensor<[2, ?, ?, ?], f32>, %rois: !mo.tensor<[?, 5], f32>, %output_height: !mo.tensor<[], si64>, %output_width: !mo.tensor<[], si32>, %spatial_scale: !mo.tensor<[], f32>, %sampling_ratio: !mo.tensor<[], f32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.roi_align(%input, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio) {aligned = false,  mode = "AVG"} : (!mo.tensor<[2, ?, ?, ?], f32>, !mo.tensor<[?, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @roi_align_params_mix<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]] -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, [[D1]], [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D4]], 5], f32>, [[ARG2:%.*]]: !mo.tensor<[], si64>, [[ARG3:%.*]]: !mo.tensor<[], si32>, [[ARG4:%.*]]: !mo.tensor<[], f32>, [[ARG5:%.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]], [[D8:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[2, [[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[2, [[D6]], [[D7]], [[D8]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D9:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D4]], 5], f32> -> !mo.tensor<[[[D9]], 5], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]], [[RMO_D13:[A-Za-z0-9_]*]], [[RMO_D14:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[2, [[D6]], [[D7]], [[D8]]], f32>) -> !mo.tensor<[[[RMO_D11]], [[RMO_D12]], [[RMO_D13]], [[RMO_D14]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[D9]], 5], f32>) -> !mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.roi_align<() -> [[RMO_D17:[A-Za-z0-9_]*]], [[RMO_D18:[A-Za-z0-9_]*]], [[RMO_D19:[A-Za-z0-9_]*]], [[RMO_D20:[A-Za-z0-9_]*]]>([[V2]], [[V3]], [[ARG2]], [[ARG3]], [[ARG4]], [[ARG5]]) {aligned = false, mode = "[[AVG:[A-Za-z0-9_]*]]"} : (!mo.tensor<[[[RMO_D11]], [[RMO_D12]], [[RMO_D13]], [[RMO_D14]]], f32>, !mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[RMO_D17]], [[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D17]], [[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], f32>) -> !mo.tensor<[[SH10]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH10]]> [[V5]] : !mo.tensor<[[SH10]], f32>

// -----

mo.graph @roi_align_static(%input: !mo.tensor<[1, 10, 10, 1], f32>, %rois: !mo.tensor<[1, 5], f32>, %output_height: !mo.tensor<[], si64>, %output_width: !mo.tensor<[], si32>, %spatial_scale: !mo.tensor<[], f32>, %sampling_ratio: !mo.tensor<[], f32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.roi_align(%input, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio) {aligned = false,  mode = "AVG"} : (!mo.tensor<[1, 10, 10, 1], f32>, !mo.tensor<[1, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @roi_align_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[1, 10, 10, 1], f32>, [[ARG1:%.*]]: !mo.tensor<[1, 5], f32>, [[ARG2:%.*]]: !mo.tensor<[], si64>, [[ARG3:%.*]]: !mo.tensor<[], si32>, [[ARG4:%.*]]: !mo.tensor<[], f32>, [[ARG5:%.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[1, 10, 10, 1], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[1, 5], f32>) -> !mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.roi_align<() -> [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]], [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]]>([[V0]], [[V1]], [[ARG2]], [[ARG3]], [[ARG4]], [[ARG5]]) {aligned = false, mode = "[[AVG:[A-Za-z0-9_]*]]"} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]], [[RMO_D6]]], f32>, !mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[RMO_D9]], [[RMO_D10]], [[RMO_D11]], [[RMO_D12]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]], [[RMO_D11]], [[RMO_D12]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V3]] : !mo.tensor<[[SH2]], f32>
