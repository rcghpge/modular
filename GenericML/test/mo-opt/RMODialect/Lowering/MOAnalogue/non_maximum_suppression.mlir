// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @nms_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape -> Sh5: !mosh.ape>(%boxes: !mo.tensor<Sh0, f32>, %scores: !mo.tensor<Sh1, f32>, %maxb: !mo.tensor<Sh2, si64>, %iou_thres: !mo.tensor<Sh3, f32>, %scores_thres: !mo.tensor<Sh4, f32>) -> !mo.tensor<Sh5, si64> {
  %0 = rmo.mo.non_maximum_suppression<() -> Sh6: !mosh.ape>(%boxes, %scores, %maxb, %iou_thres, %scores_thres) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, f32>, !mo.tensor<Sh4, f32>) -> !mo.tensor<Sh6, si64>
  mo.output<:!mosh.ape Sh6> %0 : !mo.tensor<Sh6, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @nms_dyn<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH2]], si64>, [[ARG3:%.*]]: !mo.tensor<[[SH3]], f32>, [[ARG4:%.*]]: !mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[SH5]], si64> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D0:[A-Za-z0-9_]*]], [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[[RMO_D0]], [[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[ARG2]]) {{.*}} : (!mo.tensor<[[SH2]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[ARG3]]) {{.*}} : (!mo.tensor<[[SH3]], f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape([[ARG4]]) {{.*}} : (!mo.tensor<[[SH4]], f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   [[V5:%.*]] = mo.non_maximum_suppression<() -> [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]]>([[V0]], [[V1]], [[V2]], [[V3]], [[V4]]) : (!mo.tensor<[[[RMO_D0]], [[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[RMO_D6]], [[RMO_D7]]], si64>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[RMO_D6]], [[RMO_D7]]], si64>) -> !mo.tensor<[[SH6]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]]> [[V6]] : !mo.tensor<[[SH6]], si64>

// -----

mo.graph @nms_dyn_params<Sh0: !mosh.ape, D1, D2, D3, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(%boxes: !mo.tensor<Sh0, f32>, %scores: !mo.tensor<[D1, D2, D3], f32>, %maxb: !mo.tensor<Sh1, si64>, %iou_thres: !mo.tensor<[], f32>, %scores_thres: !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, si64> {
  %0 = rmo.mo.non_maximum_suppression<() -> Sh4: !mosh.ape>(%boxes, %scores, %maxb, %iou_thres, %scores_thres) : (!mo.tensor<Sh0, f32>, !mo.tensor<[D1, D2, D3], f32>, !mo.tensor<Sh1, si64>, !mo.tensor<[], f32>, !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh4, si64>
  mo.output<:!mosh.ape Sh4> %0 : !mo.tensor<Sh4, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @nms_dyn_params<[[SH0:[A-Za-z0-9_]*]]: !mosh.ape, [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH0]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH1]], si64>, [[ARG3:%.*]]: !mo.tensor<[], f32>, [[ARG4:%.*]]: !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[SH3]], si64> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D0:[A-Za-z0-9_]*]], [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[[RMO_D0]], [[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape([[ARG2]]) {{.*}} : (!mo.tensor<[[SH1]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[ARG4]]) {{.*}} : (!mo.tensor<[[SH2]], f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.non_maximum_suppression<() -> [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]]>([[V0]], [[V1]], [[V2]], [[ARG3]], [[V3]]) : (!mo.tensor<[[[RMO_D0]], [[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[RMO_D6]], [[RMO_D7]]], si64>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D6]], [[RMO_D7]]], si64>) -> !mo.tensor<[[SH4]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH4]]> [[V5]] : !mo.tensor<[[SH4]], si64>

// -----

mo.graph @nms_params<D0, D1, D2, D3, D4, D5 -> Sh0: !mosh.ape>(%boxes: !mo.tensor<[D0, D1, D2], f32>, %scores: !mo.tensor<[D3, D4, D5], f32>, %maxb: !mo.tensor<[], si64>, %iou_thres: !mo.tensor<[], f32>, %scores_thres: !mo.tensor<[], f32>) -> !mo.tensor<Sh0, si64> {
  %0 = rmo.mo.non_maximum_suppression<() -> Sh1: !mosh.ape>(%boxes, %scores, %maxb, %iou_thres, %scores_thres) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[D3, D4, D5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<Sh1, si64>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @nms_params<[[D0:[A-Za-z0-9_]*]], [[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]] -> [[SH0:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>, [[ARG2:%.*]]: !mo.tensor<[], si64>, [[ARG3:%.*]]: !mo.tensor<[], f32>, [[ARG4:%.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[[SH0]], si64> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D0:[A-Za-z0-9_]*]], [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>) -> !mo.tensor<[[[RMO_D0]], [[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.non_maximum_suppression<() -> [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]]>([[V0]], [[V1]], [[ARG2]], [[ARG3]], [[ARG4]]) : (!mo.tensor<[[[RMO_D0]], [[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[RMO_D6]], [[RMO_D7]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D6]], [[RMO_D7]]], si64>) -> !mo.tensor<[[SH1]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V3]] : !mo.tensor<[[SH1]], si64>

// -----

mo.graph @nms_static<() -> Sh0: !mosh.ape>(%boxes: !mo.tensor<[1, 6, 4], f32>, %scores: !mo.tensor<[1, 1, 6], f32>, %maxb: !mo.tensor<[], si64>, %iou_thres: !mo.tensor<[], f32>, %scores_thres: !mo.tensor<[], f32>) -> !mo.tensor<Sh0, si64> {
  %0 = rmo.mo.non_maximum_suppression<() -> Sh1: !mosh.ape>(%boxes, %scores, %maxb, %iou_thres, %scores_thres) : (!mo.tensor<[1, 6, 4], f32>, !mo.tensor<[1, 1, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<Sh1, si64>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @nms_static<() -> [[SH0:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[1, 6, 4], f32>, [[ARG1:%.*]]: !mo.tensor<[1, 1, 6], f32>, [[ARG2:%.*]]: !mo.tensor<[], si64>, [[ARG3:%.*]]: !mo.tensor<[], f32>, [[ARG4:%.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[[SH0]], si64> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D0:[A-Za-z0-9_]*]], [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[1, 6, 4], f32>) -> !mo.tensor<[[[RMO_D0]], [[RMO_D1]], [[RMO_D2]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[1, 1, 6], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.non_maximum_suppression<() -> [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]]>([[V0]], [[V1]], [[ARG2]], [[ARG3]], [[ARG4]]) : (!mo.tensor<[[[RMO_D0]], [[RMO_D1]], [[RMO_D2]]], f32>, !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[RMO_D6]], [[RMO_D7]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D6]], [[RMO_D7]]], si64>) -> !mo.tensor<[[SH1]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> [[V3]] : !mo.tensor<[[SH1]], si64>
