// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @nms_dyn(%boxes: !mo.tensor<?, f32>, %scores: !mo.tensor<?, f32>, %maxb: !mo.tensor<?, si64>, %iou_thres: !mo.tensor<?, f32>, %scores_thres: !mo.tensor<?, f32>) -> !mo.tensor<?, si64> {
  %0 = rmo.mo.non_maximum_suppression(%boxes, %scores, %maxb, %iou_thres, %scores_thres) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, si64>
  mo.output %0 : !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @nms_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape, [[SH5:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>, [[ARG3:%.*]]: !mo.tensor<[[SH4]], f32>, [[ARG4:%.*]]: !mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH6]], si64> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH7]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH9]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG3]]) : !mo.tensor<[[SH4]], f32> -> !mo.tensor<[[SH10]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.rebind<() -> [[SH11:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG4]]) : !mo.tensor<[[SH5]], f32> -> !mo.tensor<[[SH11]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D13:[A-Za-z0-9_]*]], [[RMO_D14:[A-Za-z0-9_]*]], [[RMO_D15:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH7]], f32>) -> !mo.tensor<[[[RMO_D13]], [[RMO_D14]], [[RMO_D15]]], f32>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D16:[A-Za-z0-9_]*]], [[RMO_D17:[A-Za-z0-9_]*]], [[RMO_D18:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH8]], f32>) -> !mo.tensor<[[[RMO_D16]], [[RMO_D17]], [[RMO_D18]]], f32>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH9]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V8:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH10]], f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   [[V9:%.*]] = rmo.rebind_tensor_shape([[V4]]) {{.*}} : (!mo.tensor<[[SH11]], f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   [[V10:%.*]] = mo.non_maximum_suppression<() -> [[RMO_D19:[A-Za-z0-9_]*]], [[RMO_D20:[A-Za-z0-9_]*]]>([[V5]], [[V6]], [[V7]], [[V8]], [[V9]]) : (!mo.tensor<[[[RMO_D13]], [[RMO_D14]], [[RMO_D15]]], f32>, !mo.tensor<[[[RMO_D16]], [[RMO_D17]], [[RMO_D18]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[RMO_D19]], [[RMO_D20]]], si64>
//  CHECK-NEXT:   [[V11:%.*]] = rmo.rebind_tensor_shape<() -> [[SH12:[A-Za-z0-9_]*]]: !mosh.ape>([[V10]]) {{.*}} : (!mo.tensor<[[[RMO_D19]], [[RMO_D20]]], si64>) -> !mo.tensor<[[SH12]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH12]]> [[V11]] : !mo.tensor<[[SH12]], si64>

// -----

mo.graph @nms_dyn_params(%boxes: !mo.tensor<?, f32>, %scores: !mo.tensor<[?, ?, ?], f32>, %maxb: !mo.tensor<?, si64>, %iou_thres: !mo.tensor<[], f32>, %scores_thres: !mo.tensor<?, f32>) -> !mo.tensor<?, si64> {
  %0 = rmo.mo.non_maximum_suppression(%boxes, %scores, %maxb, %iou_thres, %scores_thres) : (!mo.tensor<?, f32>, !mo.tensor<[?, ?, ?], f32>, !mo.tensor<?, si64>, !mo.tensor<[], f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, si64>
  mo.output %0 : !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @nms_dyn_params<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]], [[SH5:[A-Za-z0-9_]*]]: !mosh.ape, [[SH6:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32>, [[ARG2:%.*]]: !mo.tensor<[[SH5]], si64>, [[ARG3:%.*]]: !mo.tensor<[], f32>, [[ARG4:%.*]]: !mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[SH7]], si64> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D9]], [[D10]], [[D11]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH12:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH5]], si64> -> !mo.tensor<[[SH12]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[SH13:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG4]]) : !mo.tensor<[[SH6]], f32> -> !mo.tensor<[[SH13]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]], [[RMO_D17:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH8]], f32>) -> !mo.tensor<[[[RMO_D15]], [[RMO_D16]], [[RMO_D17]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D18:[A-Za-z0-9_]*]], [[RMO_D19:[A-Za-z0-9_]*]], [[RMO_D20:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[D9]], [[D10]], [[D11]]], f32>) -> !mo.tensor<[[[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], f32>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH12]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH13]], f32>) -> !mo.tensor<[], f32>
//  CHECK-NEXT:   [[V8:%.*]] = mo.non_maximum_suppression<() -> [[RMO_D21:[A-Za-z0-9_]*]], [[RMO_D22:[A-Za-z0-9_]*]]>([[V4]], [[V5]], [[V6]], [[ARG3]], [[V7]]) : (!mo.tensor<[[[RMO_D15]], [[RMO_D16]], [[RMO_D17]]], f32>, !mo.tensor<[[[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[RMO_D21]], [[RMO_D22]]], si64>
//  CHECK-NEXT:   [[V9:%.*]] = rmo.rebind_tensor_shape<() -> [[SH14:[A-Za-z0-9_]*]]: !mosh.ape>([[V8]]) {{.*}} : (!mo.tensor<[[[RMO_D21]], [[RMO_D22]]], si64>) -> !mo.tensor<[[SH14]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH14]]> [[V9]] : !mo.tensor<[[SH14]], si64>

// -----

mo.graph @nms_params(%boxes: !mo.tensor<[?, ?, ?], f32>, %scores: !mo.tensor<[?, ?, ?], f32>, %maxb: !mo.tensor<[], si64>, %iou_thres: !mo.tensor<[], f32>, %scores_thres: !mo.tensor<[], f32>) -> !mo.tensor<?, si64> {
  %0 = rmo.mo.non_maximum_suppression(%boxes, %scores, %maxb, %iou_thres, %scores_thres) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<?, si64>
  mo.output %0 : !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @nms_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]], [[D6:[A-Za-z0-9_]*]] -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>, [[ARG2:%.*]]: !mo.tensor<[], si64>, [[ARG3:%.*]]: !mo.tensor<[], f32>, [[ARG4:%.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[[SH7]], si64> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D11:[A-Za-z0-9_]*]], [[D12:[A-Za-z0-9_]*]], [[D13:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32> -> !mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]], [[RMO_D17:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>) -> !mo.tensor<[[[RMO_D15]], [[RMO_D16]], [[RMO_D17]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D18:[A-Za-z0-9_]*]], [[RMO_D19:[A-Za-z0-9_]*]], [[RMO_D20:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>) -> !mo.tensor<[[[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = mo.non_maximum_suppression<() -> [[RMO_D21:[A-Za-z0-9_]*]], [[RMO_D22:[A-Za-z0-9_]*]]>([[V2]], [[V3]], [[ARG2]], [[ARG3]], [[ARG4]]) : (!mo.tensor<[[[RMO_D15]], [[RMO_D16]], [[RMO_D17]]], f32>, !mo.tensor<[[[RMO_D18]], [[RMO_D19]], [[RMO_D20]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[RMO_D21]], [[RMO_D22]]], si64>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH14:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[RMO_D21]], [[RMO_D22]]], si64>) -> !mo.tensor<[[SH14]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH14]]> [[V5]] : !mo.tensor<[[SH14]], si64>

// -----

mo.graph @nms_static(%boxes: !mo.tensor<[1, 6, 4], f32>, %scores: !mo.tensor<[1, 1, 6], f32>, %maxb: !mo.tensor<[], si64>, %iou_thres: !mo.tensor<[], f32>, %scores_thres: !mo.tensor<[], f32>) -> !mo.tensor<?, si64> {
  %0 = rmo.mo.non_maximum_suppression(%boxes, %scores, %maxb, %iou_thres, %scores_thres) : (!mo.tensor<[1, 6, 4], f32>, !mo.tensor<[1, 1, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<?, si64>
  mo.output %0 : !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @nms_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[1, 6, 4], f32>, [[ARG1:%.*]]: !mo.tensor<[1, 1, 6], f32>, [[ARG2:%.*]]: !mo.tensor<[], si64>, [[ARG3:%.*]]: !mo.tensor<[], f32>, [[ARG4:%.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[[SH1]], si64> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[1, 6, 4], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D6:[A-Za-z0-9_]*]], [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]]>([[ARG1]]) {{.*}} : (!mo.tensor<[1, 1, 6], f32>) -> !mo.tensor<[[[RMO_D6]], [[RMO_D7]], [[RMO_D8]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = mo.non_maximum_suppression<() -> [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]]>([[V0]], [[V1]], [[ARG2]], [[ARG3]], [[ARG4]]) : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>, !mo.tensor<[[[RMO_D6]], [[RMO_D7]], [[RMO_D8]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[RMO_D9]], [[RMO_D10]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]]], si64>) -> !mo.tensor<[[SH2]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V3]] : !mo.tensor<[[SH2]], si64>
