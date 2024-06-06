// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @slice_dyn(%arg0: !mo.tensor<?, f32>, %start: !mo.tensor<?, si64>, %stop: !mo.tensor<?, si32>, %step: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @slice_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si32>, [[ARG3:%.*]]: !mo.tensor<[[SH4]], si64>) -> !mo.tensor<[[SH5]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH7]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si32> -> !mo.tensor<[[SH8]], si32>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG3]]) : !mo.tensor<[[SH4]], si64> -> !mo.tensor<[[SH9]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH11:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[RMO_SH11]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D12:[A-Za-z0-9_]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH7]], si64>) -> !mo.tensor<[[[RMO_D12]]], si64>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH8]], si32>) -> !mo.tensor<[[[RMO_D12]]], si32>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH9]], si64>) -> !mo.tensor<[[[RMO_D12]]], si64>
//  CHECK-NEXT:   [[V8:%.*]] = mo.slice<() -> [[RMO_SH13:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]], [[V5]], [[V6]], [[V7]]) : (!mo.tensor<[[RMO_SH11]], f32>, !mo.tensor<[[[RMO_D12]]], si64>, !mo.tensor<[[[RMO_D12]]], si32>, !mo.tensor<[[[RMO_D12]]], si64>) -> !mo.tensor<[[RMO_SH13]], f32>
//  CHECK-NEXT:   [[V9:%.*]] = rmo.rebind_tensor_shape<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[V8]]) {{.*}} : (!mo.tensor<[[RMO_SH13]], f32>) -> !mo.tensor<[[SH10]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH10]]> [[V9]] : !mo.tensor<[[SH10]], f32>

// -----

mo.graph @slice_dyn_params(%arg0: !mo.tensor<[?, ?], f32>, %start: !mo.tensor<?, si64>, %stop: !mo.tensor<?, si32>, %step: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[?, ?], f32>, !mo.tensor<?, si64>, !mo.tensor<?, si32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @slice_dyn_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape, [[SH5:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH3]], si64>, [[ARG2:%.*]]: !mo.tensor<[[SH4]], si32>, [[ARG3:%.*]]: !mo.tensor<[[SH5]], si64>) -> !mo.tensor<[[SH6]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D7:[A-Za-z0-9_]*]], [[D8:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D7]], [[D8]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH9]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH4]], si32> -> !mo.tensor<[[SH10]], si32>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[SH11:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG3]]) : !mo.tensor<[[SH5]], si64> -> !mo.tensor<[[SH11]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D13:[A-Za-z0-9_]*]], [[RMO_D14:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]]], f32>) -> !mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH9]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH10]], si32>) -> !mo.tensor<[2], si32>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH11]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V8:%.*]] = mo.slice<() -> [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]]>([[V4]], [[V5]], [[V6]], [[V7]]) : (!mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>
//  CHECK-NEXT:   [[V9:%.*]] = rmo.rebind_tensor_shape<() -> [[SH12:[A-Za-z0-9_]*]]: !mosh.ape>([[V8]]) {{.*}} : (!mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>) -> !mo.tensor<[[SH12]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH12]]> [[V9]] : !mo.tensor<[[SH12]], f32>

// -----

mo.graph @slice_params(%arg0: !mo.tensor<[?, ?], f32>, %start: !mo.tensor<[?], si64>, %stop: !mo.tensor<[?], si32>, %step: !mo.tensor<[?], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?], si64>, !mo.tensor<[?], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @slice_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]], [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]] -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D3]]], si64>, [[ARG2:%.*]]: !mo.tensor<[[[D4]]], si32>, [[ARG3:%.*]]: !mo.tensor<[[[D5]]], si64>) -> !mo.tensor<[[SH6]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D7:[A-Za-z0-9_]*]], [[D8:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D7]], [[D8]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D9:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D3]]], si64> -> !mo.tensor<[[[D9]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[D10:[A-Za-z0-9_]*]]>([[ARG2]]) : !mo.tensor<[[[D4]]], si32> -> !mo.tensor<[[[D10]]], si32>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[D11:[A-Za-z0-9_]*]]>([[ARG3]]) : !mo.tensor<[[[D5]]], si64> -> !mo.tensor<[[[D11]]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D13:[A-Za-z0-9_]*]], [[RMO_D14:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]]], f32>) -> !mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D9]]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[[D10]]], si32>) -> !mo.tensor<[2], si32>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D11]]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V8:%.*]] = mo.slice<() -> [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]]>([[V4]], [[V5]], [[V6]], [[V7]]) : (!mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>
//  CHECK-NEXT:   [[V9:%.*]] = rmo.rebind_tensor_shape<() -> [[SH12:[A-Za-z0-9_]*]]: !mosh.ape>([[V8]]) {{.*}} : (!mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>) -> !mo.tensor<[[SH12]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH12]]> [[V9]] : !mo.tensor<[[SH12]], f32>

// -----

mo.graph @slice_params_mix(%arg0: !mo.tensor<[3, ?], f32>, %start: !mo.tensor<[?], si64>, %stop: !mo.tensor<[2], si32>, %step: !mo.tensor<[?], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[3, ?], f32>, !mo.tensor<[?], si64>, !mo.tensor<[2], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @slice_params_mix<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]] -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[3, [[D1]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D2]]], si64>, [[ARG2:%.*]]: !mo.tensor<[2], si32>, [[ARG3:%.*]]: !mo.tensor<[[[D3]]], si64>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D5:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[3, [[D1]]], f32> -> !mo.tensor<[3, [[D5]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D6:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D2]]], si64> -> !mo.tensor<[[[D6]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[D7:[A-Za-z0-9_]*]]>([[ARG3]]) : !mo.tensor<[[[D3]]], si64> -> !mo.tensor<[[[D7]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[3, [[D5]]], f32>) -> !mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D6]]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[[D7]]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V6:%.*]] = mo.slice<() -> [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]]>([[V3]], [[V4]], [[ARG2]], [[V5]]) : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V6]]) {{.*}} : (!mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V7]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @slice_dyn_mix(%arg0: !mo.tensor<?, f32>, %start: !mo.tensor<[?], si64>, %stop: !mo.tensor<[2], si32>, %step: !mo.tensor<[?], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<?, f32>, !mo.tensor<[?], si64>, !mo.tensor<[2], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @slice_dyn_mix<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[D2:[A-Za-z0-9_]*]], [[D3:[A-Za-z0-9_]*]] -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[[D2]]], si64>, [[ARG2:%.*]]: !mo.tensor<[2], si32>, [[ARG3:%.*]]: !mo.tensor<[[[D3]]], si64>) -> !mo.tensor<[[SH4]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH5]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[D6:[A-Za-z0-9_]*]]>([[ARG1]]) : !mo.tensor<[[[D2]]], si64> -> !mo.tensor<[[[D6]]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[D7:[A-Za-z0-9_]*]]>([[ARG3]]) : !mo.tensor<[[[D3]]], si64> -> !mo.tensor<[[[D7]]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D9:[A-Za-z0-9_]*]], [[RMO_D10:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D6]]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[[D7]]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V6:%.*]] = mo.slice<() -> [[RMO_D11:[A-Za-z0-9_]*]], [[RMO_D12:[A-Za-z0-9_]*]]>([[V3]], [[V4]], [[ARG2]], [[V5]]) : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[V6]]) {{.*}} : (!mo.tensor<[[[RMO_D11]], [[RMO_D12]]], f32>) -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH8]]> [[V7]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @slice_dyn_out_params(%arg0: !mo.tensor<?, f32>, %start: !mo.tensor<?, si64>, %stop: !mo.tensor<?, si32>, %step: !mo.tensor<?, si64>) -> !mo.tensor<[?, ?], f32> {
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si32>, !mo.tensor<?, si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @slice_dyn_out_params<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape -> [[D5:[A-Za-z0-9_]*]], [[D6:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si32>, [[ARG3:%.*]]: !mo.tensor<[[SH4]], si64>) -> !mo.tensor<[[[D5]], [[D6]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH7]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH8]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si32> -> !mo.tensor<[[SH9]], si32>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG3]]) : !mo.tensor<[[SH4]], si64> -> !mo.tensor<[[SH10]], si64>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D13:[A-Za-z0-9_]*]], [[RMO_D14:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH7]], f32>) -> !mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH8]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH9]], si32>) -> !mo.tensor<[2], si32>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH10]], si64>) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[V8:%.*]] = mo.slice<() -> [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]]>([[V4]], [[V5]], [[V6]], [[V7]]) : (!mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>
//  CHECK-NEXT:   [[V9:%.*]] = rmo.rebind_tensor_shape<() -> [[D11:[A-Za-z0-9_]*]], [[D12:[A-Za-z0-9_]*]]>([[V8]]) {{.*}} : (!mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>) -> !mo.tensor<[[[D11]], [[D12]]], f32>
//  CHECK-NEXT:   mo.output<[[D11]], [[D12]]> [[V9]] : !mo.tensor<[[[D11]], [[D12]]], f32>

// -----

mo.graph @slice_static(%arg0: !mo.tensor<[3, 4], f32>, %start: !mo.tensor<[2], si64>, %stop: !mo.tensor<[2], si32>, %step: !mo.tensor<[2], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[3, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @slice_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[3, 4], f32>, [[ARG1:%.*]]: !mo.tensor<[2], si64>, [[ARG2:%.*]]: !mo.tensor<[2], si32>, [[ARG3:%.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[[SH1]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[3, 4], f32>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.slice<() -> [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]]>([[V0]], [[ARG1]], [[ARG2]], [[ARG3]]) : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>) -> !mo.tensor<[[SH2]], f32>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V2]] : !mo.tensor<[[SH2]], f32>
