// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @slice_dyn(%arg0: !mo.tensor<?, f32>, %start: !mo.tensor<?, si64>, %stop: !mo.tensor<?, si32>, %step: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @slice_dyn<[[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape, [[SH3:[^ ]*]]: !mosh.ape, [[SH4:[^ ]*]]: !mosh.ape -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH2]], si64>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH3]], si32>, [[VARG3:[^ ]*]]: !mo.tensor<[[SH4]], si64>) -> !mo.tensor<[[SH5]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH7:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH7]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[SH8:[^ ]*]]: !mosh.ape>([[VARG2]]) : !mo.tensor<[[SH3]], si32> -> !mo.tensor<[[SH8]], si32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[SH9:[^ ]*]]: !mosh.ape>([[VARG3]]) : !mo.tensor<[[SH4]], si64> -> !mo.tensor<[[SH9]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH11:[^ ]*]]: !mosh.ape>([[V0]])
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D12:[^ ]*]]>([[V1]])
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape([[V2]])
// CHECK-NEXT: [[V7:[^ ]*]] = rmo.rebind_tensor_shape([[V3]])
// CHECK-NEXT: [[V8:[^ ]*]] = mo.slice<() -> [[RMO_SH13:[^ ]*]]: !mosh.ape>([[V4]], [[V5]], [[V6]], [[V7]]) : (!mo.tensor<[[RMO_SH11]], f32>, !mo.tensor<[[[RMO_D12]]], si64>, !mo.tensor<[[[RMO_D12]]], si32>, !mo.tensor<[[[RMO_D12]]], si64>) -> !mo.tensor<[[RMO_SH13]], f32>
// CHECK-NEXT: [[V9:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH10:[^ ]*]]: !mosh.ape>([[V8]])
// CHECK-NEXT: mo.output<:!mosh.ape [[SH10]]> [[V9]] : !mo.tensor<[[SH10]], f32>
// -----

mo.graph @slice_dyn_params(%arg0: !mo.tensor<[?, ?], f32>, %start: !mo.tensor<?, si64>, %stop: !mo.tensor<?, si32>, %step: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[?, ?], f32>, !mo.tensor<?, si64>, !mo.tensor<?, si32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @slice_dyn_params<[[D1:[^ ]*]], [[D2:[^ ]*]], [[SH3:[^ ]*]]: !mosh.ape, [[SH4:[^ ]*]]: !mosh.ape, [[SH5:[^ ]*]]: !mosh.ape -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH3]], si64>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH4]], si32>, [[VARG3:[^ ]*]]: !mo.tensor<[[SH5]], si64>) -> !mo.tensor<[[SH6]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D7]], [[D8]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH9:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH9]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[SH10:[^ ]*]]: !mosh.ape>([[VARG2]]) : !mo.tensor<[[SH4]], si32> -> !mo.tensor<[[SH10]], si32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[SH11:[^ ]*]]: !mosh.ape>([[VARG3]]) : !mo.tensor<[[SH5]], si64> -> !mo.tensor<[[SH11]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D13:[^ ]*]], [[RMO_D14:[^ ]*]]>([[V0]])
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape([[V1]])
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape([[V2]])
// CHECK-NEXT: [[V7:[^ ]*]] = rmo.rebind_tensor_shape([[V3]])
// CHECK-NEXT: [[V8:[^ ]*]] = mo.slice<() -> [[SH12:[^ ]*]]: !mosh.ape>([[V4]], [[V5]], [[V6]], [[V7]]) : (!mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[SH12]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH12]]> [[V8]] : !mo.tensor<[[SH12]], f32>

// -----

mo.graph @slice_params(%arg0: !mo.tensor<[?, ?], f32>, %start: !mo.tensor<[?], si64>, %stop: !mo.tensor<[?], si32>, %step: !mo.tensor<[?], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?], si64>, !mo.tensor<[?], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @slice_params<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]] -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D3]]], si64>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D4]]], si32>, [[VARG3:[^ ]*]]: !mo.tensor<[[[D5]]], si64>) -> !mo.tensor<[[SH6]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D7]], [[D8]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D9:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D3]]], si64> -> !mo.tensor<[[[D9]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D10:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D4]]], si32> -> !mo.tensor<[[[D10]]], si32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D11:[^ ]*]]>([[VARG3]]) : !mo.tensor<[[[D5]]], si64> -> !mo.tensor<[[[D11]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D13:[^ ]*]], [[RMO_D14:[^ ]*]]>([[V0]])
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape([[V1]])
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape([[V2]])
// CHECK-NEXT: [[V7:[^ ]*]] = rmo.rebind_tensor_shape([[V3]])
// CHECK-NEXT: [[V8:[^ ]*]] = mo.slice<() -> [[SH12:[^ ]*]]: !mosh.ape>([[V4]], [[V5]], [[V6]], [[V7]]) : (!mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[SH12]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH12]]> [[V8]] : !mo.tensor<[[SH12]], f32>

// -----

mo.graph @slice_params_mix(%arg0: !mo.tensor<[3, ?], f32>, %start: !mo.tensor<[?], si64>, %stop: !mo.tensor<[2], si32>, %step: !mo.tensor<[?], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[3, ?], f32>, !mo.tensor<[?], si64>, !mo.tensor<[2], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @slice_params_mix<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]] -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[3, [[D1]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]]], si64>, [[VARG2:[^ ]*]]: !mo.tensor<[2], si32>, [[VARG3:[^ ]*]]: !mo.tensor<[[[D3]]], si64>) -> !mo.tensor<[[SH4]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]]>([[VARG0]]) : !mo.tensor<[3, [[D1]]], f32> -> !mo.tensor<[3, [[D5]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D6:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]]], si64> -> !mo.tensor<[[[D6]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]]>([[VARG3]]) : !mo.tensor<[[[D3]]], si64> -> !mo.tensor<[[[D7]]], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D9:[^ ]*]], [[RMO_D10:[^ ]*]]>([[V0]])
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V1]])
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape([[V2]])
// CHECK-NEXT: [[V6:[^ ]*]] = mo.slice<() -> [[SH8:[^ ]*]]: !mosh.ape>([[V3]], [[V4]], [[VARG2]], [[V5]]) : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[SH8]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH8]]> [[V6]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @slice_dyn_mix(%arg0: !mo.tensor<?, f32>, %start: !mo.tensor<[?], si64>, %stop: !mo.tensor<[2], si32>, %step: !mo.tensor<[?], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<?, f32>, !mo.tensor<[?], si64>, !mo.tensor<[2], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @slice_dyn_mix<[[SH1:[^ ]*]]: !mosh.ape, [[D2:[^ ]*]], [[D3:[^ ]*]] -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]]], si64>, [[VARG2:[^ ]*]]: !mo.tensor<[2], si32>, [[VARG3:[^ ]*]]: !mo.tensor<[[[D3]]], si64>) -> !mo.tensor<[[SH4]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D6:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]]], si64> -> !mo.tensor<[[[D6]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]]>([[VARG3]]) : !mo.tensor<[[[D3]]], si64> -> !mo.tensor<[[[D7]]], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D9:[^ ]*]], [[RMO_D10:[^ ]*]]>([[V0]])
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V1]])
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape([[V2]])
// CHECK-NEXT: [[V6:[^ ]*]] = mo.slice<() -> [[SH8:[^ ]*]]: !mosh.ape>([[V3]], [[V4]], [[VARG2]], [[V5]]) : (!mo.tensor<[[[RMO_D9]], [[RMO_D10]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[SH8]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH8]]> [[V6]] : !mo.tensor<[[SH8]], f32>

// -----

mo.graph @slice_dyn_out_params(%arg0: !mo.tensor<?, f32>, %start: !mo.tensor<?, si64>, %stop: !mo.tensor<?, si32>, %step: !mo.tensor<?, si64>) -> !mo.tensor<[?, ?], f32> {
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si32>, !mo.tensor<?, si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @slice_dyn_out_params<[[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape, [[SH3:[^ ]*]]: !mosh.ape, [[SH4:[^ ]*]]: !mosh.ape -> [[D5:[^ ]*]], [[D6:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH2]], si64>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH3]], si32>, [[VARG3:[^ ]*]]: !mo.tensor<[[SH4]], si64>) -> !mo.tensor<[[[D5]], [[D6]]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH7:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH7]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH8:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH8]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[SH9:[^ ]*]]: !mosh.ape>([[VARG2]]) : !mo.tensor<[[SH3]], si32> -> !mo.tensor<[[SH9]], si32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[SH10:[^ ]*]]: !mosh.ape>([[VARG3]]) : !mo.tensor<[[SH4]], si64> -> !mo.tensor<[[SH10]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D13:[^ ]*]], [[RMO_D14:[^ ]*]]>([[V0]])
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape([[V1]])
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape([[V2]])
// CHECK-NEXT: [[V7:[^ ]*]] = rmo.rebind_tensor_shape([[V3]])
// CHECK-NEXT: [[V8:[^ ]*]] = mo.slice<() -> [[D11:[^ ]*]], [[D12:[^ ]*]]>([[V4]], [[V5]], [[V6]], [[V7]]) : (!mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[D11]], [[D12]]], f32>
// CHECK-NEXT: mo.output<[[D11]], [[D12]]> [[V8]] : !mo.tensor<[[[D11]], [[D12]]], f32>

// -----

mo.graph @slice_static(%arg0: !mo.tensor<[3, 4], f32>, %start: !mo.tensor<[2], si64>, %stop: !mo.tensor<[2], si32>, %step: !mo.tensor<[2], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[3, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @slice_static<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[3, 4], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[2], si64>, [[VARG2:[^ ]*]]: !mo.tensor<[2], si32>, [[VARG3:[^ ]*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[[SH1]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[RMO_D3:[^ ]*]], [[RMO_D4:[^ ]*]]>([[VARG0]])
// CHECK-NEXT: [[V1:[^ ]*]] = mo.slice<() -> [[SH2:[^ ]*]]: !mosh.ape>([[V0]], [[VARG1]], [[VARG2]], [[VARG3]]) : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>

// -----

// Checks we properly forward the annotate type + the original param decls
mo.graph @slice_static_with_decls<() -> O1, O2>(%arg0: !mo.tensor<[3, 4], f32>, %start: !mo.tensor<[2], si64>, %stop: !mo.tensor<[2], si32>, %step: !mo.tensor<[2], si64>) -> !mo.tensor<[O1, O2], f32> {
  %0 = rmo.mo.slice<()->A, B>(%arg0, %start, %stop, %step) : (!mo.tensor<[3, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[A, B], f32>
  mo.output<A, B> %0 : !mo.tensor<[A, B], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @slice_static_with_decls
// CHECK: mo.slice<() -> A, B>
// CHECK-SAME: -> !mo.tensor<[A, B], f32>
