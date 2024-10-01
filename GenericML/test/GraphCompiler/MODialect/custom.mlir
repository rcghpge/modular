// RUN: mo-opt --canonicalize %s| FileCheck %s

// CHECK-LABEL: mo.graph @custom_dynamic
// CHECK-SAME:  (%[[ARG0:.*]]: !mo.tensor<[?], f32>) -> !mo.tensor<[?], f32>
// CHECK-NEXT:  %[[CUSTOM:.*]] = mo.custom {symbol = "test_custom_op"}(%[[ARG0]])
// CHECK-NEXT:  mo.output %[[CUSTOM]]

mo.graph @custom_dynamic(%arg0 : !mo.tensor<[?], f32>) -> !mo.tensor<[?], f32> {
  %0 = mo.custom {symbol = "test_custom_op"}(%arg0) : (!mo.tensor<[?], f32>) -> !mo.tensor<[?], f32>
  mo.output %0 : !mo.tensor<[?], f32>
}

// CHECK-LABEL:      mo.graph @custom_dynamic_with_device_annotations
// CHECK-NEXT:  %[[CUSTOM:.*]] = mo.custom {operandDevices = #mo<DeviceIndex[(0 : index, <"cpu", 0>), (1 : index, <"cpu", 0>)]>, symbol = "test_custom_op"}
mo.graph @custom_dynamic_with_device_annotations(%arg0 : !mo.tensor<[?], f32>, %arg1 : !mo.tensor<[?], f32>, %arg2 : !mo.tensor<[?], f32>) -> !mo.tensor<[?], f32> {
  %0 = mo.custom {
    operandDevices = #mo<DeviceIndex[(0 : index, <"cpu", 0>), (1 : index, <"cpu", 0>)]>,
    symbol = "test_custom_op"}
    (%arg0, %arg1, %arg2) : (!mo.tensor<[?], f32>, !mo.tensor<[?], f32>, !mo.tensor<[?], f32>) -> !mo.tensor<[?], f32>
  mo.output %0 : !mo.tensor<[?], f32>
}

// CHECK-LABEL: mo.graph @inplace_custom_op
// CHECK-SAME: [[arg0:%.*]]: !mo.buffer<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: mo.custom {symbol = "inplace"}([[arg0]], [[ch0]])
mo.graph @inplace_custom_op(%arg0: !mo.buffer<[10, 10], f32>) -> !mo.chain {
  %ch0 = mo.chain.create()

  %ch1 = mo.custom {symbol = "inplace"} (%arg0, %ch0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> !mo.chain

  mo.output %ch1 : !mo.chain
}
