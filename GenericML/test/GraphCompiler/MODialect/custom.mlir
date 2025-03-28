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
// CHECK: mo.custom {symbol = "inplace"}([[ch0]], [[arg0]])
mo.graph @inplace_custom_op(%arg0: !mo.buffer<[10, 10], f32>) -> !mo.chain {
  %ch0 = mo.chain.create()

  %ch1 = mo.custom {symbol = "inplace"} (%ch0, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> !mo.chain

  mo.output %ch1 : !mo.chain
}

// CHECK-LABEL: mo.graph @custom_models_memory_effects
// Custom should not be deleted since it has memory effects
// CHECK: mo.custom
mo.graph @custom_models_memory_effects<()->D>(%arg0: !mo.tensor<[10, 10], f32>, %data: !mo.tensor<[], f32>) -> !mo.tensor<[D], f32> {
  // This result is not used, but op declares parameter so should not be deleted
  %t = mo.custom <() -> A>{symbol = "example_op"} (%arg0) : (!mo.tensor<[10, 10], f32>) -> !mo.tensor<[10, 10], f32>

  %res = mo.static.broadcast_to(%data): !mo.tensor<[], f32> -> !mo.tensor<[A], f32>
  mo.output<A> %res : !mo.tensor<[A], f32>
}

// CHECK-LABEL: mo.graph @custom_op_mutable_spec
// Custom op with a mutable spec with a mix of mo.buffer and mo.opaque
// CHECK: mo.custom
mo.graph @custom_op_mutable_spec(%arg0: !mo.buffer<[2, 2], f32>,
                                 %arg1: !mo.opaque<"Foo">)
{
  %ch0 = mo.chain.create()

  %ch1 = mo.custom  {symbol = "foo", mutSpec = #mo<muts[0:R_, 1:_W]>} (%ch0, %arg0, %arg1) :
   (!mo.chain, !mo.buffer<[2, 2], f32>, !mo.opaque<"Foo">) -> !mo.chain

  mo.output
}
