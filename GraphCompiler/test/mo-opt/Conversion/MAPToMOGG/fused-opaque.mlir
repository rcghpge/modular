// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-to-mogg | FileCheck %s

// MAPToMOGG lowers a store-epilogue (fused) output of a `map.iter.opaque`
// directly into a `mogg._tensor.create.epilogue` holding the
// `mogg._tensor.store` in its own body (capturing the raw output tensor); the
// kernel call writes into the epilogue tensor rather than the raw output
// tensor. The epilogue tensor reuses the store target's `!mogg.shape` operand
// and records the kernel `device`, and the call is wired to the body's
// `!mogg.context`.

// One fused output.
// CHECK-LABEL: mo.graph @one_fused
// CHECK: mogg._kernel (%{{.*}} : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
// CHECK: context (%[[CTX:.*]]: !mogg.context) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK: %[[OT:.*]] = mogg._tensor.create(%{{.*}}, %{{.*}}) {device = #M.device_ref<"cpu", 0>,{{.*}}mut = true} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK: %[[ESH:.*]] = mogg._shape.create()
// CHECK: %[[EPI:.*]] = mogg._tensor.create.epilogue %[[ESH]] capturing(%[[OT]] : !mogg.tensor_) : !mogg.shape -> !mogg.tensor_ attributes {device = #M.device_ref<"cpu", 0>,{{.*}}mut = true} {
// CHECK: ^bb0(%[[V:.*]]: f32, %[[I:.*]]: !map.index<2>, %[[CAP:.*]]: !mogg.tensor_):
// CHECK:   mogg._tensor.store %[[CAP]][%[[I]]] = %[[V]] : !mogg.tensor_, !map.index<2>, f32
// CHECK: mogg._call.opaque @{{.*}}MapLowerOneFusedOutput inputs(%{{.*}} : !mogg.tensor_) outputs(%[[EPI]] : !mogg.tensor_) context(%[[CTX]] : !mogg.context)
mo.graph @one_fused(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_fused_output"}(%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Two fused outputs: two epilogue tensors, both written by the call.
// CHECK-LABEL: mo.graph @two_fused
// CHECK: %[[OT0:.*]] = mogg._tensor.create(%{{.*}}, %{{.*}}) {device = #M.device_ref<"cpu", 0>,{{.*}}mut = true} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK: %[[OT1:.*]] = mogg._tensor.create(%{{.*}}, %{{.*}}) {device = #M.device_ref<"cpu", 0>,{{.*}}mut = true} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK: %[[ESH0:.*]] = mogg._shape.create()
// CHECK: %[[E0:.*]] = mogg._tensor.create.epilogue %[[ESH0]] capturing(%[[OT0]] : !mogg.tensor_) : !mogg.shape -> !mogg.tensor_ attributes {device = #M.device_ref<"cpu", 0>,{{.*}}mut = true} {
// CHECK: %[[ESH1:.*]] = mogg._shape.create()
// CHECK: %[[E1:.*]] = mogg._tensor.create.epilogue %[[ESH1]] capturing(%[[OT1]] : !mogg.tensor_) : !mogg.shape -> !mogg.tensor_ attributes {device = #M.device_ref<"cpu", 0>,{{.*}}mut = true} {
// CHECK: mogg._call.opaque @{{.*}}MapLowerTwoFusedOutputs inputs(%{{.*}} : !mogg.tensor_) outputs(%[[E0]], %[[E1]] : !mogg.tensor_, !mogg.tensor_) context(%{{.*}} : !mogg.context)
mo.graph @two_fused(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %c:2 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_two_fused_outputs"}(%a) : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %c#0, %c#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// Two inputs feeding one fused output: the epilogue's captured output block
// arg number is offset by the two leading input block args, so this exercises
// that the capture still resolves to the right (only) output tensor.
// CHECK-LABEL: mo.graph @two_inputs_one_fused
// CHECK: mogg._kernel (%{{.*}}, %{{.*}} : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
// CHECK: %[[OT:.*]] = mogg._tensor.create(%{{.*}}, %{{.*}}) {device = #M.device_ref<"cpu", 0>,{{.*}}mut = true} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK: %[[ESH:.*]] = mogg._shape.create()
// CHECK: %[[EPI:.*]] = mogg._tensor.create.epilogue %[[ESH]] capturing(%[[OT]] : !mogg.tensor_) : !mogg.shape -> !mogg.tensor_ attributes {device = #M.device_ref<"cpu", 0>,{{.*}}mut = true} {
// CHECK: mogg._call.opaque @{{.*}}MapLowerTwoInputsOneFusedOutput inputs(%{{.*}}, %{{.*}} : !mogg.tensor_, !mogg.tensor_) outputs(%[[EPI]] : !mogg.tensor_) context(%{{.*}} : !mogg.context)
mo.graph @two_inputs_one_fused(%a: !mo.tensor<[2, 2], f32>, %b: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_two_inputs_one_fused_output"}(%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Mixed: first output fused (epilogue tensor), second plain (raw output tensor).
// CHECK-LABEL: mo.graph @two_partial
// CHECK: %[[EPI:.*]] = mogg._tensor.create.epilogue %{{.*}} capturing(%{{.*}} : !mogg.tensor_) : !mogg.shape -> !mogg.tensor_ attributes {device = #M.device_ref<"cpu", 0>,{{.*}}mut = true} {
// CHECK: mogg._call.opaque @{{.*}}MapLowerPartialFusedOutputs inputs(%{{.*}} : !mogg.tensor_) outputs(%[[EPI]], %{{.*}} : !mogg.tensor_, !mogg.tensor_) context(%{{.*}} : !mogg.context)
mo.graph @two_partial(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %c:2 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_partial_fused_outputs"}(%a) : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %c#0, %c#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}
