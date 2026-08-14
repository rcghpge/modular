// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --map-to-mogg | FileCheck %s

// CHECK-LABEL: mo.graph @partial_fused
// CHECK: %[[A:.*]] = mogg._tensor.create(%{{.*}}, %{{.*}}) {device = #M.device_ref<"cpu", 0>,{{.*}}mut = false} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK: %[[B:.*]] = mogg._tensor.create(%{{.*}}, %{{.*}}) {device = #M.device_ref<"cpu", 0>,{{.*}}mut = false} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK: %[[PRO:.*]] = mogg._tensor.create.prologue %{{.*}} capturing(%[[A]] : !mogg.tensor_) : !mogg.shape -> !mogg.tensor_ attributes {device = #M.device_ref<"cpu", 0>, elementType = f32, io = 1 : i32, mut = false} {
// CHECK: ^bb0(%[[I:.*]]: !map.index<2>, %[[CAP:.*]]: !mogg.tensor_):
// CHECK:   %[[V:.*]] = mogg._tensor.load %[[CAP]][%[[I]]] : !mogg.tensor_, !map.index<2> -> f32
// CHECK:   mogg.yield %[[V]] : f32
// CHECK: mogg._call.opaque @{{.*}}MapLowerPartialFusedInputs inputs(%[[PRO]], %[[B]] : !mogg.tensor_, !mogg.tensor_) outputs(%{{.*}} : !mogg.tensor_) context(%{{.*}} : !mogg.context)
mo.graph @partial_fused(%a: !mo.tensor<[2, 2], f32>, %b: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_partial_fused_inputs"}(%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: mo.graph @two_fused
// CHECK: %[[P0:.*]] = mogg._tensor.create.prologue %{{.*}} capturing(%{{.*}} : !mogg.tensor_) : !mogg.shape -> !mogg.tensor_
// CHECK: %[[P1:.*]] = mogg._tensor.create.prologue %{{.*}} capturing(%{{.*}} : !mogg.tensor_) : !mogg.shape -> !mogg.tensor_
// CHECK: mogg._call.opaque @{{.*}}MapLowerTwoFusedInputs inputs(%[[P0]], %[[P1]] : !mogg.tensor_, !mogg.tensor_) outputs(%{{.*}} : !mogg.tensor_) context(%{{.*}} : !mogg.context)
mo.graph @two_fused(%a: !mo.tensor<[2, 2], f32>, %b: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_two_fused_inputs"}(%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: mo.graph @fused_producer
// CHECK: fusionInfo = #mogg.fusion<#mogg.fusion_prologue<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"custom__map_lower_partial_fused_inputs"{{.*}}], operandIndices = [0]>>
// CHECK: %[[PRO:.*]] = mogg._tensor.create.prologue %{{.*}} capturing(%[[C0:.*]], %[[C1:.*]] : !mogg.tensor_, !mogg.tensor_) : !mogg.shape -> !mogg.tensor_
// CHECK: ^bb0(%[[I:.*]]: !map.index<2>, %[[A0:.*]]: !mogg.tensor_, %[[A1:.*]]: !mogg.tensor_):
// CHECK:   %[[L0:.*]] = mogg._tensor.load %[[A0]][%[[I]]] : !mogg.tensor_, !map.index<2> -> f32
// CHECK:   %[[L1:.*]] = mogg._tensor.load %[[A1]][%[[I]]] : !mogg.tensor_, !map.index<2> -> f32
// CHECK:   %[[ADD:.*]] = mogg._call.elementwise["{{.*}}Add"] (%[[L0]], %[[L1]]) {moOpName = "mo.add"} : f32, f32 -> f32
// CHECK:   mogg.yield %[[ADD]] : f32
// CHECK: mogg._call.opaque @{{.*}}MapLowerPartialFusedInputs inputs(%[[PRO]], %{{.*}} : !mogg.tensor_, !mogg.tensor_)
// CHECK-NOT: map.iter.prologue
mo.graph @fused_producer(%a: !mo.tensor<[2, 2], f32>, %b: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %p = mo.add(%a, %b) : !mo.tensor<[2, 2], f32>
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_partial_fused_inputs"}(%p, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: mo.graph @two_fused_producers
// CHECK: fusionInfo = #mogg.fusion<#mogg.fusion_prologue<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.mul"{{.*}}, #mogg.fusion_leaf<"custom__map_lower_two_fused_inputs"{{.*}}], operandIndices = [0, 1]>>
// CHECK: mogg._call.elementwise["{{.*}}Add"]
// CHECK: mogg._call.elementwise["{{.*}}Mul"]
// CHECK: mogg._call.opaque @{{.*}}MapLowerTwoFusedInputs
mo.graph @two_fused_producers(%a: !mo.tensor<[2, 2], f32>, %b: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %p0 = mo.add(%a, %b) : !mo.tensor<[2, 2], f32>
  %p1 = mo.mul(%a, %b) : !mo.tensor<[2, 2], f32>
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_two_fused_inputs"}(%p0, %p1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}
