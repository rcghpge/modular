// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-prologue --map-to-mogg | FileCheck %s

// A fusible input gets a load prologue whether or not a producer fuses into
// it, so the first two graphs below carry no `hasPrologueFusion`: their bodies
// yield a bare load. Only @fused_producer and @two_fused_producers do.

// CHECK-LABEL: mo.graph @partial_fused
// CHECK: %[[A:.*]] = mogg.tensor.create(%{{.*}}, %{{.*}}) <device = "cpu",{{.*}}mut = false{{.*}}> : (!mogg.pointer, !mogg.shape) -> !mogg.tensor
// CHECK: %[[B:.*]] = mogg.tensor.create(%{{.*}}, %{{.*}}) <device = "cpu",{{.*}}mut = false{{.*}}> : (!mogg.pointer, !mogg.shape) -> !mogg.tensor
// CHECK: %[[PRO:.*]] = mogg.tensor.create.prologue %{{.*}} capturing(%[[A]] : !mogg.tensor) : !mogg.shape -> !mogg.tensor device = "cpu", elementType = f32, io = Input, mut = false {
// CHECK: ^bb0(%[[I:.*]]: !map.index<2>, %[[CAP:.*]]: !mogg.tensor):
// CHECK:   %[[V:.*]] = mogg.tensor.load %[[CAP]][%[[I]]] : !mogg.tensor, !map.index<2> -> f32
// CHECK:   mogg.yield %[[V]] : f32
// CHECK: mogg.call.opaque @{{.*}}MapLowerPartialFusedInputs inputs(%[[PRO]], %[[B]] : !mogg.tensor, !mogg.tensor) outputs(%{{.*}} : !mogg.tensor) context(%{{.*}} : !mogg.context)
// CHECK-NOT: hasPrologueFusion
mo.graph @partial_fused(%a: !mo.tensor<[2, 2], f32>, %b: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_partial_fused_inputs"}(%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: mo.graph @two_fused
// CHECK: %[[P0:.*]] = mogg.tensor.create.prologue %{{.*}} capturing(%{{.*}} : !mogg.tensor) : !mogg.shape -> !mogg.tensor
// CHECK: %[[P1:.*]] = mogg.tensor.create.prologue %{{.*}} capturing(%{{.*}} : !mogg.tensor) : !mogg.shape -> !mogg.tensor
// CHECK: mogg.call.opaque @{{.*}}MapLowerTwoFusedInputs inputs(%[[P0]], %[[P1]] : !mogg.tensor, !mogg.tensor) outputs(%{{.*}} : !mogg.tensor) context(%{{.*}} : !mogg.context)
// CHECK-NOT: hasPrologueFusion
mo.graph @two_fused(%a: !mo.tensor<[2, 2], f32>, %b: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_two_fused_inputs"}(%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: mo.graph @fused_producer
// CHECK: fusionInfo = #mogg.fusion<#mogg.fusion_prologue<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"custom__map_lower_partial_fused_inputs"{{.*}}], operandIndices = [0]>>
// CHECK: %[[PRO:.*]] = mogg.tensor.create.prologue %{{.*}} capturing(%[[C0:.*]], %[[C1:.*]] : !mogg.tensor, !mogg.tensor) : !mogg.shape -> !mogg.tensor
// CHECK: ^bb0(%[[I:.*]]: !map.index<2>, %[[A0:.*]]: !mogg.tensor, %[[A1:.*]]: !mogg.tensor):
// CHECK:   %[[L0:.*]] = mogg.tensor.load %[[A0]][%[[I]]] : !mogg.tensor, !map.index<2> -> f32
// CHECK:   %[[L1:.*]] = mogg.tensor.load %[[A1]][%[[I]]] : !mogg.tensor, !map.index<2> -> f32
// CHECK:   %[[ADD:.*]] = mogg.call.elementwise @{{.*}}Add(%[[L0]], %[[L1]]) moOpName = "mo.add" : f32, f32 -> f32
// CHECK:   mogg.yield %[[ADD]] : f32
// CHECK: mogg.call.opaque @{{.*}}MapLowerPartialFusedInputs inputs(%[[PRO]], %{{.*}} : !mogg.tensor, !mogg.tensor) {{.*}}hasPrologueFusion = true
// CHECK-NOT: map.iter.prologue
mo.graph @fused_producer(%a: !mo.tensor<[2, 2], f32>, %b: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %p = mo.add(%a, %b) : !mo.tensor<[2, 2], f32>
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_partial_fused_inputs"}(%p, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: mo.graph @two_fused_producers
// CHECK: fusionInfo = #mogg.fusion<#mogg.fusion_prologue<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.mul"{{.*}}, #mogg.fusion_leaf<"custom__map_lower_two_fused_inputs"{{.*}}], operandIndices = [0, 1]>>
// CHECK: mogg.call.elementwise @{{.*}}Add
// CHECK: mogg.call.elementwise @{{.*}}Mul
// CHECK: mogg.call.opaque @{{.*}}MapLowerTwoFusedInputs
mo.graph @two_fused_producers(%a: !mo.tensor<[2, 2], f32>, %b: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %p0 = mo.add(%a, %b) : !mo.tensor<[2, 2], f32>
  %p1 = mo.mul(%a, %b) : !mo.tensor<[2, 2], f32>
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_two_fused_inputs"}(%p0, %p1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}
