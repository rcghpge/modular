// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --canonicalize --cse --map-outline-structured-kernel-bodies --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @matmul_diamond
// CHECK-SAME:  (%[[A:[^:]*]]: !mo.tensor<[8, 8], f32>, %[[B:[^:]*]]: !mo.tensor<[8, 8], f32>, %[[Z:[^:]*]]: !mo.tensor<[8, 8], f32>, %[[W:[^:]*]]: !mo.tensor<[8, 8], f32>)

// The matmul opaque is unchanged: %[[D]] is live only through the epilogue
// capture, and the two captures bind distinct values.
// CHECK-NEXT:  %[[RES:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs (%[[A]]: !mo.tensor<[8, 8], f32>, %[[B]]: !mo.tensor<[8, 8], f32>, %[[D:[^:]*]]: !mo.tensor<[8, 8], f32>)
// CHECK-NEXT:  outputs (%[[O:[^ ]*]] : !mo.tensor<[8, 8], f32>) {
// CHECK-NEXT:    %[[EPI:.*]] = map.iter.epilogue (%[[V:[^ ]*]] : f32, %[[I:[^ ]*]] : !map.index<2>) capturing (%[[CO:[^ ]*]] = %[[O]] : !mo.tensor<[8, 8], f32>, %[[CD:[^ ]*]] = %[[D]] : !mo.tensor<[8, 8], f32>) {
// CHECK-NEXT:      %[[DV:.*]] = map.load %[[CD]][%[[I]]] : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:      %[[SUM:.*]] = map.call.elementwise @{{.*}}.Add(%[[I]], %[[V]], %[[DV]])
// CHECK-NEXT:      map.store %[[CO]][%[[I]]] = %[[SUM]] : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:    } : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:    map.call.opaque @{{.*}}.Matmul[packed_b = false, transpose_b = false] inputs(%[[A]], %[[B]]) outputs(%[[EPI]])
// CHECK-NEXT:  } {fusionInfo = #mogg.fusion<#mogg.fusion_epilogue<[#mogg.fusion_leaf<"mo.matmul"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}], operandIndices = [0]>>, hasReadEffect = false, hasWriteEffect = false}

// The diamond opaque: 6 inputs down to 2, every load reading one of them.
// CHECK-NEXT:  %[[D]] = map.iter.opaque
// CHECK-NEXT:  inputs (%[[Z]]: !mo.tensor<[8, 8], f32>, %[[W]]: !mo.tensor<[8, 8], f32>)
// CHECK-NEXT:  outputs (%[[DO:[^ ]*]] : !mo.tensor<[8, 8], f32>) {
// CHECK-NEXT:    map.iter.foreach.body<[8, 8]>
// CHECK-NEXT:      index (%[[J:[^:]*]]: !map.index<2>) {
// CHECK-NEXT:      %[[Z0:.*]] = map.load %[[Z]][%[[J]]] : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:      %[[W0:.*]] = map.load %[[W]][%[[J]]] : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:      %[[ADD0:.*]] = map.call.elementwise @{{.*}}.Add(%[[J]], %[[Z0]], %[[W0]])
// CHECK-NEXT:      %[[Z1:.*]] = map.load %[[Z]][%[[J]]] : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:      %[[W1:.*]] = map.load %[[W]][%[[J]]] : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:      %[[ADD1:.*]] = map.call.elementwise @{{.*}}.Add(%[[J]], %[[Z1]], %[[W1]])
// CHECK-NEXT:      %[[Z2:.*]] = map.load %[[Z]][%[[J]]] : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:      %[[W2:.*]] = map.load %[[W]][%[[J]]] : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:      %[[ADD2:.*]] = map.call.elementwise @{{.*}}.Add(%[[J]], %[[Z2]], %[[W2]])
// CHECK-NEXT:      %[[RELU:.*]] = map.call.elementwise @{{.*}}.ReLU(%[[J]], %[[ADD2]])
// CHECK-NEXT:      %[[MUL:.*]] = map.call.elementwise @{{.*}}.Mul(%[[J]], %[[ADD0]], %[[ADD1]])
// CHECK-NEXT:      %[[DSUM:.*]] = map.call.elementwise @{{.*}}.Add(%[[J]], %[[MUL]], %[[RELU]])
// CHECK-NEXT:      map.store %[[DO]][%[[J]]] = %[[DSUM]] : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:    }
// CHECK-NEXT:  } {fusionInfo = #mogg.fusion<#mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.relu"{{.*}}, #mogg.fusion_leaf<"mo.mul"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>>, hasReadEffect = false, hasWriteEffect = false}
// CHECK-NEXT:  mo.output %[[RES]] : !mo.tensor<[8, 8], f32>
mo.graph @matmul_diamond(
  %a: !mo.tensor<[8, 8], f32>,
  %b: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[8, 8], f32>,
  %w: !mo.tensor<[8, 8], f32>
) -> !mo.tensor<[8, 8], f32> {
  %mm = mo.matmul(%a, %b) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %t0 = mo.add(%z, %w) : !mo.tensor<[8, 8], f32>
  %t1 = mo.mul(%t0, %t0) : !mo.tensor<[8, 8], f32>
  %t2 = mo.relu(%t0) : !mo.tensor<[8, 8], f32>
  %d = mo.add(%t1, %t2) : !mo.tensor<[8, 8], f32>
  %out = mo.add(%mm, %d) : !mo.tensor<[8, 8], f32>
  mo.output %out : !mo.tensor<[8, 8], f32>
}
