// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels | FileCheck %s

// ViewFuser cooperating with ElementwiseFuser across rounds: the add/relu
// chain must be fully collapsed by ElementwiseFuser before ViewFuser can
// fold the combined kernel into the transpose in a later round of the same
// fixed-point loop -- exercises the full map-fuse-kernels pipeline, not a
// single fuser in isolation.

// CHECK-LABEL: mo.graph @elementwiseChainIntoTranspose
// CHECK:       %{{.*}} = map.iter.foreach<[32, 3]>
// CHECK:       index   (%[[IDX:.*]]: !map.index<2>) -> !mo.tensor<[32, 3], f32> {
// CHECK-NEXT:  %[[TIDX:.*]] = map.index.transpose<{{.*}}> (%[[IDX]], %{{.*}})
// CHECK-NEXT:  %[[XV:.*]] = map.load %{{.*}}[%[[TIDX]]]
// CHECK-NEXT:  %[[YV:.*]] = map.load %{{.*}}[%[[TIDX]]]
// CHECK-NEXT:  %[[ADD:.*]] = map.call.elementwise @{{.*}}.Add(%[[TIDX]], %[[XV]], %[[YV]])
// CHECK-NEXT:  map.call.elementwise @{{.*}}.ReLU(%[[TIDX]], %[[ADD]])
// CHECK-NEXT:  map.store %{{.*}}[%[[IDX]]]
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_view<[#mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.relu"{{.*}}]>, #mogg.fusion_leaf<"mo.transpose"{{.*}}]>>
// CHECK-NOT:   map.iter.foreach
mo.graph @elementwiseChainIntoTranspose(%x: !mo.tensor<[3, 32], f32>, %y: !mo.tensor<[3, 32], f32>) -> !mo.tensor<[32, 3], f32> {
  %added = mo.add(%x, %y) : !mo.tensor<[3, 32], f32>
  %relued = mo.relu(%added) : !mo.tensor<[3, 32], f32>
  %perm = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.transpose(%relued, %perm) : (!mo.tensor<[3, 32], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[32, 3], f32>
  mo.output %res : !mo.tensor<[32, 3], f32>
}
