// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels | FileCheck %s


mo.graph @add_fuses_into_mutable_store(
  %buf: !mo.buffer<[8], f32>,
  %a: !mo.tensor<[8], f32>,
  %b: !mo.tensor<[8], f32>
) {
  %chain = mo.chain.create()
  %res = mo.add(%a, %b) : !mo.tensor<[8], f32>
  %out = mo.mutable.store(%buf, %res, %chain) : (!mo.buffer<[8], f32>, !mo.tensor<[8], f32>, !mo.chain) -> !mo.chain
  mo.output
}

// CHECK-LABEL: mo.graph @add_fuses_into_mutable_store
// CHECK-SAME:  (%[[BUF:.*]]: !mo.buffer<[8], f32>, %[[A:.*]]: !mo.tensor<[8], f32>, %[[B:.*]]: !mo.tensor<[8], f32>)
// CHECK:       map.iter.foreach<[8]>
// CHECK-NEXT:  inputs  (%[[A]]: !mo.tensor<[8], f32>, %[[B]]: !mo.tensor<[8], f32>, %[[BUF]]: !mo.buffer<[8], f32>)
// CHECK-NEXT:  outputs ()
// CHECK:       %[[AV:.*]] = map.load %[[A]][%{{.*}}] : !mo.tensor<[8], f32>
// CHECK-NEXT:  %[[BV:.*]] = map.load %[[B]][%{{.*}}] : !mo.tensor<[8], f32>
// CHECK-NEXT:  %[[ADDV:.*]] = map.call.elementwise @{{.*}}.Add(%{{.*}}, %[[AV]], %[[BV]])
// CHECK-NEXT:  map.call.elementwise @{{.*}}MutableStore(%{{.*}}, %[[ADDV]])
// CHECK-NEXT:  map.store %[[BUF]]
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.mutable.store"{{.*}}]>>
// The fused kernel must keep mo.mutable.store's write effect (dst's effects
// survive the merge in ElementwiseFuser::fuse, mirroring MOGG).
// CHECK-SAME:  hasReadEffect = false, hasWriteEffect = true
// CHECK-NOT:   map.iter.foreach


// %chain has two uses (no mo.chain.create merge), so fusing into @out1
// would be unsafe relative to sibling @out2 (mirrors checkMutableKernelFusible).
mo.graph @shared_chain_blocks_fusion(
  %buf1: !mo.buffer<[8], f32>,
  %buf2: !mo.buffer<[8], f32>,
  %a: !mo.tensor<[8], f32>,
  %b: !mo.tensor<[8], f32>
) {
  %chain = mo.chain.create()
  %res1 = mo.add(%a, %b) : !mo.tensor<[8], f32>
  %out1 = mo.mutable.store(%buf1, %res1, %chain) : (!mo.buffer<[8], f32>, !mo.tensor<[8], f32>, !mo.chain) -> !mo.chain
  %out2 = mo.mutable.store(%buf2, %b, %chain) : (!mo.buffer<[8], f32>, !mo.tensor<[8], f32>, !mo.chain) -> !mo.chain
  mo.output
}

// CHECK-LABEL: mo.graph @shared_chain_blocks_fusion
// CHECK:       %[[ADD2:.*]] = map.iter.foreach<[8]>
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK-NOT:   map.call.elementwise @{{.*}}.Add
// CHECK:       map.iter.foreach<[8]>
// CHECK-NEXT:  inputs  (%[[ADD2]]: !mo.tensor<[8], f32>,
// CHECK:       map.call.elementwise @{{.*}}MutableStore
