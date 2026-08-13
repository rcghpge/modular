// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels | FileCheck %s

// CHECK-LABEL: mo.graph @matmul_add
// CHECK-SAME:  (%[[X:.*]]: !mo.tensor<[8, 8], f32>, %[[Y:.*]]: !mo.tensor<[8, 8], f32>, %[[Z:.*]]: !mo.tensor<[8, 8], f32>)
// CHECK-NEXT:  %[[OUT:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs (%[[X]]: !mo.tensor<[8, 8], f32>, %[[Y]]: !mo.tensor<[8, 8], f32>, %[[Z]]: !mo.tensor<[8, 8], f32>)
// CHECK-NEXT:  outputs (%{{.*}} : !mo.tensor<[8, 8], f32>) {
// CHECK-NEXT:  %[[EPI:.*]] = map.iter.epilogue (%[[VAL:.*]] : f32, %{{.*}} : !map.index<2>) capturing (%[[CAP0:.*]] = %{{.*}} : !mo.tensor<[8, 8], f32>, %[[CAP1:.*]] = %[[Z]] : !mo.tensor<[8, 8], f32>)
// CHECK-NEXT:  %[[ZV:.*]] = map.load %[[CAP1]][%{{.*}}] : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:  %[[ADD:.*]] = map.call.elementwise @{{.*}}.Add(%{{.*}}, %[[VAL]], %[[ZV]])
// CHECK-NEXT:  map.store %[[CAP0]][%{{.*}}] = %[[ADD]] : !mo.tensor<[8, 8], f32>
// CHECK:       map.call.opaque @{{.*}}.Matmul{{.*}} inputs(%[[X]], %[[Y]]) outputs(%[[EPI]])
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_epilogue<[#mogg.fusion_leaf<"mo.matmul"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}], operandIndices = [0]>>
// CHECK-NOT:   map.iter.foreach
// CHECK:       mo.output %[[OUT]] : !mo.tensor<[8, 8], f32>
mo.graph @matmul_add(
  %x: !mo.tensor<[8, 8], f32>,
  %y: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[8, 8], f32>
) -> !mo.tensor<[8, 8], f32> {
  %mm = mo.matmul(%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %out = mo.add(%mm, %z) : !mo.tensor<[8, 8], f32>
  mo.output %out : !mo.tensor<[8, 8], f32>
}

// CHECK-LABEL: mo.graph @custom_add
// CHECK:      map.iter.opaque
// CHECK:      map.iter.epilogue
// CHECK:      map.call.elementwise @{{.*}}.Add
// CHECK:      map.call.opaque @{{.*}}MapLowerOneFusedOutput
// CHECK:      fusionInfo = #mogg.fusion<#mogg.fusion_epilogue<[#mogg.fusion_leaf<"custom__map_lower_one_fused_output"{{.*}}, #mogg.fusion_leaf<"mo.add"
// CHECK-NOT:  map.iter.foreach
mo.graph @custom_add(%a: !mo.tensor<[4], f32>, %b: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_fused_output"}(%a) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %out = mo.add(%c, %b) : !mo.tensor<[4], f32>
  mo.output %out : !mo.tensor<[4], f32>
}

// CHECK-LABEL: mo.graph @chain
// CHECK:      map.iter.opaque
// CHECK:      map.iter.epilogue
// CHECK:      map.call.elementwise @{{.*}}.Add
// CHECK:      map.call.elementwise @{{.*}}.ReLU
// CHECK:      map.store
// CHECK:      map.call.opaque @{{.*}}.Matmul
// CHECK:      #mogg.fusion_epilogue<[#mogg.fusion_leaf<"mo.matmul"
// CHECK-SAME: #mogg.fusion_leaf<"mo.add"
// CHECK-SAME: #mogg.fusion_leaf<"mo.relu"
// CHECK-NOT:  map.iter.foreach
mo.graph @chain(
  %x: !mo.tensor<[8, 8], f32>,
  %y: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[8, 8], f32>
) -> !mo.tensor<[8, 8], f32> {
  %mm = mo.matmul(%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %sum = mo.add(%mm, %z) : !mo.tensor<[8, 8], f32>
  %out = mo.relu(%sum) : !mo.tensor<[8, 8], f32>
  mo.output %out : !mo.tensor<[8, 8], f32>
}

// CHECK-LABEL: mo.graph @fuse_in_if
// CHECK:      mo.if
// CHECK:      map.iter.opaque
// CHECK:      map.call.elementwise @{{.*}}.Add
// CHECK:      #mogg.fusion_epilogue<[#mogg.fusion_leaf<"custom__map_lower_one_fused_output"{{.*}}, #mogg.fusion_leaf<"mo.add"
// CHECK-NOT:  map.iter.foreach
// CHECK:      } else {
// CHECK:      map.iter.opaque
// CHECK:      map.call.elementwise @{{.*}}.Mul
// CHECK:      #mogg.fusion_epilogue<[#mogg.fusion_leaf<"custom__map_lower_one_fused_output"{{.*}}, #mogg.fusion_leaf<"mo.mul"
// CHECK-NOT:  map.iter.foreach
// CHECK:      mo.output
mo.graph @fuse_in_if(
  %a: !mo.tensor<[4], f32>,
  %b: !mo.tensor<[4], f32>,
  %cond: !mo.tensor<[], bool>
) -> !mo.tensor<[4], f32> {
  %res = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[4], f32> {
    %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_fused_output"}(%a) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    %out = mo.add(%c, %b) : !mo.tensor<[4], f32>
    mo.yield %out : !mo.tensor<[4], f32>
  } else {
    %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_fused_output"}(%a) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    %out = mo.mul(%c, %b) : !mo.tensor<[4], f32>
    mo.yield %out : !mo.tensor<[4], f32>
  }
  mo.output %res : !mo.tensor<[4], f32>
}

// CHECK-LABEL: mo.graph @no_fuse_across_blocks
// CHECK:      map.iter.opaque
// CHECK:      #mogg.fusion<#mogg.fusion_leaf<"custom__map_lower_one_fused_output"
// CHECK:      mo.if
// CHECK:      map.iter.foreach
// CHECK:      map.call.elementwise @{{.*}}.Add
mo.graph @no_fuse_across_blocks(
  %a: !mo.tensor<[4], f32>,
  %b: !mo.tensor<[4], f32>,
  %cond: !mo.tensor<[], bool>
) -> !mo.tensor<[4], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_fused_output"}(%a) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %res = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[4], f32> {
    %out = mo.add(%c, %b) : !mo.tensor<[4], f32>
    mo.yield %out : !mo.tensor<[4], f32>
  } else {
    mo.yield %b : !mo.tensor<[4], f32>
  }
  mo.output %res : !mo.tensor<[4], f32>
}

// CHECK-LABEL: mo.graph @cast_fused
// CHECK:      %[[DST:.*]] = map.iter.opaque
// CHECK:        outputs (%[[OUT:.*]] : !mo.tensor<[4], si32>) {
// CHECK-NEXT:   map.iter.epilogue (%[[VAL:.*]] : f32, %[[IDX:.*]] : !map.index<1>) capturing (%[[CAP:.*]] = %[[OUT]] : !mo.tensor<[4], si32>)
// CHECK-NEXT:   map.call.elementwise @{{.*}}.Cast(%[[IDX]], %[[VAL]]) : !map.index<1>, f32 -> si32
// CHECK-NEXT:   map.store %[[CAP]][%[[IDX]]] = %{{.*}} : !mo.tensor<[4], si32>
// CHECK:      } {fusionInfo = #mogg.fusion<#mogg.fusion_epilogue<[#mogg.fusion_leaf<"custom__map_lower_one_fused_output"
// CHECK-SAME: #mogg.fusion_leaf<"mo.cast"
// CHECK-NOT:  map.iter.foreach
// CHECK:      mo.output %[[DST]] : !mo.tensor<[4], si32>
mo.graph @cast_fused(%a: !mo.tensor<[4], f32>) -> !mo.tensor<[4], si32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_fused_output"}(%a) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %out = mo.cast(%c) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], si32>
  mo.output %out : !mo.tensor<[4], si32>
}

// CHECK-LABEL: mo.graph @no_fuse_matmul_cast
// CHECK:      %[[MM:.*]] = map.iter.opaque
// CHECK:        outputs (%{{.*}} : !mo.tensor<[8, 8], f32>) {
// CHECK:      %[[CAST:.*]] = map.iter.foreach
// CHECK-NEXT: inputs  (%[[MM]]: !mo.tensor<[8, 8], f32>)
// CHECK:      map.call.elementwise @{{.*}}.Cast
// CHECK:      } {fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.cast"
// CHECK:      mo.output %[[CAST]] : !mo.tensor<[8, 8], si32>
mo.graph @no_fuse_matmul_cast(
  %x: !mo.tensor<[8, 8], f32>,
  %y: !mo.tensor<[8, 8], f32>
) -> !mo.tensor<[8, 8], si32> {
  %mm = mo.matmul(%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %out = mo.cast(%mm) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], si32>
  mo.output %out : !mo.tensor<[8, 8], si32>
}

// CHECK-LABEL: mo.graph @no_fuse_multi_use
// CHECK:      map.iter.opaque
// CHECK:      map.iter.foreach
// CHECK:      map.iter.foreach
mo.graph @no_fuse_multi_use(%a: !mo.tensor<[4], f32>, %b: !mo.tensor<[4], f32>)
    -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_fused_output"}(%a) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %r0 = mo.add(%c, %b) : !mo.tensor<[4], f32>
  %r1 = mo.mul(%c, %b) : !mo.tensor<[4], f32>
  mo.output %r0, %r1 : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>
}

// CHECK-LABEL: mo.graph @no_fuse_plain_output
// CHECK:      map.iter.opaque
// CHECK-NOT:  map.iter.epilogue
// CHECK:      map.call.opaque @{{.*}}MapLowerOnePlainOutput
// CHECK:      map.iter.foreach
// CHECK:      map.call.elementwise @{{.*}}.Add
mo.graph @no_fuse_plain_output(%a: !mo.tensor<[4], f32>, %b: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_plain_output"}(%a) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %out = mo.add(%c, %b) : !mo.tensor<[4], f32>
  mo.output %out : !mo.tensor<[4], f32>
}

// CHECK-LABEL: mo.graph @no_fuse_same_result_twice
// CHECK:      map.iter.opaque
// CHECK:      map.iter.foreach
// CHECK:      map.call.elementwise @{{.*}}.Add
mo.graph @no_fuse_same_result_twice(%a: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_fused_output"}(%a) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %out = mo.add(%c, %c) : !mo.tensor<[4], f32>
  mo.output %out : !mo.tensor<[4], f32>
}

// CHECK-LABEL: mo.graph @two_producers_one_fuses
// CHECK-DAG:  map.call.elementwise @{{.*}}.Add
// CHECK-DAG:  #mogg.fusion_epilogue<[#mogg.fusion_leaf<"custom__map_lower_one_fused_output"{{.*}}, #mogg.fusion_leaf<"mo.add"
// CHECK-NOT:  map.iter.foreach
// CHECK:      mo.output
mo.graph @two_producers_one_fuses(%a: !mo.tensor<[4], f32>, %b: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %c0 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_fused_output"}(%a) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %c1 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_fused_output"}(%b) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %out = mo.add(%c0, %c1) : !mo.tensor<[4], f32>
  mo.output %out : !mo.tensor<[4], f32>
}

// CHECK-LABEL: mo.graph @cycle_fuses_other_producer
// CHECK:      %[[DST:.*]]:2 = map.iter.opaque
// CHECK-NOT:  map.call.elementwise
// CHECK:      map.call.opaque @{{.*}}MapLowerTwoFusedOutputs{{.*}} inputs(%{{.*}}) outputs(%{{.*}}, %{{.*}})
// CHECK:      } {fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"custom__map_lower_two_fused_outputs"
// CHECK:      %{{.*}} = map.iter.opaque
// CHECK-NEXT: inputs (%[[DST]]#1: !mo.tensor<[4], f32>, %[[DST]]#0: !mo.tensor<[4], f32>)
// CHECK:      capturing (%[[CAP0:.*]] = %{{.*}} : !mo.tensor<[4], f32>, %[[CAP1:.*]] = %[[DST]]#0 : !mo.tensor<[4], f32>)
// CHECK:      map.call.elementwise @builtin_kernels.elementwise.Add
// CHECK:      map.call.opaque @{{.*}}MapLowerOneFusedOutput{{.*}}
// CHECK:      } {fusionInfo = #mogg.fusion<#mogg.fusion_epilogue<[#mogg.fusion_leaf<"custom__map_lower_one_fused_output"{{.*}}, #mogg.fusion_leaf<"mo.add"
mo.graph @cycle_fuses_other_producer(%a: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %r0, %r1 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_two_fused_outputs"}(%a) : (!mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>)
  %bar = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_fused_output"}(%r1) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %out = mo.add(%bar, %r0) : !mo.tensor<[4], f32>
  mo.output %out : !mo.tensor<[4], f32>
}
