// RUN: mo-opt %s --gc-pipeline="constant-fold=false" --mlir-print-ir-after=mo-to-mogg 2>&1 | FileCheck %s

// Test shape function lowering to MOGG dialect with experimental Mojo path

// CHECK-LABEL: mo.graph @avg_pool_mixed_dims
// CHECK: mogg.kernel
// CHECK-SAME: (%[[INPUT:.*]]: !mo.tensor<[32, 10, 20, 3], f32>, %[[FILTER:.*]]: !mo.tensor<[2], si64>, %[[STRIDES:.*]]: !mo.tensor<[2], si64>, %[[DILATIONS:.*]]: !mo.tensor<[2], si64>, %[[PADDINGS:.*]]: !mo.tensor<[4], si64>) -> ()
// CHECK:   mogg.call.shape["{{.*}}.AvgPool"]
// CHECK-SAME: (%[[INPUT]], %[[FILTER]], %[[STRIDES]], %[[DILATIONS]], %[[PADDINGS]]
// CHECK-SAME: : !mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mogg.shape_func_result<4>
// CHECK-SAME: {isThrows, kernel_param = {output_rank = 4 : index}}
// CHECK: } fusionInfo = "shapeof(mo.avg_pool)":{{[0-9]+}} {allocIndices = [], device = #M.device_ref<"cpu", 0>, hasReadEffect = false, hasWriteEffect = false, isAsync = true, outputParamDecls = #kgen<param.decls[F1 : !kgen.scalar<si64>, D4 : !kgen.scalar<si64>, D5 : !kgen.scalar<si64>, F2 : !kgen.scalar<si64>]>}
mo.graph @avg_pool_mixed_dims<() -> D1, D2>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %filter_shape: !mo.tensor<[2], si64>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>,
  %paddings: !mo.tensor<[4], si64>
) -> !mo.tensor<[32, D1, D2, 3], f32> {

  %res = mo.avg_pool<() -> D4, D5>(%input) [filter_shape = %filter_shape, strides = %strides, dilations = %dilations, paddings = %paddings] {count_boundary = false} :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
    -> !mo.tensor<[32, D4, D5, 3], f32>

  mo.output<D4, D5> %res : !mo.tensor<[32, D4, D5, 3], f32>
}

// CHECK-LABEL: mo.graph @gather_fallback_shape_func
// CHECK: mogg.kernel
// CHECK-SAME: (%[[G_INPUT:.*]]: !mo.tensor<[10, D0], f32>, %[[G_INDICES:.*]]: !mo.tensor<[D1, 20], si32>) -> !mo.tensor<[D1, 20, D0], f32>
// CHECK-NOT: mogg.as_scalar
// CHECK:   mogg.call.execute["{{.*}}.Gather"]
// CHECK-SAME: inputs(%{{.*}}, %{{.*}}, %{{.*}} : !mo.tensor<[10, D0], f32>, !mo.tensor<[D1, 20], si32>, !mogg.context)
// CHECK-SAME: outputs(%{{.*}} : !mo.tensor<[D1, 20, D0], f32>)
// CHECK-SAME: {isThrows, kernel_param = {axis = 0 : index}, moOpName = "mo.gather"}
// CHECK: } fusionInfo = "mo.gather":{{[0-9]+}} {allocIndices = [], device = #M.device_ref<"cpu", 0>, hasReadEffect = false, hasWriteEffect = false, isAsync = true, outputParamDecls = #kgen<param.decls[]>}
mo.graph @gather_fallback_shape_func<D0, D1 -> D2, D3, D4>(
  %input : !mo.tensor<[10, D0], f32>,
  %indices : !mo.tensor<[D1, 20], si32>
) -> !mo.tensor<[D2, D3, D4], f32> {

  %res = mo.gather<() -> D5, D6, D7>(%input, %indices) {axis = 0 : index} :
    (!mo.tensor<[10, D0], f32>, !mo.tensor<[D1, 20], si32>) -> !mo.tensor<[D5, D6, D7], f32>

  mo.output<D5, D6, D7> %res : !mo.tensor<[D5, D6, D7], f32>
}
