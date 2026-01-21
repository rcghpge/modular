// RUN: mo-opt %s --mo-to-mgp="constant-fold=false" --mlir-print-ir-after=mo-to-mogg 2>&1 | FileCheck %s

// Test shape function lowering to MOGG dialect with experimental Mojo path

// CHECK-LABEL: mo.graph @avg_pool_mixed_dims
// CHECK: mogg.experimental.kernel
// CHECK-SAME: (%[[INPUT:.*]]: !mo.tensor<[32, 10, 20, 3], f32>, %[[FILTER:.*]]: !mo.tensor<[2], si64>, %[[STRIDES:.*]]: !mo.tensor<[2], si64>, %[[DILATIONS:.*]]: !mo.tensor<[2], si64>, %[[PADDINGS:.*]]: !mo.tensor<[4], si64>) -> ()
// CHECK:   mogg.call.shape["mo.avg_pool"]
// CHECK-SAME: (%[[INPUT]], %[[FILTER]], %[[STRIDES]], %[[DILATIONS]], %[[PADDINGS]]
// CHECK-SAME: : !mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mogg.shape_func_result<4>
// CHECK-SAME: {isThrows, kernel_param = {output_rank = 4 : index}}
// CHECK: } {allocIndices = [], device = #M.device_ref<"cpu", 0>, hasReadEffect = false, hasWriteEffect = false, isAsync = true, outputParamDecls = #kgen<param.decls[ F1 : si64, D4 : si64, D5 : si64, F2 : si64]>, summary = "shapeof(mo.avg_pool){{.*}}"}
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
// CHECK: mogg.experimental.kernel
// CHECK-SAME: (%[[G_INPUT:.*]]: !mo.tensor<[10, D0], f32>, %[[G_INDICES:.*]]: !mo.tensor<[D1, 20], si32>, %[[G_AXIS:.*]]: !mo.tensor<[], si32>) -> ()
// CHECK:   mogg.as_scalar
// CHECK:   mogg.call.shape["mo.gather"]
// CHECK-SAME: (%[[G_INPUT]], %[[G_INDICES]]
// CHECK-SAME: : !mo.tensor<[10, D0], f32>, !mo.tensor<[D1, 20], si32>, !mo.scalar<si32>) -> !mogg.shape_func_result<3>
// CHECK-SAME: {isThrows, kernel_param = {output_rank = 3 : index}}
// CHECK: } {allocIndices = [], device = #M.device_ref<"cpu", 0>, hasReadEffect = false, hasWriteEffect = false, isAsync = true, outputParamDecls = #kgen<param.decls[ D5 : si64, D6 : si64, D7 : si64]>, summary = "shapeof(mo.gather){{.*}}"}
mo.graph @gather_fallback_shape_func<D0, D1 -> D2, D3, D4>(
  %input : !mo.tensor<[10, D0], f32>,
  %indices : !mo.tensor<[D1, 20], si32>,
  %axis: !mo.tensor<[], si32>
) -> !mo.tensor<[D2, D3, D4], f32> {

  %res = mo.gather<() -> D5, D6, D7>(%input, %indices, %axis) :
    (!mo.tensor<[10, D0], f32>, !mo.tensor<[D1, 20], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[D5, D6, D7], f32>

  mo.output<D5, D6, D7> %res : !mo.tensor<[D5, D6, D7], f32>
}
