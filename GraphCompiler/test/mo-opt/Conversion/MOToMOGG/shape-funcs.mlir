// RUN: mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg" --mlir-print-ir-after=mo-to-mogg 2>&1 | FileCheck %s
// RUN: mo-opt %s --gc-pipeline-pre-jit="constant-fold=false kernel-library-paths=%kernel_api_test_pkg dump-mojo-text" -o /dev/null 2>&1 | FileCheck %s --check-prefix=EMIT

// Test shape function lowering to MOGG dialect with experimental Mojo path

// CHECK-LABEL: mo.graph @avg_pool_mixed_dims
// CHECK: mogg._kernel
// CHECK-SAME: : !mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mogg.shape_func_result<4>
// CHECK: fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"shapeof(mo.avg_pool)"
// CHECK:   mogg._call.shape @{{.*}}.avg_pool_shape
// CHECK-SAME: : !mogg.tensor_, !mogg.tensor_, !mogg.tensor_, !mogg.tensor_, !mogg.tensor_) -> <4>
// CHECK-SAME: {isThrows, kernel_param = {output_rank = 4 : index}}
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
// CHECK: mogg._kernel
// CHECK-SAME: : !mo.tensor<[10, D0], f32>, !mo.tensor<[D1, 20], si32>) -> !mo.tensor<[D1, 20, D0], f32>
// CHECK-NOT: mogg.as_scalar
// CHECK: fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.gather"
// CHECK:   mogg._call.opaque @{{.*}}.Gather
// CHECK-SAME: inputs(%{{.*}}, %{{.*}} : !mogg.tensor_, !mogg.tensor_)
// CHECK-SAME: outputs(%{{.*}} : !mogg.tensor_)
// CHECK-SAME: {kernel_param = {axis = 0 : index}}
mo.graph @gather_fallback_shape_func<D0, D1 -> D2, D3, D4>(
  %input : !mo.tensor<[10, D0], f32>,
  %indices : !mo.tensor<[D1, 20], si32>
) -> !mo.tensor<[D2, D3, D4], f32> {

  %res = mo.gather<() -> D5, D6, D7>(%input, %indices) {axis = 0 : index} :
    (!mo.tensor<[10, D0], f32>, !mo.tensor<[D1, 20], si32>) -> !mo.tensor<[D5, D6, D7], f32>

  mo.output<D5, D6, D7> %res : !mo.tensor<[D5, D6, D7], f32>
}

// Shape function from a standalone @register_shape_function on a custom op.
// CHECK-LABEL: mo.graph @custom_op_shape_func
// CHECK: mogg._kernel
// CHECK:   mogg._call.shape @{{.*}}.test_custom_op_shape
// CHECK-SAME: -> <2>
mo.graph @custom_op_shape_func<D0, D1 -> D2, D3>(
  %input: !mo.tensor<[D0, D1], f32>
) -> !mo.tensor<[D2, D3], f32> {
  %res = mo.custom<() -> D4, D5> {device = #M.device_ref<"cpu", 0>, symbol = "test_custom_op"}(%input) :
    (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %res : !mo.tensor<[D4, D5], f32>
}

// A trait-typed, metadata-only shape function (`Some[DenseTensor]`, no
// io-role), emitted as `_MoggShapeTensorArg` instead of `InputTensor`.
// `DenseTensor` is data-visible, so `_data_visible = True`.
// EMIT-DAG: struct _MoggShapeTensorArg[
// EMIT-DAG: _MoggShapeTensorArg[type_of({{.*}}).dtype, type_of({{.*}}), 64, True]
// EMIT-DAG: shapeof_trait_metadata_shape_op_map(
// CHECK-LABEL: mo.graph @trait_metadata_shape_func
// CHECK: mogg._kernel
// CHECK:   mogg._call.shape @{{.*}}.trait_metadata_shape_op_shape
// CHECK-SAME: -> <2>
// CHECK-NOT: mogg.input_tensor_arg_contracts
mo.graph @trait_metadata_shape_func<D0, D1 -> D2, D3>(
  %input: !mo.tensor<[D0, D1], f32>
) -> !mo.tensor<[D2, D3], f32> {
  %res = mo.custom<() -> D4, D5> {device = #M.device_ref<"cpu", 0>, symbol = "trait_metadata_shape_op"}(%input) :
    (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %res : !mo.tensor<[D4, D5], f32>
}

// A trait-typed shape function bound to bare `Tensor`. `Tensor` is not
// data-visible, so `_data_visible = False`.
// EMIT-DAG: _MoggShapeTensorArg[type_of({{.*}}).dtype, type_of({{.*}}), 64, False]
// EMIT-DAG: shapeof_trait_tensor_shape_op_map(
// CHECK-LABEL: mo.graph @trait_tensor_shape_func
// CHECK: mogg._kernel
// CHECK:   mogg._call.shape @{{.*}}.trait_tensor_shape_op_shape
// CHECK-SAME: -> <2>
mo.graph @trait_tensor_shape_func<D0, D1 -> D2, D3>(
  %input: !mo.tensor<[D0, D1], f32>
) -> !mo.tensor<[D2, D3], f32> {
  %res = mo.custom<() -> D4, D5> {device = #M.device_ref<"cpu", 0>, symbol = "trait_tensor_shape_op"}(%input) :
    (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %res : !mo.tensor<[D4, D5], f32>
}

// A trait-typed shape function that mixes tiers within one signature
// (`trait_reshape_shape_op_shape[D: Tensor, S: TileTensorable]`) and reads
// the second operand's element values via `to_tile_tensor()`.
// EMIT-DAG: shapeof_trait_reshape_shape_op_map(
// CHECK-LABEL: mo.graph @trait_reshape_shape_func
// CHECK: mogg._kernel
// CHECK:   mogg._call.shape @{{.*}}.trait_reshape_shape_op_shape
// CHECK-SAME: -> <2>
mo.graph @trait_reshape_shape_func<D0, D1 -> D2, D3>(
  %data: !mo.tensor<[D0, D1], f32>,
  %sizes: !mo.tensor<[2], si64>
) -> !mo.tensor<[D2, D3], f32> {
  %res = mo.custom<() -> D4, D5> {device = #M.device_ref<"cpu", 0>, symbol = "trait_reshape_shape_op"}(%data, %sizes) :
    (!mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %res : !mo.tensor<[D4, D5], f32>
}
