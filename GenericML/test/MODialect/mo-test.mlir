// RUN: mo-opt -cse %s | mo-opt | FileCheck %s

// CHECK: mo.graph @dyn_rank(%[[ARG0:.*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
mo.graph @dyn_rank(%arg0: !mo.tensor<?, f32>) -> (!mo.tensor<?, f32>) {
  mo.output %arg0 : !mo.tensor<?, f32>
}

// CHECK: mo.graph @dyn_dims(%[[ARG0:.*]]: !mo.tensor<[2, ?, 1], f32>) -> !mo.tensor<[2, ?, 1], f32> {
mo.graph @dyn_dims(%arg0: !mo.tensor<[2, ?, 1], f32>) -> (!mo.tensor<[2, ?, 1], f32>) {
  mo.output %arg0 : !mo.tensor<[2, ?, 1], f32>
}

// CHECK: mo.graph @empty_dims(%[[ARG0:.*]]: !mo.tensor<[2, 0, 1], f32>) -> !mo.tensor<[2, 0, 1], f32> {
mo.graph @empty_dims(%arg0: !mo.tensor<[2, 0, 1], f32>) -> (!mo.tensor<[2, 0, 1], f32>) {
  mo.output %arg0 : !mo.tensor<[2, 0, 1], f32>
}

// CHECK: mo.graph @scalar(%[[ARG0:.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[], f32> {
mo.graph @scalar(%arg0: !mo.tensor<[], f32>) -> (!mo.tensor<[], f32>) {
  mo.output %arg0 : !mo.tensor<[], f32>
}

// CHECK: mo.graph @invalid_type(%[[ARG0:.*]]: !mo.tensor<[2, 3], invalid>) -> !mo.tensor<[2, 3], invalid> {
mo.graph @invalid_type(%arg0: !mo.tensor<[2, 3], invalid>) -> (!mo.tensor<[2, 3], invalid>) {
  mo.output %arg0 : !mo.tensor<[2, 3], invalid>
}

// CHECK: mo.graph @tensor_str_metadata(%[[ARG0:.*]]: !mo.tensor<[], si32, "foo">) -> !mo.tensor<[], si32, "foo"> {
mo.graph @tensor_str_metadata(%arg0: !mo.tensor<[], si32, "foo">) -> !mo.tensor<[], si32, "foo"> {
  mo.output %arg0 : !mo.tensor<[], si32, "foo">
}

// CHECK: mo.graph @tensor_type_metadata(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32, f16>) -> !mo.tensor<[2, 3], f32, f16> {
mo.graph @tensor_type_metadata(%arg0: !mo.tensor<[2, 3], f32, f16>) -> !mo.tensor<[2, 3], f32, f16> {
  mo.output %arg0 : !mo.tensor<[2, 3], f32, f16>
}

// CHECK: mo.graph @tensor_dict_metadata(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32, {foo = "bar"}>) -> !mo.tensor<[2, 3], f32, {foo = "bar"}> {
mo.graph @tensor_dict_metadata(%arg0: !mo.tensor<[2, 3], f32, {foo = "bar"}>) -> !mo.tensor<[2, 3], f32, {foo = "bar"}> {
  mo.output %arg0 : !mo.tensor<[2, 3], f32, {foo = "bar"}>
}

// CHECK: mo.graph @constant() -> !mo.tensor<[2, 3], f32>
mo.graph @constant() -> !mo.tensor<[2, 3], f32> {
  // CHECK: %[[VAL_0:.*]] = mo.constant {value = #M.dense_array<1.{{0+}}e+00, 2.{{0+}}e+00, 3.{{0+}}e+00, 4.{{0+}}e+00, 5.{{0+}}e+00, 6.{{0+}}e+00> : tensor<2x3xf32>} : !mo.tensor<[2, 3], f32>
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<2x3xf32>} : !mo.tensor<[2, 3], f32>
  // CHECK: mo.output %[[VAL_0]] : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @constant_cse() -> (!mo.tensor<[2], ui32>, !mo.tensor<[2], ui32>)
mo.graph @constant_cse() -> (!mo.tensor<[2], ui32>, !mo.tensor<[2], ui32>) {
  // CHECK: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<1, 2> : tensor<2xui32>} : !mo.tensor<[2], ui32>
  %0 = mo.constant {value = #M.dense_array<1, 2> : tensor<2xui32>} : !mo.tensor<[2], ui32>
  %1 = mo.constant {value = #M.dense_array<1, 2> : tensor<2xui32>} : !mo.tensor<[2], ui32>
  // CHECK: mo.output %[[VAL0]], %[[VAL0]] : !mo.tensor<[2], ui32>, !mo.tensor<[2], ui32>
  mo.output %0, %1 : !mo.tensor<[2], ui32>, !mo.tensor<[2], ui32>
}

// CHECK: mo.graph @two_constants() -> (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], si8>)
mo.graph @two_constants() -> (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], si8>) {
  // CHECK-DAG: %[[VAL_0:.*]] = mo.constant {value = #M.dense_array<0.{{0+}}e+00, 1.{{0+}}e+00, 2.{{0+}}e+00, 3.{{0+}}e+00, 4.{{0+}}e+00, 5.{{0+}}e+00> : tensor<2x3xf32>} : !mo.tensor<[2, 3], f32>
  // CHECK-DAG: %[[VAL_1:.*]] = mo.constant {value = dense_resource<mtfl_buffer> : tensor<4xsi8>} : !mo.tensor<[4], si8>
  %0 = mo.constant {value = #M.dense_array<0.0, 1.0, 2.0, 3.0, 4.0, 5.0> : tensor<2x3xf32>} : !mo.tensor<[2, 3], f32>
  %1 = mo.constant {value = dense_resource<mtfl_buffer> : tensor<4xsi8>} : !mo.tensor<[4], si8>
  // CHECK: mo.output %[[VAL_0]], %[[VAL_1]] : !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si8>
  mo.output %0, %1 : !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si8>
}

// CHECK: mo.graph @shape_of_static_rank(%[[ARG:.*]]: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], si64>
mo.graph @shape_of_static_rank(%arg: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], si64> {
  // CHECK: %[[RES:.*]] = mo.shape_of(%[[ARG]]) : (!mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], si64>
  %res = mo.shape_of(%arg) : (!mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], si64>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[2], si64>
  mo.output %res : !mo.tensor<[2], si64>
}

// CHECK: mo.graph @shape_of_dyn_rank(%[[ARG:.*]]: !mo.tensor<?, f32>) -> !mo.tensor<[?], si64>
mo.graph @shape_of_dyn_rank(%arg: !mo.tensor<?, f32>) -> !mo.tensor<[?], si64> {
  // CHECK: %[[RES:.*]] = mo.shape_of(%[[ARG]]) : (!mo.tensor<?, f32>) -> !mo.tensor<[?], si64>
  %res = mo.shape_of(%arg) : (!mo.tensor<?, f32>) -> !mo.tensor<[?], si64>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[?], si64>
  mo.output %res : !mo.tensor<[?], si64>
}

// CHECK: mo.graph @broadcast_same_rank(%[[FROM:.*]]: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2, 3], f32>
mo.graph @broadcast_same_rank(%from: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  // CHECK: %[[TO:.*]] = mo.constant {value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %to  = mo.constant {value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: %[[RES:.*]] = mo.broadcast_to(%[[FROM]], %[[TO]]) : (!mo.tensor<[1, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<[1, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @broadcast_diff_rank(%[[FROM:.*]]: !mo.tensor<[3], f32>) -> !mo.tensor<[2, 3], f32>
mo.graph @broadcast_diff_rank(%from: !mo.tensor<[3], f32>) -> !mo.tensor<[2, 3], f32> {
  // CHECK: %[[TO:.*]] = mo.constant {value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %to  = mo.constant {value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: %[[RES:.*]] = mo.broadcast_to(%[[FROM]], %[[TO]]) : (!mo.tensor<[3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<[3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @concat(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 5], f32>) -> !mo.tensor<[2, 8], f32> {
mo.graph @concat(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 5], f32>) -> !mo.tensor<[2, 8], f32> {
  // CHECK: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %ax = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK: mo.concat[%[[VAL0]] : !mo.tensor<[], si64>] (%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>) -> !mo.tensor<[2, 8], f32>
  %0 = mo.concat[%ax: !mo.tensor<[], si64>](%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>) -> !mo.tensor<[2, 8], f32>
  mo.output %0 : !mo.tensor<[2, 8], f32>
}

// CHECK: mo.graph @concat_dyn_rank(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<?, f32>) -> !mo.tensor<[2, 8], f32> {
mo.graph @concat_dyn_rank(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<[2, 8], f32> {
  // CHECK: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %ax = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK: mo.concat[%[[VAL0]] : !mo.tensor<[], si64>] (%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<?, f32>) -> !mo.tensor<[2, 8], f32>
  %0 = mo.concat[%ax: !mo.tensor<[], si64>](%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<?, f32>) -> !mo.tensor<[2, 8], f32>
  mo.output %0 : !mo.tensor<[2, 8], f32>
}

// CHECK: mo.graph @gather(%[[PARAMS:.*]]: !mo.tensor<[2, 2], f32>, %[[INDICES:.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32> {
mo.graph @gather(%params: !mo.tensor<[2, 2], f32>, %indices: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32> {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  // CHECK: mo.gather(%[[PARAMS]], %[[INDICES]], %[[AXIS]]) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 2], f32>
  %0 = mo.gather(%params, %indices, %axis) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

// CHECK: mo.graph @gather_dyn_rank_param(%[[PARAMS:.*]]: !mo.tensor<?, f32>, %[[INDICES:.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32> {
mo.graph @gather_dyn_rank_param(%params: !mo.tensor<?, f32>, %indices: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32> {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  // CHECK: mo.gather(%[[PARAMS]], %[[INDICES]], %[[AXIS]]) : (!mo.tensor<?, f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 2], f32>
  %0 = mo.gather(%params, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

{-#
  dialect_resources: {
    builtin: {
      mtfl_buffer: "0x4000000087654321"
    }
  }
#-}
