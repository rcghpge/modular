// RUN: mo-opt --canonicalize %s | mo-opt | FileCheck %s

//===----------------------------------------------------------------------===//
// GraphOP, RebindOp and TensorType tests
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @empty_parametric_graph(%[[ARG:.*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
mo.graph @empty_parametric_graph<>(%arg: !mo.tensor<?, f32>) -> (!mo.tensor<?, f32>) {
  // CHECK-NEXT: mo.output %[[ARG]] : !mo.tensor<?, f32>
  mo.output %arg : !mo.tensor<?, f32>
}

// CHECK: mo.graph @nullary_parametric_graph<() -> index>(%[[ARG:.*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
mo.graph @nullary_parametric_graph<() -> index>(%arg: !mo.tensor<?, f32>) -> (!mo.tensor<?, f32>) {
  // CHECK-NEXT: mo.output %[[ARG]] : !mo.tensor<?, f32>
  mo.output %arg : !mo.tensor<?, f32>
}

// CHECK: mo.graph @unary_void_parametric_graph<N>(%[[ARG:.*]]: !mo.tensor<[N], f32>) -> !mo.tensor<[N], f32> {
mo.graph @unary_void_parametric_graph<N: index>(%arg: !mo.tensor<[N], f32>) -> (!mo.tensor<[N], f32>) {
  // CHECK-NEXT: mo.output %[[ARG]] : !mo.tensor<[N], f32>
  mo.output %arg : !mo.tensor<[N], f32>
}

// CHECK: mo.graph @binary_parametric_graph<A, B -> index>(%[[ARG:.*]]: !mo.tensor<[A, B], f32>) -> !mo.tensor<[A, B], f32> {
mo.graph @binary_parametric_graph<A: index, B: index -> index>(%arg: !mo.tensor<[A, B], f32>) -> (!mo.tensor<[A, B], f32>) {
  // CHECK-NEXT: mo.output %[[ARG]] : !mo.tensor<[A, B], f32>
  mo.output %arg : !mo.tensor<[A, B], f32>
}

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

// CHECK: mo.graph @cast_w_params(%[[ARG0:.*]]: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[], si8> {
mo.graph @cast_w_params(%arg0: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[], si8> {
  // CHECK: kgen.param.declare [[N:.*]] = <2>
  kgen.param.declare N = <2>
  // CHECK: kgen.param.declare [[M:.*]] = <5>
  kgen.param.declare M = <5>
  // CHECK: %[[VAL0:.*]] = mo.rebind(%[[ARG0]]) : !mo.tensor<[?, ?], f32> -> !mo.tensor<[N, M], f32>
  %0 = mo.rebind(%arg0) : !mo.tensor<[?, ?], f32> -> !mo.tensor<[N, M], f32>
  // CHECK: kgen.param.declare [[SH:.*]]: !mo.shape = <#mo<shape[[[N]], ?, 3]>>
  kgen.param.declare Sh: !mo.shape = <#mo<shape[N, ?, 3]>>
  // CHECK: %[[VAL1:.*]] = mo.rebind(%[[VAL0]]) : !mo.tensor<[[[N]], [[M]]], f32> -> !mo.tensor<[[SH]], f16>
  %1 = mo.rebind(%0) : !mo.tensor<[N, M], f32> -> !mo.tensor<Sh, f16>
  // CHECK: %[[VAL2:.*]] = mo.rebind(%[[VAL1]]) : !mo.tensor<[[SH]], f16> -> !mo.tensor<[], si8>
  %2 = mo.rebind(%1) : !mo.tensor<Sh, f16> -> !mo.tensor<[], si8>

  // CHECK-NEXT: mo.output %[[VAL2]] : !mo.tensor<[], si8>
  %3 = mo.rebind(%2) : !mo.tensor<[], si8> -> !mo.tensor<[], si8>
  mo.output %3 : !mo.tensor<[], si8>
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

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
  // CHECK-DAG: %[[VAL_1:.*]] = mo.constant {value = dense_resource<some_buffer> : tensor<4xsi8>} : !mo.tensor<[4], si8>
  %0 = mo.constant {value = #M.dense_array<0.0, 1.0, 2.0, 3.0, 4.0, 5.0> : tensor<2x3xf32>} : !mo.tensor<[2, 3], f32>
  %1 = mo.constant {value = dense_resource<some_buffer> : tensor<4xsi8>} : !mo.tensor<[4], si8>
  // CHECK: mo.output %[[VAL_0]], %[[VAL_1]] : !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si8>
  mo.output %0, %1 : !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si8>
}

{-#
  dialect_resources: {
    builtin: {
      some_buffer: "0x4000000087654321"
    }
  }
#-}
