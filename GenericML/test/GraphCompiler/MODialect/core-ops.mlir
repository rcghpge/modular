// RUN: mo-opt --canonicalize %s | mo-opt | FileCheck %s

//===----------------------------------------------------------------------===//
// IndexToTensorOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @index_to_tensor(%[[ARG0:.*]]: index) -> !mo.tensor<[], si64> {
mo.graph @index_to_tensor(%arg0: index) -> !mo.tensor<[], si64> {
  // CHECK: %[[RES:.*]] = mo.index.to_tensor(%[[ARG0]]) -> !mo.tensor<[], si64>
  %0 = mo.index.to_tensor(%arg0) -> !mo.tensor<[], si64>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[], si64>
  mo.output %0: !mo.tensor<[], si64>
}

// CHECK: mo.graph @index_to_tensor_fold() -> !mo.tensor<[], si64> {
mo.graph @index_to_tensor_fold() -> !mo.tensor<[], si64> {
  %0 = index.constant 42
  // CHECK-NEXT: %[[RES:.*]] = mo.constant {value = #M.dense_array<42> : tensor<si64>} : !mo.tensor<[], si64>
  %1 = mo.index.to_tensor(%0) -> !mo.tensor<[], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[], si64>
  mo.output %1: !mo.tensor<[], si64>
}

//===----------------------------------------------------------------------===//
// GraphOP and TensorType tests
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @empty_parametric_graph(%[[ARG:.*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
mo.graph @empty_parametric_graph<>(%arg: !mo.tensor<?, f32>) -> (!mo.tensor<?, f32>) {
  // CHECK-NEXT: mo.output %[[ARG]] : !mo.tensor<?, f32>
  mo.output %arg : !mo.tensor<?, f32>
}

// CHECK: mo.graph @nullary_parametric_graph<() -> [[R1:.*]]>(%[[ARG:.*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
mo.graph @nullary_parametric_graph<() -> R1>(%arg: !mo.tensor<?, f32>) -> (!mo.tensor<?, f32>) {
  // CHECK-NEXT: mo.output<42> %[[ARG]] : !mo.tensor<?, f32>
  mo.output<42> %arg : !mo.tensor<?, f32>
}

// CHECK: mo.graph @unary_void_parametric_graph<[[N:.*]]>(%[[ARG:.*]]: !mo.tensor<[[[N]]], f32>) -> !mo.tensor<[[[N]]], f32> {
mo.graph @unary_void_parametric_graph<N: index>(%arg: !mo.tensor<[N], f32>) -> (!mo.tensor<[N], f32>) {
  // CHECK-NEXT: mo.output %[[ARG]] : !mo.tensor<[[[N]]], f32>
  mo.output %arg : !mo.tensor<[N], f32>
}

// CHECK: mo.graph @binary_parametric_graph<[[A:.*]], [[B:.*]] -> [[R1:.*]]>(%[[ARG:.*]]: !mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[[[A]], [[B]]], f32> {
mo.graph @binary_parametric_graph<A: index, B: index -> R1>(%arg: !mo.tensor<[A, B], f32>) -> (!mo.tensor<[A, B], f32>) {
  // CHECK-NEXT: mo.output<42> %[[ARG]] : !mo.tensor<[[[A]], [[B]]], f32>
  mo.output<42> %arg : !mo.tensor<[A, B], f32>
}

// CHECK: mo.graph @parametric_result<() -> [[D2:.*]]>() {
mo.graph @parametric_result<() -> D2>() {
  // CHECK-NEXT: kgen.param.declare [[D1:.*]] = <42>
  kgen.param.declare D1 = <42>
  // CHECK-NEXT: mo.output<[[D1]]>
  mo.output<D1>
}

// CHECK: mo.graph @output_with_bindings<[[D1:.*]] -> [[D2:.*]]>(%[[ARG:.*]]: !mo.tensor<[[[D1]]], f32>) -> !mo.tensor<[[[D2]]], f32> {
mo.graph @output_with_bindings<D1 -> D2>(%arg: !mo.tensor<[D1], f32>) -> (!mo.tensor<[D2], f32>) {
  // CHECK-NEXT: mo.output<D1> %[[ARG]] : !mo.tensor<[[[D1]]], f32>
  mo.output<D1> %arg : !mo.tensor<[D1], f32>
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

// CHECK: mo.graph @tensor_int_metadata(%[[ARG0:.*]]: !mo.tensor<[], si32, {foo = 42 : i64}>) -> !mo.tensor<[], si32, {foo = 42 : i64}> {
mo.graph @tensor_int_metadata(%arg0: !mo.tensor<[], si32, {foo = 42}>) -> !mo.tensor<[], si32, {foo = 42}> {
  mo.output %arg0 : !mo.tensor<[], si32, {foo = 42}>
}

// CHECK: mo.graph @tensor_str_metadata(%[[ARG0:.*]]: !mo.tensor<[], si32, {bar = "str"}>) -> !mo.tensor<[], si32, {bar = "str"}> {
  mo.graph @tensor_str_metadata(%arg0: !mo.tensor<[], si32, {bar = "str"}>) -> !mo.tensor<[], si32, {bar = "str"}> {
  mo.output %arg0 : !mo.tensor<[], si32, {bar = "str"}>
}

// CHECK: mo.graph @tensor_float_metadata(%[[ARG0:.*]]: !mo.tensor<[], si32, {fp = 4.200000e+00 : f64}>) -> !mo.tensor<[], si32, {fp = 4.200000e+00 : f64}> {
  mo.graph @tensor_float_metadata(%arg0: !mo.tensor<[], si32, {fp = 4.2}>) -> !mo.tensor<[], si32, {fp = 4.2}> {
  mo.output %arg0 : !mo.tensor<[], si32, {fp = 4.2}>
}

//===----------------------------------------------------------------------===//
// IfOp and YieldOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @if_op(%[[ARG0:.*]]: !mo.tensor<[2, ?, 1], f32>, %[[ARG1:.*]]: !mo.tensor<[2, ?, 1], f32>, %[[ARG2:.*]]: !mo.tensor<[], bool>) -> !mo.tensor<[2, ?, 1], f32> {
mo.graph @if_op(%arg0: !mo.tensor<[2, ?, 1], f32>, %arg1: !mo.tensor<[2, ?, 1], f32>, %cond : !mo.tensor<[], bool>) -> (!mo.tensor<[2, ?, 1], f32>) {
  // CHECK: %0 = mo.if(%[[ARG2]] : !mo.tensor<[], bool>) -> !mo.tensor<[2, ?, 1], f32> {
  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2, ?, 1], f32> {
    // CHECK: mo.yield %[[ARG0]] : !mo.tensor<[2, ?, 1], f32>
    mo.yield %arg0 : !mo.tensor<[2, ?, 1], f32>
  // CHECK: } else {
  } else {
    // CHECK: mo.yield %[[ARG1]] : !mo.tensor<[2, ?, 1], f32>
    mo.yield %arg1 : !mo.tensor<[2, ?, 1], f32>
  }
  // CHECK: mo.output %0 : !mo.tensor<[2, ?, 1], f32>
  mo.output %0 : !mo.tensor<[2, ?, 1], f32>
}

// CHECK: mo.graph @empty_if_op(%[[ARG0:.*]]: !mo.tensor<[2, ?, 1], f32>, %[[ARG2:.*]]: !mo.tensor<[], bool>) -> !mo.tensor<[2, ?, 1], f32> {
mo.graph @empty_if_op(%arg0: !mo.tensor<[2, ?, 1], f32>, %cond : !mo.tensor<[], bool>) -> (!mo.tensor<[2, ?, 1], f32>) {
  // CHECK: mo.output %[[ARG0]] : !mo.tensor<[2, ?, 1], f32>
  mo.if(%cond : !mo.tensor<[], bool>) {
    mo.yield
  } else {
    mo.yield
  }
  mo.output %arg0 : !mo.tensor<[2, ?, 1], f32>
}

//===----------------------------------------------------------------------===//
// WhileOp and YieldOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @while_op(%[[ARG0:.*]]: !mo.tensor<[2], f32>, %[[COND:.*]]: !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
mo.graph @while_op(%arg0: !mo.tensor<[2], f32>, %cond : !mo.tensor<[], bool>) -> (!mo.tensor<[2], f32>) {
  // CHECK: %[[RES:.*]]:2 = mo.while (%[[ARG0]]: !mo.tensor<[2], f32>, %[[COND]]: !mo.tensor<[], bool>) {
  %0:2 = mo.while (%arg0 : !mo.tensor<[2], f32>, %cond: !mo.tensor<[], bool>) {
    // CHECK: mo.yield %[[COND]] : !mo.tensor<[], bool>
    mo.yield %cond : !mo.tensor<[], bool>
  // CHECK: } do {
  } do {
    // CHECK: %[[NEW_ARG0:.*]] = mo.add(%[[ARG0]], %[[ARG0]]) : !mo.tensor<[2], f32>
    %new_arg0 = mo.add (%arg0, %arg0) : !mo.tensor<[2], f32>
    // CHECK: mo.yield %[[NEW_ARG0]], %[[COND]] : !mo.tensor<[2], f32>, !mo.tensor<[], bool>
    mo.yield %new_arg0, %cond : !mo.tensor<[2], f32>, !mo.tensor<[], bool>
  }
  // CHECK: mo.output %[[RES]]#0 : !mo.tensor<[2], f32>
  mo.output %0#0 : !mo.tensor<[2], f32>
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

// CHECK: mo.graph @constant_unknown_shape_fold() -> !mo.tensor<[2, ?], si32>
mo.graph @constant_unknown_shape_fold() -> !mo.tensor<[2, ?], si32> {
  // CHECK-NEXT: %[[VAL:.*]] = mo.constant {value = #M.dense_array<1, 2, 3, 4> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>
  // CHECK-NEXT: %[[VAL_REB:.*]] = mo.rebind(%[[VAL]]) : !mo.tensor<[2, 2], si32> -> !mo.tensor<[2, ?], si32>
  %0 = mo.constant {value = #M.dense_array<1, 2, 3, 4> : tensor<2x2xsi32>} : !mo.tensor<[2, ?], si32>
  // CHECK-NEXT: mo.output %[[VAL_REB]] : !mo.tensor<[2, ?], si32>
  mo.output %0 : !mo.tensor<[2, ?], si32>
}

{-#
  dialect_resources: {
    builtin: {
      some_buffer: "0x4000000087654321"
    }
  }
#-}
