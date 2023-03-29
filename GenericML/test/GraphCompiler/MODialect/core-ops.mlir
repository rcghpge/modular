// RUN: mo-opt --canonicalize %s | mo-opt | FileCheck %s

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

// CHECK: kgen.func @assert(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> (!mo.chain, !mo.chain) {
kgen.func @assert(%arg0: index, %arg1: index) -> (!mo.chain, !mo.chain) {
  // CHECK: %[[VAL0:.*]] = index.cmp eq(%[[ARG0]], %[[ARG1]])
  %0 = index.cmp eq(%arg0, %arg1)
  // CHECK: mo.assert %[[VAL0]]
  %ch0 = mo.assert %0
  // CHECK: mo.assert %[[VAL0]], "args are not equal"
  %ch1 = mo.assert %0, "args are not equal"
  kgen.return %ch0, %ch1 : !mo.chain, !mo.chain
}

//===----------------------------------------------------------------------===//
// GuardOp
//===----------------------------------------------------------------------===//

// CHECK: kgen.func @guard(%[[ARG0:.*]]: !mo.shape, %[[ARG1:.*]]: !mo.shape, %[[COND:.*]]: i1) -> !mo.chain {
kgen.func @guard(%sh0: !mo.shape, %sh1: !mo.shape, %cond: i1) -> !mo.chain {
  // CHECK: %[[CH:.*]] = mo.assert %[[COND]]
  %ch = mo.assert %cond
  // CHECK: mo.guard[%[[CH]]] %[[ARG0]], %[[ARG1]] : !mo.shape, !mo.shape
  mo.guard[%ch] %sh0, %sh1 : !mo.shape, !mo.shape
  // CHECK: kgen.return %[[CH]] : !mo.chain
  kgen.return %ch : !mo.chain
}

//===----------------------------------------------------------------------===//
// IndexToTensorOp
//===----------------------------------------------------------------------===//

// CHECK: kgen.func @index_to_tensor(%[[ARG0:.*]]: index) -> !mo.tensor<[], si64> {
kgen.func @index_to_tensor(%arg0: index) -> !mo.tensor<[], si64> {
  // CHECK: %[[RES:.*]] = mo.index.to_tensor(%[[ARG0]]) -> !mo.tensor<[], si64>
  %0 = mo.index.to_tensor(%arg0) -> !mo.tensor<[], si64>
  // CHECK: kgen.return %[[RES]] : !mo.tensor<[], si64>
  kgen.return %0: !mo.tensor<[], si64>
}

// CHECK: kgen.func @index_to_tensor_fold() -> !mo.tensor<[], si64> {
kgen.func @index_to_tensor_fold() -> !mo.tensor<[], si64> {
  %0 = index.constant 42
  // CHECK-NEXT: %[[RES:.*]] = mo.constant {value = #M.dense_array<42> : tensor<si64>} : !mo.tensor<[], si64>
  %1 = mo.index.to_tensor(%0) -> !mo.tensor<[], si64>
  // CHECK-NEXT: kgen.return %[[RES]] : !mo.tensor<[], si64>
  kgen.return %1: !mo.tensor<[], si64>
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
