// RUN: mo-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect

mo.graph @param_operator_inside_tensor_type(
  // expected-error @below {{TensorType shape parameter must be ShapeAttr or ParamDeclRefAttr}}
  %arg : !mo.tensor<apply(:() -> !mosh.ape "mosh.foobar"), f32>
) {
  mo.output
}

// -----

mo.graph @invalid_layout(
  // expected-error @below {{invalid layout format: "xx"}}
  %arg: !mo.tensor<[10, 10], f32, {layout = #mo.layout<xx>}>
) {
  mo.output
}

// -----

mo.graph @invalid_dtype_attr() {
  // No ops defined which eat DType attributes
  // expected-error @below {{failed to parse MO_DTypeAttr parameter 'dtype' which is to be a `DType`}}
  %foo1 = "fakedialect.bar_op" () {dtype = #mo.dtype<fakedtype>}: () -> (!mo.tensor<[], f32>)
  mo.output
}
