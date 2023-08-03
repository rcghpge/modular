// RUN: mo-opt %s --split-input-file --verify-diagnostics

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
