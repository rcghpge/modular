// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @param_operator_inside_tensor_type(
  // expected-error @below {{TensorType shape parameter must be ShapeAttr or ParamDeclRefAttr}}
  %arg : !mo.tensor<apply(:() -> !mosh.ape "mosh.foobar"), f32>
) {
  kgen.return
}
