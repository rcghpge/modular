// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @shape_attr_invalid_dim_1(%arg : !mo.tensor<[], f32>) -> (!mo.tensor<[], f32>) {
  kgen.param.declare N = <4>
  // expected-error@+1 {{invalid shape dimension: -1 at position 1}}
  kgen.param.declare S1: !mo.shape = <[N, -1, ?]>
  mo.output %arg : !mo.tensor<[], f32>
}

// -----

mo.graph @shape_attr_invalid_dim_2(%arg : !mo.tensor<[], f32>) -> (!mo.tensor<[], f32>) {
  kgen.param.declare N = <4>
  // expected-error@+1 {{invalid shape dimension: -2 at position 2}}
  kgen.param.declare S1: !mo.shape = <[N, 0, -2]>
  mo.output %arg : !mo.tensor<[], f32>
}

// -----

kgen.func @param_operator_inside_tensor_type(
  // expected-error @below {{TensorType shape parameter must be ShapeAttr or ParamDeclRefAttr}}
  %arg : !mo.tensor<apply(:() -> !mo.shape "mo.shape.foobar"), f32>
) {
  kgen.return
}
