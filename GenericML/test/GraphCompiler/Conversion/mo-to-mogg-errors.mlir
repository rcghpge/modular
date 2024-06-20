// RUN: mo-opt --mo-to-mogg %s --verify-diagnostics --split-input-file 

mo.graph @unimplemented_custom_op() -> !mo.tensor<[1], f32> {
  %0 = mo.constant {value = #M.dense_array<1.0> : tensor<1xf32>} : !mo.tensor<[1], f32>
  // expected-error @+1 {{'mo.custom' op [MO_TO_MOGG] Could not find mojo kernel registered for mo.unimplemented_op}}
  %1 = mo.custom {symbol="mo.unimplemented_op"}(%0) : (!mo.tensor<[1], f32>) -> !mo.tensor<[1], f32>
  mo.output %1 : !mo.tensor<[1], f32>
}

// -----

mo.graph @unimplemented_custom_op<D0, D1, D2, D3, D4, D5>(
  %n: !mo.tensor<[], f32>,
  %lhs: !mo.tensor<[D0, D1, D2], f32>,
  %rhs: !mo.tensor<[D3, D4, D5], f32>,
  %target_shape: !mo.tensor<[D0], si32>
) {
  // expected-error @+1 {{'mo.invoke_shape_func' op [MO_TO_MOGG] dynamic rank isn't supported in shape function}}
  mo.invoke_shape_func["batch_matmul_shape"]<() -> Sh3: !mosh.ape>(%lhs, %rhs)
    {dataDeptTensors = []} :
    (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[D3, D4, D5], f32>)
  mo.output
}
