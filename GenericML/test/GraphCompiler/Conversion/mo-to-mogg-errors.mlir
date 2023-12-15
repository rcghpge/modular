// RUN: mo-opt --mo-to-mogg %s --verify-diagnostics

mo.graph @unimplemented_custom_op() -> !mo.tensor<[1], f32> {
 %0 = mo.constant {value = #M.dense_array<1.0> : tensor<1xf32>} : !mo.tensor<[1], f32>
// expected-error @+1 {{'mo.custom' op [MO_TO_MOGG] Could not find mojo kernel registered for mo.unimplemented_op}}
 %1 = mo.custom {symbol="mo.unimplemented_op"}(%0) (!mo.tensor<[1], f32>) -> !mo.tensor<[1], f32>
 mo.output %1 : !mo.tensor<[1], f32>
}
