// RUN: EMIT_MOJO=1 mo-opt %s --mo-to-mgp="constant-fold=false" --mlir-print-ir-after=mo-to-mogg 2>&1 | FileCheck %s

// Test that dynamic dimensions from kernel result types are properly passed to
// the kernel stub. This tests the fix for passing dynamic dimensions from
// result types of operations inside the kernel stub.

// This test uses mo.gather which produces output with dynamic dimensions.
// The fix ensures that these dynamic dimensions are properly passed to the
// kernel stub via outputParamDecls.

// CHECK-LABEL: mo.graph @gather_dynamic_output_dims
// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.shape["mo.gather"]
// The dynamic dimensions D4, D5 from gather's result type should be passed
// to the kernel stub as output parameter declarations.
// CHECK: outputParamDecls = #kgen<param.decls[{{.*}}D4 : si64{{.*}}D5 : si64{{.*}}]>
mo.graph @gather_dynamic_output_dims<D0, D1 -> D2, D3>(
  %input : !mo.tensor<[D0, 64], f32>,
  %indices : !mo.tensor<[D1], si32>,
  %axis: !mo.tensor<[], si32>
) -> !mo.tensor<[D2, D3], f32> {
  // The gather operation uses new dimension names (D4, D5) for its output
  %res = mo.gather<() -> D4, D5>(%input, %indices, %axis) :
    (!mo.tensor<[D0, 64], f32>, !mo.tensor<[D1], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %res : !mo.tensor<[D4, D5], f32>
}

// CHECK-LABEL: mo.graph @avg_pool_dynamic_output_dims
// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.shape["mo.avg_pool"]
// The dynamic dimensions D3, D4 from avg_pool's result type should be passed
// to the kernel stub as output parameter declarations.
// CHECK: outputParamDecls = #kgen<param.decls[{{.*}}D3 : si64{{.*}}D4 : si64{{.*}}]>
mo.graph @avg_pool_dynamic_output_dims<() -> D1, D2>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %filter_shape: !mo.tensor<[2], si64>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>,
  %paddings: !mo.tensor<[4], si64>
) -> !mo.tensor<[32, D1, D2, 3], f32> {
  // The avg_pool operation uses new dimension names (D3, D4) for its output
  %res = mo.avg_pool<() -> D3, D4>(%input) [filter_shape = %filter_shape, strides = %strides, dilations = %dilations, paddings = %paddings] {count_boundary = false} :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
    -> !mo.tensor<[32, D3, D4, 3], f32>
  mo.output<D3, D4> %res : !mo.tensor<[32, D3, D4, 3], f32>
}
