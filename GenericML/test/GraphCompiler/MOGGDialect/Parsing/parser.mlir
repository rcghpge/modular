// RUN: mo-opt %s | mo-opt

// Test we can parse the custom mogg operations.

mo.graph @kernelMinimalTest(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[2, 3], f32>, %arg3: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>{
    %1 = mogg.output_placeholders : !mo.tensor<[2, 3], f32>
    mogg.output %1 : !mo.tensor<[2, 3], f32>
  } { shapeFunc = "test", device = #M.device_ref<"cpu", 0>}
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

mo.graph @kernelWithHooksTest(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[2, 3], f32>, %arg3: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>{
    %1 = mogg.output_placeholders : !mo.tensor<[2, 3], f32>
    mogg.output %1 : !mo.tensor<[2, 3], f32>
  } { fusionBodyLambda = "elemwise_compute_lambda", inputFusionInterface = ["", ""], outputFusionInterface = [""], shapeFunc = "test", device = #M.device_ref<"cpu", 0>}
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

mo.graph @kernelsDynamicTest(%arg0: !mo.tensor<[?], f32>, %arg1: !mo.tensor<[?], f32>) -> !mo.tensor<[?], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[?], f32>, %arg3: !mo.tensor<[?], f32>) -> !mo.tensor<[?], f32>{
    %1 = mogg.output_placeholders : !mo.tensor<[?], f32>
    mogg.output %1 : !mo.tensor<[?], f32>
  } { fusionBodyLambda = "elemwise_compute_lambda", inputFusionInterface = ["", ""], outputFusionInterface = [""], shapeFunc = "test", device = #M.device_ref<"cpu", 0>}
  mo.output %0 : !mo.tensor<[?], f32>
}

mo.graph @getDim<N, K, L>(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mogg.kernel (%arg0) : (%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>{
    %output = mogg.output_placeholders : !mo.tensor<[2, 3], f32>

    %0 = mogg.get_dim <N> -> !pop.scalar<si64>
    %1 = mogg.get_dim <K> -> !pop.scalar<si64>
    %2 = mogg.get_dim <L> -> !pop.scalar<si64>

    mogg.output %output : !mo.tensor<[2, 3], f32>
  } { shapeFunc = "test", device = #M.device_ref<"cpu", 0>}
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// Test MOGG_ShapeFuncResult type roundtrip with different ranks
func.func @shapeFuncResultTypeRoundTrip(%shape_rank3: !mogg.shape_func_result<3>,
                                        %shape_rank5: !mogg.shape_func_result<5>) -> (!mogg.shape_func_result<3>, !mogg.shape_func_result<5>) {
  return %shape_rank3, %shape_rank5 : !mogg.shape_func_result<3>, !mogg.shape_func_result<5>
}
