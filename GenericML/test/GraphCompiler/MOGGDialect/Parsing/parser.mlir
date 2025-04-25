// RUN: mo-opt %s | mo-opt

// Test we can parse the custom mogg operations.

mo.graph @kernelMinimalTest(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[2, 3], f32>, %arg3: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>{
    %1 = mogg.output_placeholders : !mo.tensor<[2, 3], f32>
    mogg.output %1 : !mo.tensor<[2, 3], f32>
  } { shapeFunc = "test"}
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

mo.graph @kernelWithHooksTest(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[2, 3], f32>, %arg3: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>{
    %1 = mogg.output_placeholders : !mo.tensor<[2, 3], f32>
    mogg.output %1 : !mo.tensor<[2, 3], f32>
  } { fusionBodyLambda = "elemwise_compute_lambda", inputFusionInterface = ["", ""], outputFusionInterface = [""], shapeFunc = "test"}
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

mo.graph @kernelsDynamicTest(%arg0: !mo.tensor<[?], f32>, %arg1: !mo.tensor<[?], f32>) -> !mo.tensor<[?], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[?], f32>, %arg3: !mo.tensor<[?], f32>) -> !mo.tensor<[?], f32>{
    %1 = mogg.output_placeholders : !mo.tensor<[?], f32>
    mogg.output %1 : !mo.tensor<[?], f32>
  } { fusionBodyLambda = "elemwise_compute_lambda", inputFusionInterface = ["", ""], outputFusionInterface = [""], shapeFunc = "test"}
  mo.output %0 : !mo.tensor<[?], f32>
}

mo.graph @getDim<N, K, L>(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mogg.kernel (%arg0) : (%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>{
    %output = mogg.output_placeholders : !mo.tensor<[2, 3], f32>

    %0 = mogg.get_dim <N> -> !pop.scalar<si64>
    %1 = mogg.get_dim <K> -> !pop.scalar<si64>
    %2 = mogg.get_dim <L> -> !pop.scalar<si64>

    mogg.output %output : !mo.tensor<[2, 3], f32>
  } { shapeFunc = "test"}
  mo.output %0 : !mo.tensor<[2, 3], f32>
}


mo.graph @stress_test(%arg2: !mo.tensor<[3, 4], f64>, %arg3: !mo.tensor<[4, 3], f64>) -> !mo.tensor<[3, 3], f32> {
  // mogg-based in/out fusible kernel representation.
  %0 = mogg.experimental.kernel (%arg2: !mo.tensor<[3, 4], f64>, %arg3: !mo.tensor<[4, 3], f64>) -> !mo.tensor<[3, 3], f64> {
    %2 = mogg.output_placeholders : !mo.tensor<[3, 3], f64>
    %3 = mogg.bind (%2) output_lambda (%arg9: f64, %arg10: !pop.array<2, index>) {
      mogg.output %arg9, %arg10 : f64, !pop.array<2, index>
    }: !mo.tensor<[3, 3], f64>
    mogg.call.execute["mo.matmul"] inputs(%arg2, %arg3 : !mo.tensor<[3, 4], f64>, !mo.tensor<[4, 3], f64>) outputs(%3: !mo.tensor<[3, 3], f64>)
  } {device = #M.device_ref<"cpu", 0>, isAsync = true, summary = "mo.matmul:11"}

  // mogg-based element-wise kernel representation
  %1 = mogg.experimental.kernel (%0: !mo.tensor<[3, 3], f64>) -> !mo.tensor<[3, 3], f32> {
    %2 = mogg.output_placeholders : !mo.tensor<[3, 3], f32>
    %3 = mogg.bind (%0) elemwise_lambda (%arg8: f64, %arg9: !pop.array<2, index>) {
      %4 = mogg.call.outlined["mo.cast"] (%arg8) : f64 -> f32
      mogg.output %4 : f32
    }: (!mo.tensor<[3, 3], f64>) -> !mo.tensor<[3, 3], f32>
    mogg.call.foreach input(%3) output(%2) : !mo.tensor<[3, 3], f32>
  } {device = #M.device_ref<"cpu", 0>, elementwise, isAsync = true, summary = "mo.cast:12"}
  mo.output %1 : !mo.tensor<[3, 3], f32>
}
