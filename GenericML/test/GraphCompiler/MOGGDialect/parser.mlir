// RUN: mo-opt %s

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

mo.graph @lambdaTest(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[2, 3], f32>, %arg3: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>{
    mogg.lambda @test_lambda = (%arg4: !mogg.indices, %arg5 : !mogg.simd<f32>, %arg6 : !mogg.simd<f32>) -> !mogg.simd<f32>{
      mogg.output %arg5 : !mogg.simd<f32>
    }

    %1 = mogg.output_placeholders : !mo.tensor<[2, 3], f32>
    mogg.output %1 : !mo.tensor<[2, 3], f32>
  } { shapeFunc = "test"}
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

mo.graph @lambdaInvokeTest(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[2, 3], f32>, %arg3: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>{
    mogg.lambda @test_lambda = (%arg4: !mogg.indices, %arg5 : !mogg.simd<f32>, %arg6 : !mogg.simd<f32>) -> !mogg.simd<f32>{
      mogg.output %arg5 : !mogg.simd<f32>
    }

    mogg.lambda @test_lambda2 = (%arg4: !mogg.indices, %arg5 : !mogg.simd<f32>, %arg6 : !mogg.simd<f32>) -> !mogg.simd<f32>{
      %0 = mogg.invoke @test_lambda(%arg4, %arg5, %arg6): !mogg.indices, !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      mogg.output %arg5 : !mogg.simd<f32>
    }

    %1 = mogg.output_placeholders : !mo.tensor<[2, 3], f32>
    mogg.output %1 : !mo.tensor<[2, 3], f32>
  } { shapeFunc = "test"}
  mo.output %0 : !mo.tensor<[2, 3], f32>
}



mo.graph @lambdaCallTest(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mogg.kernel (%arg0) : (%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>{
    %output = mogg.output_placeholders : !mo.tensor<[2, 3], f32>

    mogg.lambda @test_lambda = (%indices: !mogg.indices) -> !mogg.simd<f32>{
      %2 = mogg.call "fake_load"(%input, %indices) : !mo.tensor<[2, 3], f32>, !mogg.indices -> !mogg.simd<f32>
      mogg.output %2 : !mogg.simd<f32>
    }

    mogg.lambda @test_lambda2 = (%indices: !mogg.indices, %value : !mogg.simd<f32>) {
      mogg.call "fake_store"(%output, %indices, %value) : !mo.tensor<[2, 3], f32>, !mogg.indices, !mogg.simd<f32>
      mogg.output
    }

    mogg.call "fake_op" [@test_lambda, @test_lambda2](%input, %output) : !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>

    mogg.output %output : !mo.tensor<[2, 3], f32>
  } { shapeFunc = "test"}
  mo.output %0 : !mo.tensor<[2, 3], f32>
}
