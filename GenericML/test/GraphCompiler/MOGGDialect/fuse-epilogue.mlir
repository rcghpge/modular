// RUN: mo-opt --mo-to-mogg=use-jit=true --fuse-elementwise --cse --fuse-bcasts --cse --fuse-in-out-lambdas --cse %s | FileCheck %s

mo.graph @basicMatmul<N, M, K>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32, {layout = #mo.layout<KN>}>, %arg3: !mo.tensor<[], f32>) -> (!mo.tensor<[M, N], f32>) {
  %mm = mo.matmul(%arg0, %arg1) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32, {layout = #mo.layout<KN>}>)  -> !mo.tensor<[M, N], f32>
  %bcast = mo.static.broadcast_to(%arg3) : !mo.tensor<[], f32> -> !mo.tensor<[M, N], f32>
  %1 = mo.add(%mm, %bcast) : !mo.tensor<[M, N], f32>
  mo.output %1 : !mo.tensor<[M, N], f32>
}

// CHECK-LABEL: mo.graph @basicMatmul
// CHECK: %[[OUT:.*]] = mogg.kernel
// CHECK-NEXT: mogg.output_placeholders
// CHECK-NEXT: mogg.param_shape_to_value
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"
// CHECK-NEXT: mogg.lambda @output_0_fn
// CHECK-NEXT: mogg.simd_load
// CHECK-NEXT: mogg.call "mo.add"
// CHECK-NEXT: mogg.call "simd_store"
// CHECK-NEXT: mogg.output
// CHECK-NEXT: }
// CHECK-NEXT: mogg.call "mo.matmul" [@output_0_fn{{.*}}{lambdas_have_fusion = true
// CHECK: mo.output %[[OUT]]

mo.graph @noFuseMultiUse<N, M, K>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32, {layout = #mo.layout<KN>}>, %arg3: !mo.tensor<[], f32>, %arg4: !mo.tensor<[], f32>) -> (!mo.tensor<[M, N], f32>, !mo.tensor<[M, N], f32>) {
  %mm = mo.matmul(%arg0, %arg1) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32, {layout = #mo.layout<KN>}>)  -> !mo.tensor<[M, N], f32>
  %bcast = mo.static.broadcast_to(%arg3) : !mo.tensor<[], f32> -> !mo.tensor<[M, N], f32>
  %1 = mo.add(%mm, %bcast) : !mo.tensor<[M, N], f32>
  %bcast2 = mo.static.broadcast_to(%arg4) : !mo.tensor<[], f32> -> !mo.tensor<[M, N], f32>
  %2 = mo.add(%mm, %bcast2) : !mo.tensor<[M, N], f32>
  mo.output %1, %2 : !mo.tensor<[M, N], f32>, !mo.tensor<[M, N], f32>
}

// CHECK-LABEL: mo.graph @noFuseMultiUse
// CHECK: %[[OUT:.*]] = mogg.kernel
// CHECK: %[[OUT2:.*]] = mogg.kernel (%[[OUT]]
// CHECK: %[[OUT3:.*]] = mogg.kernel (%[[OUT]]
// CHECK: mo.output %[[OUT2]], %[[OUT3]]

mo.graph @checkNoFuseInputInterfaceDisabled<N, M, K>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[M, K], f32>, %arg3: !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32>{
    %2 = mogg.output_placeholders : !mo.tensor<[M, N], f32>
    mogg.lambda @output_0_fn = (%arg4: !mogg.indices, %arg5: !mogg.simd<f32>){
        mogg.call "simd_store"(%2, %arg4, %arg5) : !mo.tensor<[M, N], f32>, !mogg.indices, !mogg.simd<f32>
        mogg.output
    }
    mogg.call "mo.matmul" [@output_0_fn](%arg2, %arg3, %2) {lambdas_have_fusion = false, packed_in_1 = false, single_thread_blocking_override = false, transpose_in_1 = false} : !mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32>, !mo.tensor<[M, N], f32>
    mogg.output %2 : !mo.tensor<[M, N], f32>
  } {outputFusionInterface = ["output_0_fn"]}
  %1 = mogg.kernel (%0) : (%arg2: !mo.tensor<[M, N], f32>) -> !mo.tensor<[M, N], f32>{
    %2 = mogg.output_placeholders : !mo.tensor<[M, N], f32>
    mogg.lambda @input_0_fn = (%arg3: !mogg.indices) -> !mogg.simd<f32>{
        %3 = mogg.simd_load(%arg2, %arg3) : (!mo.tensor<[M, N], f32>, !mogg.indices) -> !mogg.simd<f32>
        mogg.output %3 : !mogg.simd<f32>
    }
    mogg.lambda @output_0_fn = (%arg3: !mogg.indices, %arg4: !mogg.simd<f32>){
        mogg.call "simd_store"(%2, %arg3, %arg4) : !mo.tensor<[M, N], f32>, !mogg.indices, !mogg.simd<f32>
        mogg.output
    }
  mogg.lambda @elemwise_compute_lambda = (%arg3: !mogg.indices){
    %3 = mogg.invoke @input_0_fn(%arg3) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
    %4 = mogg.call "mo.relu"(%3) : !mogg.simd<f32> -> !mogg.simd<f32>
    mogg.invoke @output_0_fn(%arg3, %4) : !mogg.indices, !mogg.simd<f32>
    mogg.output
  }
    mogg.call "elementwise" [@elemwise_compute_lambda](%2) {single_thread_blocking_override = false} : !mo.tensor<[M, N], f32>
    mogg.output %2 : !mo.tensor<[M, N], f32>
  } {fusionBodyLambda = "elemwise_compute_lambda", inputFusionInterface = [""], outputFusionInterface = ["output_0_fn"]}
  mo.output %1 : !mo.tensor<[M, N], f32>
}

// CHECK-LABEL: @checkNoFuseInputInterfaceDisabled
// CHECK: %[[OUT:.*]] = mogg.kernel
// CHECK: %[[OUT2:.*]] = mogg.kernel (%[[OUT]])
// CHECK: mo.output %[[OUT2]]

mo.graph @checkNoFuseOutputInterfaceDisabled<N, M, K>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[M, K], f32>, %arg3: !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32>{
    %2 = mogg.output_placeholders : !mo.tensor<[M, N], f32>
    mogg.lambda @output_0_fn = (%arg4: !mogg.indices, %arg5: !mogg.simd<f32>){
        mogg.call "simd_store"(%2, %arg4, %arg5) : !mo.tensor<[M, N], f32>, !mogg.indices, !mogg.simd<f32>
        mogg.output
    }
    mogg.call "mo.matmul" [@output_0_fn](%arg2, %arg3, %2) {lambdas_have_fusion = false, packed_in_1 = false, single_thread_blocking_override = false, transpose_in_1 = false} : !mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32>, !mo.tensor<[M, N], f32>
    mogg.output %2 : !mo.tensor<[M, N], f32>
  }
  %1 = mogg.kernel (%0) : (%arg2: !mo.tensor<[M, N], f32>) -> !mo.tensor<[M, N], f32>{
    %2 = mogg.output_placeholders : !mo.tensor<[M, N], f32>
    mogg.lambda @input_0_fn = (%arg3: !mogg.indices) -> !mogg.simd<f32>{
        %3 = mogg.simd_load(%arg2, %arg3) : (!mo.tensor<[M, N], f32>, !mogg.indices) -> !mogg.simd<f32>
        mogg.output %3 : !mogg.simd<f32>
    }
    mogg.lambda @output_0_fn = (%arg3: !mogg.indices, %arg4: !mogg.simd<f32>){
        mogg.call "simd_store"(%2, %arg3, %arg4) : !mo.tensor<[M, N], f32>, !mogg.indices, !mogg.simd<f32>
        mogg.output
    }
  mogg.lambda @elemwise_compute_lambda = (%arg3: !mogg.indices){
    %3 = mogg.invoke @input_0_fn(%arg3) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
    %4 = mogg.call "mo.relu"(%3) : !mogg.simd<f32> -> !mogg.simd<f32>
    mogg.invoke @output_0_fn(%arg3, %4) : !mogg.indices, !mogg.simd<f32>
    mogg.output
  }
    mogg.call "elementwise" [@elemwise_compute_lambda](%2) {single_thread_blocking_override = false} : !mo.tensor<[M, N], f32>
    mogg.output %2 : !mo.tensor<[M, N], f32>
  } {fusionBodyLambda = "elemwise_compute_lambda", inputFusionInterface = ["input_0_fn"], outputFusionInterface = ["output_0_fn"]}
  mo.output %1 : !mo.tensor<[M, N], f32>
}

// CHECK-LABEL: @checkNoFuseOutputInterfaceDisabled
// CHECK: %[[OUT:.*]] = mogg.kernel
// CHECK: %[[OUT2:.*]] = mogg.kernel (%[[OUT]])
// CHECK: mo.output %[[OUT2]]

mo.graph @checkNoFuseNoElementwiseAttr<N, M, K>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[M, K], f32>, %arg3: !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32>{
    %2 = mogg.output_placeholders : !mo.tensor<[M, N], f32>
    mogg.lambda @output_0_fn = (%arg4: !mogg.indices, %arg5: !mogg.simd<f32>){
        mogg.call "simd_store"(%2, %arg4, %arg5) : !mo.tensor<[M, N], f32>, !mogg.indices, !mogg.simd<f32>
        mogg.output
    }
    mogg.call "mo.matmul" [@output_0_fn](%arg2, %arg3, %2) {lambdas_have_fusion = false, packed_in_1 = false, single_thread_blocking_override = false, transpose_in_1 = false} : !mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32>, !mo.tensor<[M, N], f32>
    mogg.output %2 : !mo.tensor<[M, N], f32>
  } {outputFusionInterface = ["output_0_fn"]}
  %1 = mogg.kernel (%0) : (%arg2: !mo.tensor<[M, N], f32>) -> !mo.tensor<[M, N], f32>{
    %2 = mogg.output_placeholders : !mo.tensor<[M, N], f32>
    mogg.lambda @input_0_fn = (%arg3: !mogg.indices) -> !mogg.simd<f32>{
        %3 = mogg.simd_load(%arg2, %arg3) : (!mo.tensor<[M, N], f32>, !mogg.indices) -> !mogg.simd<f32>
        mogg.output %3 : !mogg.simd<f32>
    }
    mogg.lambda @output_0_fn = (%arg3: !mogg.indices, %arg4: !mogg.simd<f32>){
        mogg.call "simd_store"(%2, %arg3, %arg4) : !mo.tensor<[M, N], f32>, !mogg.indices, !mogg.simd<f32>
        mogg.output
    }
  mogg.lambda @elemwise_compute_lambda = (%arg3: !mogg.indices){
    %3 = mogg.invoke @input_0_fn(%arg3) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
    %4 = mogg.call "mo.relu"(%3) : !mogg.simd<f32> -> !mogg.simd<f32>
    mogg.invoke @output_0_fn(%arg3, %4) : !mogg.indices, !mogg.simd<f32>
    mogg.output
  }
    mogg.call "elementwise" [@elemwise_compute_lambda](%2) {single_thread_blocking_override = false} : !mo.tensor<[M, N], f32>
    mogg.output %2 : !mo.tensor<[M, N], f32>
  } {inputFusionInterface = ["input_0_fn"], outputFusionInterface = ["output_0_fn"]}
  mo.output %1 : !mo.tensor<[M, N], f32>
}

// CHECK-LABEL: @checkNoFuseNoElementwiseAttr
// CHECK: %[[OUT:.*]] = mogg.kernel
// CHECK: %[[OUT2:.*]] = mogg.kernel (%[[OUT]])
// CHECK: mo.output %[[OUT2]]

// Check when fusing two kernels with different output dtypes, we only do input fusion
mo.graph @checkNoFuseDifferentOutDTypes() -> !mo.tensor<[3, 3], f32> {
  %0 = mo.constant {value = #M.dense_array<1, 2, 3, 4, 5, 6, 7, 8, 9> : tensor<3x3xsi32>} : !mo.tensor<[3, 3], si32, {layout = #mo.layout<KN>}>
  %1 = mo.constant {value = #M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00> : tensor<3x3xf32>} : !mo.tensor<[3, 3], f32>
  %2 = mogg.kernel (%0, %0) : (%arg0: !mo.tensor<[3, 3], si32, {layout = #mo.layout<KN>}>, %arg1: !mo.tensor<[3, 3], si32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[3, 3], si32>{
    %4 = mogg.output_placeholders : !mo.tensor<[3, 3], si32>
    mogg.lambda @output_0_fn = (%arg2: !mogg.indices, %arg3: !mogg.simd<si32>){
      mogg.call "simd_store"(%4, %arg2, %arg3) : !mo.tensor<[3, 3], si32>, !mogg.indices, !mogg.simd<si32>
      mogg.output
    }
    mogg.call "mo.matmul" [@output_0_fn](%arg0, %arg1, %4) {lambdas_have_fusion = false, packed_in_1 = false, single_thread_blocking_override = false, transpose_in_1 = false} : !mo.tensor<[3, 3], si32, {layout = #mo.layout<KN>}>, !mo.tensor<[3, 3], si32, {layout = #mo.layout<KN>}>, !mo.tensor<[3, 3], si32>
    mogg.output %4 : !mo.tensor<[3, 3], si32>
  } {outputFusionInterface = ["output_0_fn"]}
  %3 = mogg.kernel (%1, %2) : (%arg0: !mo.tensor<[3, 3], f32>, %arg1: !mo.tensor<[3, 3], si32>) -> !mo.tensor<[3, 3], f32>{
    %4 = mogg.output_placeholders : !mo.tensor<[3, 3], f32>
    mogg.lambda @input_0_fn_0 = (%arg2: !mogg.indices) -> !mogg.simd<si32>{
      %5 = mogg.simd_load(%arg1, %arg2) : (!mo.tensor<[3, 3], si32>, !mogg.indices) -> !mogg.simd<si32>
      mogg.output %5 : !mogg.simd<si32>
    }
    mogg.lambda @input_1_fn = (%arg2: !mogg.indices) -> !mogg.simd<f32>{
      %5 = mogg.simd_load(%arg0, %arg2) : (!mo.tensor<[3, 3], f32>, !mogg.indices) -> !mogg.simd<f32>
      mogg.output %5 : !mogg.simd<f32>
    }
    mogg.lambda @output_0_fn = (%arg2: !mogg.indices, %arg3: !mogg.simd<f32>){
      mogg.call "simd_store"(%4, %arg2, %arg3) : !mo.tensor<[3, 3], f32>, !mogg.indices, !mogg.simd<f32>
      mogg.output
    }
    mogg.lambda @elemwise_compute_lambda = (%arg2: !mogg.indices){
      %5 = mogg.invoke @input_0_fn_0(%arg2) {readOnly = true} : !mogg.indices -> !mogg.simd<si32>
      %6 = mogg.call "mo.cast"(%5) : !mogg.simd<si32> -> !mogg.simd<f32>
      %7 = mogg.invoke @input_1_fn(%arg2) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
      %8 = mogg.call "mo.add"(%6, %7) : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      mogg.invoke @output_0_fn(%arg2, %8) : !mogg.indices, !mogg.simd<f32>
      mogg.output
    }
    mogg.call "elementwise" [@elemwise_compute_lambda](%4) {single_thread_blocking_override = false} : !mo.tensor<[3, 3], f32>
    mogg.output %4 : !mo.tensor<[3, 3], f32>
  } {fusionBodyLambda = "elemwise_compute_lambda", inputFusionInterface = ["input_1_fn", "input_0_fn_0"], outputFusionInterface = ["output_0_fn"]}
  mo.output %3 : !mo.tensor<[3, 3], f32>
}

// CHECK-LABEL: @checkNoFuseDifferentOutDTypes
// CHECK: %[[OUT:.*]] = mogg.kernel
// CHECK: %[[OUT2:.*]] = mogg.kernel (%[[CONSTANT:.*]], %[[OUT]])
// CHECK: mo.output %[[OUT2]]
