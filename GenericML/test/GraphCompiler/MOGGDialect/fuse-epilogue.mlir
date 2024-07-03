// RUN: mo-opt --mo-to-mogg --fuse-elementwise --cse --fuse-bcasts --cse --fuse-in-out-lambdas --cse %s | FileCheck %s

mo.graph @basicMatmul<N, M, K>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32, {layout = #mo.layout<KN>}>, %arg3: !mo.tensor<[], f32>) -> (!mo.tensor<[M, N], f32>) {
  %mm = mo.matmul(%arg0, %arg1) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32, {layout = #mo.layout<KN>}>)  -> !mo.tensor<[M, N], f32>
  %bcast = mo.static.broadcast_to(%arg3) : !mo.tensor<[], f32> -> !mo.tensor<[M, N], f32>
  %1 = mo.add(%mm, %bcast) : !mo.tensor<[M, N], f32>
  mo.output %1 : !mo.tensor<[M, N], f32>
}

// CHECK-LABEL: mo.graph @basicMatmul
// CHECK: %[[OUT:.*]] = mogg.kernel
// CHECK: mogg.output_placeholders
// CHECK-NEXT: mogg.param_shape_to_value
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"
// CHECK-NEXT: mogg.lambda @normal_op_output_0_fn
// CHECK-NEXT: mogg.call "simd_load"
// CHECK-NEXT: mogg.call "mo.add"
// CHECK-NEXT: mogg.call "simd_store"
// CHECK-NEXT: mogg.output
// CHECK-NEXT: }
// CHECK-NEXT: mogg.call "mo.matmul" [@normal_op_output_0_fn{{.*}}{lambdas_have_fusion = true
// CHECK: mo.output

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
        %3 = mogg.call "simd_load"(%arg2, %arg3) : !mo.tensor<[M, N], f32>, !mogg.indices -> !mogg.simd<f32>
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
        %3 = mogg.call "simd_load"(%arg2, %arg3) : !mo.tensor<[M, N], f32>, !mogg.indices -> !mogg.simd<f32>
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
        %3 = mogg.call "simd_load"(%arg2, %arg3) : !mo.tensor<[M, N], f32>, !mogg.indices -> !mogg.simd<f32>
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

mo.graph @mixed_precision(%arg0: !mo.tensor<[2, 2], ui32>, %arg1: !mo.tensor<[2, 2], ui32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[2, 2], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[2, 2], ui32>, %arg3: !mo.tensor<[2, 2], ui32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[2, 2], ui32>{
    %2 = mogg.output_placeholders : !mo.tensor<[2, 2], ui32>
    mogg.lambda @output_0_fn = (%arg4: !mogg.indices, %arg5: !mogg.simd<ui32>){
      mogg.call "simd_store"(%2, %arg4, %arg5) : !mo.tensor<[2, 2], ui32>, !mogg.indices, !mogg.simd<ui32>
      mogg.output
    }
    mogg.call "mo.matmul" [@output_0_fn](%arg2, %arg3, %2) {lambdas_have_fusion = false, packed_in_1 = false, single_thread_blocking_override = false, transpose_in_1 = false} : !mo.tensor<[2, 2], ui32>, !mo.tensor<[2, 2], ui32, {layout = #mo.layout<KN>}>, !mo.tensor<[2, 2], ui32>
    mogg.output %2 : !mo.tensor<[2, 2], ui32>
  } {outputFusionInterface = ["output_0_fn"]}
  %1 = mogg.kernel (%0) : (%arg2: !mo.tensor<[2, 2], ui32>) -> !mo.tensor<[2, 2], f32>{
    %2 = mogg.output_placeholders : !mo.tensor<[2, 2], f32>
    mogg.lambda @input_0_fn = (%arg3: !mogg.indices) -> !mogg.simd<ui32>{
      %3 = mogg.call "simd_load"(%arg2, %arg3) : !mo.tensor<[2, 2], ui32>, !mogg.indices -> !mogg.simd<ui32>
      mogg.output %3 : !mogg.simd<ui32>
    }
    mogg.lambda @output_0_fn = (%arg3: !mogg.indices, %arg4: !mogg.simd<f32>){
      mogg.call "simd_store"(%2, %arg3, %arg4) : !mo.tensor<[2, 2], f32>, !mogg.indices, !mogg.simd<f32>
      mogg.output
    }
    mogg.lambda @elemwise_compute_lambda = (%arg3: !mogg.indices){
      %3 = mogg.invoke @input_0_fn(%arg3) {readOnly = true} : !mogg.indices -> !mogg.simd<ui32>
      %4 = mogg.call "mo.cast"(%3) : !mogg.simd<ui32> -> !mogg.simd<f32>
      mogg.invoke @output_0_fn(%arg3, %4) : !mogg.indices, !mogg.simd<f32>
      mogg.output
    }
    mogg.call "elementwise" [@elemwise_compute_lambda](%2) {single_thread_blocking_override = false} : !mo.tensor<[2, 2], f32>
    mogg.output %2 : !mo.tensor<[2, 2], f32>
  } {fusionBodyLambda = "elemwise_compute_lambda", inputFusionInterface = ["input_0_fn"], outputFusionInterface = ["output_0_fn"]}
  mo.output %1 : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: @mixed_precision
// CHECK-NEXT: %[[OUT:.*]] = mogg.kernel
// CHECK-NEXT: %[[PLACEHOLDER:.*]] = mogg.output_placeholders : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], ui32>

mo.graph @fused_mixed_precision_cuda_device(%arg0: !mo.tensor<[2, 2], ui32>, %arg1: !mo.tensor<[2, 2], ui32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[2, 2], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[2, 2], ui32>, %arg3: !mo.tensor<[2, 2], ui32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[2, 2], ui32>{
    %2 = mogg.output_placeholders : !mo.tensor<[2, 2], ui32>
    mogg.lambda @output_0_fn = (%arg4: !mogg.indices, %arg5: !mogg.simd<ui32>){
      mogg.call "simd_store"(%2, %arg4, %arg5) : !mo.tensor<[2, 2], ui32>, !mogg.indices, !mogg.simd<ui32>
      mogg.output
    }
    mogg.call "mo.matmul" [@output_0_fn](%arg2, %arg3, %2) {lambdas_have_fusion = false, packed_in_1 = false, single_thread_blocking_override = false, transpose_in_1 = false} : !mo.tensor<[2, 2], ui32>, !mo.tensor<[2, 2], ui32, {layout = #mo.layout<KN>}>, !mo.tensor<[2, 2], ui32>
    mogg.output %2 : !mo.tensor<[2, 2], ui32>
  } {device = #M.device_ref<"cuda", 0>, outputFusionInterface = ["output_0_fn"]}
  %1 = mogg.kernel (%0) : (%arg2: !mo.tensor<[2, 2], ui32>) -> !mo.tensor<[2, 2], f32>{
    %2 = mogg.output_placeholders : !mo.tensor<[2, 2], f32>
    mogg.lambda @input_0_fn = (%arg3: !mogg.indices) -> !mogg.simd<ui32>{
      %3 = mogg.call "simd_load"(%arg2, %arg3) : !mo.tensor<[2, 2], ui32>, !mogg.indices -> !mogg.simd<ui32>
      mogg.output %3 : !mogg.simd<ui32>
    }
    mogg.lambda @output_0_fn = (%arg3: !mogg.indices, %arg4: !mogg.simd<f32>){
      mogg.call "simd_store"(%2, %arg3, %arg4) : !mo.tensor<[2, 2], f32>, !mogg.indices, !mogg.simd<f32>
      mogg.output
    }
    mogg.lambda @elemwise_compute_lambda = (%arg3: !mogg.indices){
      %3 = mogg.invoke @input_0_fn(%arg3) {readOnly = true} : !mogg.indices -> !mogg.simd<ui32>
      %4 = mogg.call "mo.cast"(%3) : !mogg.simd<ui32> -> !mogg.simd<f32>
      mogg.invoke @output_0_fn(%arg3, %4) : !mogg.indices, !mogg.simd<f32>
      mogg.output
    }
    mogg.call "elementwise" [@elemwise_compute_lambda](%2) {single_thread_blocking_override = false} : !mo.tensor<[2, 2], f32>
    mogg.output %2 : !mo.tensor<[2, 2], f32>
  } {device = #M.device_ref<"cuda", 0>, fusionBodyLambda = "elemwise_compute_lambda", inputFusionInterface = ["input_0_fn"], outputFusionInterface = ["output_0_fn"]}
  mo.output %1 : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: @fused_mixed_precision_cuda_device
// CHECK-NEXT: %[[OUT:.*]] = mogg.kernel
// CHECK-NEXT: %[[PLACEHOLDER:.*]] = mogg.output_placeholders : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], ui32>
// CHECK: device = #M.device_ref<"cuda", 0>

mo.graph @nofuse_mixed_precision_mixed_device(%arg0: !mo.tensor<[2, 2], ui32>, %arg1: !mo.tensor<[2, 2], ui32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[2, 2], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[2, 2], ui32>, %arg3: !mo.tensor<[2, 2], ui32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[2, 2], ui32>{
    %2 = mogg.output_placeholders : !mo.tensor<[2, 2], ui32>
    mogg.lambda @output_0_fn = (%arg4: !mogg.indices, %arg5: !mogg.simd<ui32>){
      mogg.call "simd_store"(%2, %arg4, %arg5) : !mo.tensor<[2, 2], ui32>, !mogg.indices, !mogg.simd<ui32>
      mogg.output
    }
    mogg.call "mo.matmul" [@output_0_fn](%arg2, %arg3, %2) {lambdas_have_fusion = false, packed_in_1 = false, single_thread_blocking_override = false, transpose_in_1 = false} : !mo.tensor<[2, 2], ui32>, !mo.tensor<[2, 2], ui32, {layout = #mo.layout<KN>}>, !mo.tensor<[2, 2], ui32>
    mogg.output %2 : !mo.tensor<[2, 2], ui32>
  } {outputFusionInterface = ["output_0_fn"]}
  %1 = mogg.kernel (%0) : (%arg2: !mo.tensor<[2, 2], ui32>) -> !mo.tensor<[2, 2], f32>{
    %2 = mogg.output_placeholders : !mo.tensor<[2, 2], f32>
    mogg.lambda @input_0_fn = (%arg3: !mogg.indices) -> !mogg.simd<ui32>{
      %3 = mogg.call "simd_load"(%arg2, %arg3) : !mo.tensor<[2, 2], ui32>, !mogg.indices -> !mogg.simd<ui32>
      mogg.output %3 : !mogg.simd<ui32>
    }
    mogg.lambda @output_0_fn = (%arg3: !mogg.indices, %arg4: !mogg.simd<f32>){
      mogg.call "simd_store"(%2, %arg3, %arg4) : !mo.tensor<[2, 2], f32>, !mogg.indices, !mogg.simd<f32>
      mogg.output
    }
    mogg.lambda @elemwise_compute_lambda = (%arg3: !mogg.indices){
      %3 = mogg.invoke @input_0_fn(%arg3) {readOnly = true} : !mogg.indices -> !mogg.simd<ui32>
      %4 = mogg.call "mo.cast"(%3) : !mogg.simd<ui32> -> !mogg.simd<f32>
      mogg.invoke @output_0_fn(%arg3, %4) : !mogg.indices, !mogg.simd<f32>
      mogg.output
    }
    mogg.call "elementwise" [@elemwise_compute_lambda](%2) {single_thread_blocking_override = false} : !mo.tensor<[2, 2], f32>
    mogg.output %2 : !mo.tensor<[2, 2], f32>
  } {device = #M.device_ref<"cuda", 0>, fusionBodyLambda = "elemwise_compute_lambda", inputFusionInterface = ["input_0_fn"], outputFusionInterface = ["output_0_fn"]}
  mo.output %1 : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: @nofuse_mixed_precision_mixed_device
// CHECK: %[[OUT:.*]] = mogg.kernel
// CHECK: %[[OUT2:.*]] = mogg.kernel (%[[OUT]])
// CHECK: device = #M.device_ref<"cuda", 0>
