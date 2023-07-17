// RUN: mo-opt --mo-to-mogg=use-jit=true --fuse-elementwise --cse --fuse-bcasts --cse --fuse-in-out-lambdas --cse %s | FileCheck %s

mo.graph @basicMatmul<N, M, K>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32>, %arg3: !mo.tensor<[], f32>) -> (!mo.tensor<[M, N], f32>) {
  %mm = mo.matmul(%arg0, %arg1) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32>)  -> !mo.tensor<[M, N], f32>
  %bcast = mo.static.broadcast_to(%arg3) : !mo.tensor<[], f32> -> !mo.tensor<[M, N], f32>
  %1 = mo.add(%mm, %bcast) : !mo.tensor<[M, N], f32>
  mo.output %1 : !mo.tensor<[M, N], f32>
}

// CHECK-LABEL: mo.graph @basicMatmul
// CHECK: %[[OUT:.*]] = mogg.kernel
// CHECK-NEXT: mogg.output_placeholders
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"
// CHECK-NEXT: mogg.lambda @output_0_fn
// CHECK-NEXT: mogg.simd_load
// CHECK-NEXT: mogg.call "mo.add"
// CHECK-NEXT: mogg.call "simd_store"
// CHECK-NEXT: mogg.output
// CHECK-NEXT: }
// CHECK-NEXT: mogg.call "mo.matmul" [@output_0_fn{{.*}}{lambdas_have_fusion = true
// CHECK: mo.output %[[OUT]]

mo.graph @noFuseMultiUse<N, M, K>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32>, %arg3: !mo.tensor<[], f32>, %arg4: !mo.tensor<[], f32>) -> (!mo.tensor<[M, N], f32>, !mo.tensor<[M, N], f32>) {
  %mm = mo.matmul(%arg0, %arg1) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32>)  -> !mo.tensor<[M, N], f32>
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
