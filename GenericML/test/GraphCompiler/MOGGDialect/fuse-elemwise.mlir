// RUN: mo-opt --mo-to-mogg=use-jit=true --fuse-elementwise --canonicalize --cse %s | FileCheck %s


mo.graph @simpleDoubleUse(%arg0: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 3], f32>) {
  %0 = mo.sqrt(%arg0) : !mo.tensor<[4, 3], f32>
  %1 = mo.add(%0, %0) : !mo.tensor<[4, 3], f32>
  mo.output %1 : !mo.tensor<[4, 3], f32>
}

// CHECK-LABEL: mo.graph @simpleDoubleUse
// CHECK: %[[KERNEL_OUT:.*]] = mogg.kernel (%[[ARG1:.*]])

// Find the input lambda name.
// CHECK: mogg.lambda @[[IN_NAME:.*]] = (%[[INDICES:.*]]: !mogg.indices) -> !mogg.simd<f32>{

// Check the main body.
// CHECK: mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices){
// CHECK-NEXT:    %[[LOAD:.*]] = mogg.invoke @[[IN_NAME]](%[[INDICES]]) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SQRT:.*]] = mogg.call "mo.sqrt"(%[[LOAD]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[ADD:.*]] = mogg.call "mo.add"(%[[SQRT]], %[[SQRT]]) : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    mogg.invoke @output_0_fn(%[[INDICES]], %[[ADD]]) : !mogg.indices, !mogg.simd<f32>


mo.graph @rightmostTreeFusion(%arg0: !mo.tensor<[4, 3], f32>, %arg1: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 3], f32>) {
  %0 = mo.sqrt(%arg0) : !mo.tensor<[4, 3], f32>
  %1 = mo.sqrt(%0) : !mo.tensor<[4, 3], f32>
  %2 = mo.sqrt(%1) : !mo.tensor<[4, 3], f32>
  %3 = mo.sqrt(%2) : !mo.tensor<[4, 3], f32>
  %4 = mo.sub(%arg1, %3) : !mo.tensor<[4, 3], f32>
  %5 = mo.add(%arg1, %4) : !mo.tensor<[4, 3], f32>
  mo.output %5 : !mo.tensor<[4, 3], f32>
}

// CHECK-LABEL: mo.graph @rightmostTreeFusion
// CHECK: %[[KERNEL_OUT:.*]] = mogg.kernel (%[[ARG1:.*]], %[[ARG2:.*]])
// CHECK: mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices){
// CHECK-NEXT:    %[[LOAD1:.*]] = mogg.invoke @[[ANY_NAME:.*]](%[[INDICES]]) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
// CHECK-NEXT:    %[[LOAD2:.*]] = mogg.invoke @[[ANY_NAME:.*]](%[[INDICES]]) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SQRT1:.*]] = mogg.call "mo.sqrt"(%[[LOAD2]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SQRT2:.*]] =  mogg.call "mo.sqrt"(%[[SQRT1]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SQRT3:.*]] =  mogg.call "mo.sqrt"(%[[SQRT2]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SQRT4:.*]] =  mogg.call "mo.sqrt"(%[[SQRT3]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SUB:.*]] = mogg.call "mo.sub"(%[[LOAD1]], %[[SQRT4]]) : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[ADD:.*]] = mogg.call "mo.add"(%[[LOAD1]], %[[SUB]]) : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    mogg.invoke @[[ANY_NAME:.*]](%[[INDICES]], %[[ADD]]) : !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:    mogg.output
// CHECK-NEXT:}
// CHECK-NEXT: mogg.call "elementwise" [@elemwise_compute_lambda]
// CHECK-NEXT: mogg.output
// CHECK-NEXT:} {fusionBodyLambda = "elemwise_compute_lambda", inputFusionInterface = ["input_0_fn", "input_0_fn_1"], outputFusionInterface = ["output_0_fn"]}
// CHECK:  mo.output %[[KERNEL_OUT]]


mo.graph @leftmostTreeFusion(%arg0: !mo.tensor<[4, 3], f32>, %arg1: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 3], f32>) {
  %0 = mo.sqrt(%arg0) : !mo.tensor<[4, 3], f32>
  %1 = mo.sqrt(%0) : !mo.tensor<[4, 3], f32>
  %2 = mo.sqrt(%1) : !mo.tensor<[4, 3], f32>
  %3 = mo.sqrt(%2) : !mo.tensor<[4, 3], f32>
  %4 = mo.sub(%3, %arg1) : !mo.tensor<[4, 3], f32>
  %5 = mo.add(%4, %arg1) : !mo.tensor<[4, 3], f32>
  mo.output %5 : !mo.tensor<[4, 3], f32>
}

// CHECK-LABEL: mo.graph @leftmostTreeFusion
// CHECK: %[[KERNEL_OUT:.*]] = mogg.kernel (%[[ARG1:.*]], %[[ARG2:.*]])
// CHECK: mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices){
// CHECK-NEXT:    %[[LOAD1:.*]] = mogg.invoke @[[ANY_NAME:.*]](%[[INDICES]]) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SQRT1:.*]] = mogg.call "mo.sqrt"(%[[LOAD1]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SQRT2:.*]] = mogg.call "mo.sqrt"(%[[SQRT1]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SQRT3:.*]] = mogg.call "mo.sqrt"(%[[SQRT2]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SQRT4:.*]] = mogg.call "mo.sqrt"(%[[SQRT3]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[LOAD2:.*]] = mogg.invoke @[[ANY_NAME:.*]](%[[INDICES]]) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SUB:.*]] = mogg.call "mo.sub"(%[[SQRT4]], %[[LOAD2]]) : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[ADD:.*]] = mogg.call "mo.add"(%[[SUB]], %[[LOAD2]]) : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    mogg.invoke @[[ANY_NAME:.*]](%[[INDICES]], %[[ADD]]) : !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:    mogg.output
// CHECK-NEXT:}
// CHECK-NEXT: mogg.call "elementwise" [@elemwise_compute_lambda]
// CHECK-NEXT: mogg.output
// CHECK-NEXT:} {fusionBodyLambda = "elemwise_compute_lambda", inputFusionInterface = ["input_1_fn", "input_0_fn_0"], outputFusionInterface = ["output_0_fn"]}
// CHECK:  mo.output %[[KERNEL_OUT]]


mo.graph @multipleDiamondPattern(%arg0: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 3], f32>) {
  %0 = mo.sqrt(%arg0) : !mo.tensor<[4, 3], f32>
  %1 = mo.rsqrt(%0) : !mo.tensor<[4, 3], f32>
  %2 = mo.tanh(%0) : !mo.tensor<[4, 3], f32>
  %3 = mo.sub(%1, %2) : !mo.tensor<[4, 3], f32>
  %4 = mo.sqrt(%3) : !mo.tensor<[4, 3], f32>
  %5 = mo.rsqrt(%3) : !mo.tensor<[4, 3], f32>
  %6 = mo.add(%4, %5) : !mo.tensor<[4, 3], f32>
  %7 = mo.sqrt(%6) : !mo.tensor<[4, 3], f32>
  %8 = mo.rsqrt(%6) : !mo.tensor<[4, 3], f32>
  %9 = mo.mul(%7, %8) : !mo.tensor<[4, 3], f32>
  %10 = mo.sqrt(%9) : !mo.tensor<[4, 3], f32>
  %11 = mo.rsqrt(%9) : !mo.tensor<[4, 3], f32>
  %12 = mo.mul(%10, %11) : !mo.tensor<[4, 3], f32>
  mo.output %12 : !mo.tensor<[4, 3], f32>
}

// CHECK: mo.graph @multipleDiamondPattern
// CHECK-NEXT: mogg.kernel (%[[NULL:.*]]) : (%[[ARG1:.*]]: !mo.tensor<[4, 3], f32>) -> !mo.tensor<[4, 3], f32>
// CHECK: mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices){
// CHECK-NEXT:    %[[IN:.*]] = mogg.invoke @[[INPUT_LAMBDA:.*]](%[[INDICES]]) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SQRT:.*]] = mogg.call "mo.sqrt"(%[[IN]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[RSQRT:.*]] = mogg.call "mo.rsqrt"(%[[SQRT]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[TANH:.*]] = mogg.call "mo.tanh"(%[[SQRT]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SUB:.*]] = mogg.call "mo.sub"(%[[RSQRT]], %[[TANH]]) : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SQRT2:.*]] = mogg.call "mo.sqrt"(%[[SUB]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[RSQRT2:.*]] = mogg.call "mo.rsqrt"(%[[SUB]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[ADD:.*]] = mogg.call "mo.add"(%[[SQRT2]], %[[RSQRT2]]) : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SQRT3:.*]] = mogg.call "mo.sqrt"(%[[ADD]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[RSQRT3:.*]] = mogg.call "mo.rsqrt"(%[[ADD]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[MUL:.*]] = mogg.call "mo.mul"(%[[SQRT3]], %[[RSQRT3]]) : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SQRT4:.*]] = mogg.call "mo.sqrt"(%[[MUL]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[RSQRT4:.*]] = mogg.call "mo.rsqrt"(%[[MUL]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[MUL:.*]] = mogg.call "mo.mul"(%[[SQRT4]], %[[RSQRT4]]) : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    mogg.invoke @[[OUTPUT_LAMBDA:.*]](%[[INDICES]], %[[MUL]]) : !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:    mogg.output


// Check we correctly fuse a top heavy graph with a wide input surface and still fuse into one at the end.
mo.graph @inversePyramid(%arg0: !mo.tensor<[4, 3], f32>, %arg1: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 3], f32>) {
  // Top level ops.
  %0 = mo.sqrt(%arg0) : !mo.tensor<[4, 3], f32>
  %1 = mo.relu(%arg0) : !mo.tensor<[4, 3], f32>
  %2 = mo.sqrt(%arg1) : !mo.tensor<[4, 3], f32>
  %3 = mo.relu(%arg1) : !mo.tensor<[4, 3], f32>
  %4 = mo.add(%arg0, %arg1) : !mo.tensor<[4, 3], f32>
  %5 = mo.mul(%arg0, %arg1) : !mo.tensor<[4, 3], f32>
  %6 = mo.sub(%arg0, %arg1) : !mo.tensor<[4, 3], f32>
  %7 = mo.div(%arg0, %arg1) : !mo.tensor<[4, 3], f32>
  %8 = mo.tanh(%arg0) : !mo.tensor<[4, 3], f32>

  // Start reducing down into a pyramid
  %9 = mo.add(%0, %1) : !mo.tensor<[4, 3], f32>
  %10 = mo.add(%2, %3) : !mo.tensor<[4, 3], f32>
  %11 = mo.add(%5, %6) : !mo.tensor<[4, 3], f32>
  %12 = mo.add(%7, %8) : !mo.tensor<[4, 3], f32>

  %13 = mo.add(%9, %10) : !mo.tensor<[4, 3], f32>
  %14 = mo.add(%11, %12) : !mo.tensor<[4, 3], f32>

  %out = mo.mul(%13, %14) : !mo.tensor<[4, 3], f32>
  mo.output %out : !mo.tensor<[4, 3], f32>
}


// CHECK: mo.graph @inversePyramid
// CHECK-NEXT: mogg.kernel (%[[NULL:.*]]) : (%[[ARG1:.*]]: !mo.tensor<[4, 3], f32>) -> !mo.tensor<[4, 3], f32>
// CHECK: mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices){
// CHECK-NEXT:    %[[IN1:.*]] = mogg.invoke @[[INPUT_LAMBDA:.*]](%[[INDICES]])
// CHECK-NEXT:    %[[OP0:.*]] = mogg.call "mo.sqrt"(%[[IN1]])
// CHECK-NEXT:    %[[OP1:.*]] = mogg.call "mo.relu"(%[[IN1]])
// CHECK-NEXT:    %[[OP2:.*]] = mogg.call "mo.add"(%[[OP0]], %[[OP1]])
// CHECK-NEXT:    %[[IN2:.*]] = mogg.invoke @[[INPUT_LAMBDA2:.*]](%[[INDICES]])
// CHECK-NEXT:    %[[OP3:.*]] = mogg.call "mo.sqrt"(%[[IN2]])
// CHECK-NEXT:    %[[OP4:.*]] = mogg.call "mo.relu"(%[[IN2]])
// CHECK-NEXT:    %[[OP5:.*]] = mogg.call "mo.add"(%[[OP3]], %[[OP4]])
// CHECK-NEXT:    %[[OP6:.*]] = mogg.call "mo.add"(%[[OP2]], %[[OP5]])
// CHECK-NEXT:    %[[OP7:.*]] = mogg.call "mo.mul"(%[[IN1]], %[[IN2]])
// CHECK-NEXT:    %[[OP8:.*]] = mogg.call "mo.sub"(%[[IN1]], %[[IN2]])
// CHECK-NEXT:    %[[OP9:.*]] = mogg.call "mo.add"(%[[OP7]], %[[OP8]])
// CHECK-NEXT:    %[[OP10:.*]] = mogg.call "mo.div"(%[[IN1]], %[[IN2]])
// CHECK-NEXT:    %[[OP11:.*]] = mogg.call "mo.tanh"(%[[IN1]])
// CHECK-NEXT:    %[[OP12:.*]] = mogg.call "mo.add"(%[[OP10]], %[[OP11]])
// CHECK-NEXT:    %[[OP13:.*]] = mogg.call "mo.add"(%[[OP9]], %[[OP12]])
// CHECK-NEXT:    %[[FINAL:.*]] = mogg.call "mo.mul"(%[[OP6]], %[[OP13]])
// CHECK-NEXT:    mogg.invoke @[[OUTPUT_LAMBDA:.*]](%[[INDICES]], %[[FINAL]])
// CHECK-NEXT:    mogg.output


// Broadcast (currently) has its own fusion pass so should not be fused along with the elementwise ops in the fuse-elementwise pass.

mo.graph @broadcastNoFuse<N, M, K>(%arg0: !mo.tensor<[N, K], f32>, %arg1: !mo.tensor<[K, M], f32>) -> (!mo.tensor<[N, M], f32>) {
  %bcast1 = mo.static.broadcast_to(%arg0) : !mo.tensor<[N, K], f32> -> !mo.tensor<[N, M], f32>
  %bcast2 = mo.static.broadcast_to(%arg1) : !mo.tensor<[K, M], f32> -> !mo.tensor<[N, M], f32>
  %1 = mo.add(%bcast1, %bcast2) : !mo.tensor<[N, M], f32>
  mo.output %1 : !mo.tensor<[N, M], f32>
}

// Check each broadcast is in its own standalone kernel.
// CHECK-LABEL: mo.graph @broadcastNoFuse
// CHECK-NEXT: %[[BCAST1:.*]] = mogg.kernel
// CHECK: %[[BCAST2:.*]] = mogg.kernel
// CHECK: %[[ADD:.*]] = mogg.kernel (%[[BCAST1]], %[[BCAST2]])
// CHECK: mo.output %[[ADD]] : !mo.tensor<[N, M], f32>

mo.graph @fused_si32(%arg0: !mo.tensor<[6], si32>, %arg1: !mo.tensor<[6], si32>) -> (!mo.tensor<[6], si32>) {
  %0 = mo.add(%arg0, %arg1) : !mo.tensor<[6], si32>
  %1 = mo.relu(%0) : !mo.tensor<[6], si32>
  mo.output %1 : !mo.tensor<[6], si32>
}

// CHECK-LABEL: mo.graph @fused_si32

// CHECK: mogg.lambda @elemwise_compute_lambda
// CHECK-NEXT: %[[LOAD1:.*]] = mogg.invoke @input_0_fn_0
// CHECK-NEXT: %[[LOAD2:.*]] = mogg.invoke @input_1_fn
// CHECK-NEXT: %[[ADD:.*]] = mogg.call "mo.add"(%[[LOAD1]], %[[LOAD2]]) : !mogg.simd<si32>, !mogg.simd<si32> -> !mogg.simd<si32>
// CHECK-NEXT: %[[RELU:.*]] = mogg.call "mo.relu"(%[[ADD]]) : !mogg.simd<si32> -> !mogg.simd<si32>


mo.graph @check_no_fuse_if<N, M>(%cond : !mo.tensor<[], bool>, %arg0: !mo.tensor<[N, M], f32>) -> (!mo.tensor<[N, M], f32>) {
  %relu = mo.relu(%arg0) : !mo.tensor<[N, M], f32>

  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[N, M], f32> {
    %1 = mo.add(%relu, %relu) : !mo.tensor<[N, M], f32>
    mo.yield %1 : !mo.tensor<[N, M], f32>
  } else {
    %2 = mo.mul(%relu, %relu) : !mo.tensor<[N, M], f32>
    mo.yield %2 : !mo.tensor<[N, M], f32>
  }
  mo.output %0 : !mo.tensor<[N, M], f32>
}

// Check all kernels are compiled but unfused.

// CHECK-LABEL: mo.graph @check_no_fuse_if

// CHECK-NEXT: %[[BCAST1:.*]] = mogg.kernel
// CHECK-NEXT: mogg.output_placeholders
// CHECK: mogg.call "mo.relu"

// CHECK: mo.if
// CHECK-NEXT: mogg.kernel
// CHECK: mogg.call "mo.add"
// CHECK: mo.yield
// CHECK-NEXT: } else {
// CHECK-NEXT: mogg.kernel
// CHECK: mogg.call "mo.mul"
// CHECK: mo.yield

// Here fusion may greedily fuse compute lambda, making some values out of
// order of the definitions. Test we topographically sort the final compute
// lambda expressions.
mo.graph @out_of_order() -> !mo.tensor<[5], f32> {
  %0 = mo.constant {value = #M.dense_array<-1.000000e+00, 2.000000e+00, -3.000000e+00, -4.000000e+00, 5.000000e+00> : tensor<5xf32>} : !mo.tensor<[5], f32>
  %1 = mo.constant {value = #M.dense_array<3.000000e+00, 1.000000e+00, -6.000000e+00, 2.000000e+00, 5.000000e+00> : tensor<5xf32>} : !mo.tensor<[5], f32>
  %2 = mo.add(%0, %1) : !mo.tensor<[5], f32>
  %3 = mo.mul(%0, %2) : !mo.tensor<[5], f32>
  %4 = mo.sub(%0, %2) : !mo.tensor<[5], f32>
  %5 = mo.relu(%4) : !mo.tensor<[5], f32>
  %6 = mo.add(%5, %2) : !mo.tensor<[5], f32>
  mo.output %6 : !mo.tensor<[5], f32>
}

// CHECK-LABEL: mo.graph @out_of_order
// CHECK: mogg.lambda @elemwise_compute_lambda
// CHECK-NEXT: %[[V0:.*]] = mogg.invoke
// CHECK-NEXT: %[[V1:.*]] = mogg.invoke
// CHECK-NEXT: %[[V2:.*]] = mogg.call "mo.add"
// CHECK-NEXT: %[[V3:.*]] = mogg.call "mo.sub"
// CHECK-NEXT: %[[V4:.*]] = mogg.call "mo.relu"
// CHECK-NEXT: %[[V5:.*]] = mogg.call "mo.add"
