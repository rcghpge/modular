// RUN: mo-opt --mo-to-mogg --fuse-bcasts --cse %s | FileCheck %s

mo.graph @basicReshapeFuse(%arg0 : !mo.tensor<[50], ui32>, %arg1 : !mo.tensor<[5, 10], ui32>) -> (!mo.tensor<[5, 5, 2], ui32>) {
  %A = mo.static.reshape(%arg0) : !mo.tensor<[50], ui32> -> !mo.tensor<[5, 5, 2], ui32>
  %B = mo.static.reshape(%arg1) : !mo.tensor<[5, 10], ui32> -> !mo.tensor<[5, 5, 2], ui32>
  %res = mo.add(%A, %B) : !mo.tensor<[5, 5, 2], ui32>
  mo.output %res : !mo.tensor<[5, 5, 2], ui32>
}
// CHECK-LABEL: mo.graph @basicReshapeFuse
// CHECK: (%[[ARG0:.*]]: !mo.tensor<[50], ui32>, %[[ARG1:.*]]: !mo.tensor<[5, 10], ui32>)
// CHECK: mogg.kernel (%[[ARG1]], %[[ARG0]]) : (%[[IN0:.*]]: !mo.tensor<[5, 10], ui32>, %[[IN1:.*]]: !mo.tensor<[50], ui32>)

// CHECK: %[[SHAPE:.*]] = mogg.param_shape_to_value -> !mogg.shape<[5, 5, 2]>
// CHECK-NEXT: %[[NEW_0:.*]] = mogg.call "ndbuffer_reshape"(%[[IN1]], %[[SHAPE]])

// CHECK-NEXT: %[[NEW_1:.*]] = mogg.call "ndbuffer_reshape"(%[[IN0]], %[[SHAPE]])


// CHECK: mogg.call "simd_load"(%[[NEW_0]]
// CHECK: mogg.call "simd_load"(%[[NEW_1]]


mo.graph @basicReshapeDyn<N, L>(%arg0 : !mo.tensor<[50], ui32>, %arg1 : !mo.tensor<[5, 10], ui32>) -> (!mo.tensor<[N, 5, L], ui32>) {
  %A = mo.static.reshape(%arg0) : !mo.tensor<[50], ui32> -> !mo.tensor<[N, 5, L], ui32>
  %B = mo.static.reshape(%arg1) : !mo.tensor<[5, 10], ui32> -> !mo.tensor<[N, 5, L], ui32>
  %res = mo.add(%A, %B) : !mo.tensor<[N, 5, L], ui32>
  mo.output %res : !mo.tensor<[N, 5, L], ui32>
}
// CHECK-LABEL: mo.graph @basicReshapeDyn
// CHECK: (%[[ARG0:.*]]: !mo.tensor<[50], ui32>, %[[ARG1:.*]]: !mo.tensor<[5, 10], ui32>)
// CHECK: mogg.kernel (%[[ARG1]], %[[ARG0]]) : (%[[IN0:.*]]: !mo.tensor<[5, 10], ui32>, %[[IN1:.*]]: !mo.tensor<[50], ui32>)

// CHECK: %[[SHAPE:.*]] = mogg.param_shape_to_value -> !mogg.shape<[N, 5, L]>
// CHECK-NEXT: %[[NEW_0:.*]] = mogg.call "ndbuffer_reshape"(%[[IN1]], %[[SHAPE]])

// CHECK-NEXT: %[[NEW_1:.*]] = mogg.call "ndbuffer_reshape"(%[[IN0]], %[[SHAPE]])

// CHECK: mogg.call "simd_load"(%[[NEW_0]]
// CHECK: mogg.call "simd_load"(%[[NEW_1]]



mo.graph @multiReshape<N, L>(%arg0: !mo.tensor<[50], ui32>, %arg1: !mo.tensor<[5, 10], ui32>) -> !mo.tensor<[N, 5, L], ui32> {
    %0 = mo.static.reshape(%arg0) : !mo.tensor<[50], ui32> -> !mo.tensor<[N, 5, L], ui32>

    %3 = mogg.kernel (%0, %0, %0) : (%arg2: !mo.tensor<[N, 5, L], ui32>, %arg3: !mo.tensor<[N, 5, L], ui32>, %arg4: !mo.tensor<[N, 5, L], ui32>) -> !mo.tensor<[N, 5, L], ui32>{
        %3 = mogg.output_placeholders : !mo.tensor<[N, 5, L], ui32>

        mogg.lambda @elemwise_compute_lambda = (%indices: !mogg.indices){
            %4 = mogg.call "simd_load"(%arg2, %indices) : !mo.tensor<[N, 5, L], ui32>, !mogg.indices -> !mogg.simd<ui32>
            %5 = mogg.call "simd_load"(%arg3, %indices) : !mo.tensor<[N, 5, L], ui32>, !mogg.indices -> !mogg.simd<ui32>
            %6 = mogg.call "simd_load"(%arg4, %indices) : !mo.tensor<[N, 5, L], ui32>, !mogg.indices -> !mogg.simd<ui32>
            mogg.output
        }
        mogg.output %3 : !mo.tensor<[N, 5, L], ui32>
    }
    mo.output %3 : !mo.tensor<[N, 5, L], ui32>
}

// CHECK-LABEL: mo.graph @multiReshape
// CHECK: (%[[ARG0:.*]]: !mo.tensor<[50], ui32>, %[[ARG1:.*]]: !mo.tensor<[5, 10], ui32>)
// CHECK: mogg.kernel (%[[ARG0]]) : (%[[IN0:.*]]: !mo.tensor<[50], ui32>)

// CHECK: %[[SHAPE:.*]] = mogg.param_shape_to_value -> !mogg.shape<[N, 5, L]>
// CHECK: %[[RESHAPED:.*]] = mogg.call "ndbuffer_reshape"(%[[IN0]], %[[SHAPE]])

// CHECK: mogg.call "simd_load"(%[[RESHAPED]]
// CHECK: mogg.call "simd_load"(%[[RESHAPED]]
// CHECK: mogg.call "simd_load"(%[[RESHAPED]]


mo.graph @dontFuseCompound<N, M>(%arg0 : !mo.tensor<[M], ui32>, %arg1 : !mo.tensor<[N], ui32>) -> (!mo.tensor<[N], ui32>) {
  %1 = mo.rebind(%arg0) : !mo.tensor<[M], ui32> -> !mo.tensor<[add(N, 2)], ui32>
  %A = mo.static.reshape(%1) : !mo.tensor<[add(N, 2)], ui32> -> !mo.tensor<[N], ui32>
  %res = mo.add(%A, %arg1) : !mo.tensor<[N], ui32>
  mo.output %res : !mo.tensor<[N], ui32>
}

// CHECK-LABEL: mo.graph @dontFuseCompound<N, M>
// CHECK-NEXT: [[REBIND:%.*]] = mo.rebind
// CHECK-NEXT: mogg.kernel ([[REBIND]], %arg1) : (%{{.*}}: !mo.tensor<[N], ui32>, %{{.*}}: !mo.tensor<[N], ui32>) -> !mo.tensor<[N], ui32>{
