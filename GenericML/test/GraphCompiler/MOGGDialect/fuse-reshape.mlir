// REQUIRES: disabled
// TODO(GEX-1573): Delete or convert to the new API
// RUN: mo-opt --mo-to-mogg --canonicalize --fuse-views --cse %s | FileCheck %s

mo.graph @basicReshapeFuse(%arg0 : !mo.tensor<[50], ui32>, %arg1 : !mo.tensor<[5, 10], ui32>) -> (!mo.tensor<[5, 5, 2], ui32>) {
  %A = mo.static.reshape(%arg0) : !mo.tensor<[50], ui32> -> !mo.tensor<[5, 5, 2], ui32>
  %B = mo.static.reshape(%arg1) : !mo.tensor<[5, 10], ui32> -> !mo.tensor<[5, 5, 2], ui32>
  %res = mo.add(%A, %B) : !mo.tensor<[5, 5, 2], ui32>
  mo.output %res : !mo.tensor<[5, 5, 2], ui32>
}
// CHECK-LABEL: mo.graph @basicReshapeFuse
// CHECK: (%[[ARG0:.*]]: !mo.tensor<[50], ui32>, %[[ARG1:.*]]: !mo.tensor<[5, 10], ui32>)
// CHECK: mogg.kernel (%[[ARG1]], %[[ARG0]]) : (%[[IN0:.*]]: !mo.tensor<[5, 10], ui32>, %[[IN1:.*]]: !mo.tensor<[50], ui32>)

// CHECK: %[[D0:.*]] = kgen.param.constant: scalar<si64> = <5>
// CHECK: %[[D1:.*]] = kgen.param.constant: scalar<si64> = <2>
// CHECK: %[[ARR:.*]] = pop.array.create [%[[D0]], %[[D0]], %[[D1]]] : !pop.array<3, scalar<si64>>
// CHECK: %[[SHAPE:.*]] = "mogg.to_mogg"(%[[ARR]])

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

// CHECK: %[[D0:.*]] = mogg.get_dim <N> -> !pop.scalar<si64>
// CHECK: %[[D1:.*]] = kgen.param.constant: scalar<si64> = <5>
// CHECK: %[[D2:.*]] = mogg.get_dim <L> -> !pop.scalar<si64>

// CHECK: %[[ARR:.*]] = pop.array.create [%[[D0]], %[[D1]], %[[D2]]] : !pop.array<3, scalar<si64>>
// CHECK: %[[SHAPE:.*]] = "mogg.to_mogg"(%[[ARR]])

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

// CHECK: %[[D0:.*]] = mogg.get_dim <N> -> !pop.scalar<si64>
// CHECK: %[[D1:.*]] = kgen.param.constant: scalar<si64> = <5>
// CHECK: %[[D2:.*]] = mogg.get_dim <L> -> !pop.scalar<si64>

// CHECK: %[[ARR:.*]] = pop.array.create [%[[D0]], %[[D1]], %[[D2]]] : !pop.array<3, scalar<si64>>
// CHECK: %[[SHAPE:.*]] = "mogg.to_mogg"(%[[ARR]])

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

// Example with multiple view operations fused into a mutable slice
mo.graph @in_place_slice<d0, d1, d2>(%arg0: !mo.buffer<[3, 3, 3], f32>, %arg1: !mo.tensor<[27], f32>, %arg2: !mo.tensor<[d0, d1, d2], f32>, %arg3: !mo.tensor<[3], si64>, %arg4: !mo.tensor<[1], si32>, %arg5: !mo.tensor<[2], si32>, %arg6: !mo.tensor<[3], si32>) -> !mo.buffer<[3, 3, 3], f32> {
  %0 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %1 = mo.constant {value = #M.dense_array<3, 3, 3> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %2 = mo.constant {value = #M.dense_array<0, 0, 0> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %3 = mo.chain.create(%arg0) : !mo.buffer<[3, 3, 3], f32>

  %outChain, %outTensor = mo.mutable.load(%3, %arg0) : (!mo.chain, !mo.buffer<[3, 3, 3], f32>) -> (!mo.chain, !mo.tensor<[3, 3, 3], f32>)
  %4 = mo.negative(%outTensor) : !mo.tensor<[3, 3, 3], f32>
  %5 = mo.static.reshape(%4): !mo.tensor<[3, 3, 3], f32> -> !mo.tensor<[27], f32>
  %10 = mo.add(%5, %arg1) : !mo.tensor<[27], f32>
  %11 = mo.static.reshape(%10) : !mo.tensor<[27], f32> -> !mo.tensor<[3, 3, 3], f32>
  %13 = mo.mutable.store.slice(%outChain, %arg0, %11, %2, %1, %arg3) : (!mo.chain, !mo.buffer<[3, 3, 3], f32>, !mo.tensor<[3, 3, 3], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.chain

  mo.output %arg0 : !mo.buffer<[3, 3, 3], f32>
}
// Just make sure this doesn't crash/violate invariants around the chain appearing last in mogg.kernels
// CHECK-LABEL: @in_place_slice
// CHECK-NOT: mogg.kernel{{.*!mo.chain,}}
