// RUN: mo-opt --mo-to-mogg --canonicalize --fuse-views --cse %s | FileCheck %s

mo.graph @basicReshapeFuse(%arg0 : !mo.tensor<[50], ui32>, %arg1 : !mo.tensor<[5, 10], ui32>) -> (!mo.tensor<[5, 5, 2], ui32>) {
  %A = mo.static.reshape(%arg0) : !mo.tensor<[50], ui32> -> !mo.tensor<[5, 5, 2], ui32>
  %B = mo.static.reshape(%arg1) : !mo.tensor<[5, 10], ui32> -> !mo.tensor<[5, 5, 2], ui32>
  %res = mo.add(%A, %B) : !mo.tensor<[5, 5, 2], ui32>
  mo.output %res : !mo.tensor<[5, 5, 2], ui32>
}
// CHECK-LABEL: mo.graph @basicReshapeFuse
// CHECK: (%[[ARG0:.*]]: !mo.tensor<[50], ui32>, %[[ARG1:.*]]: !mo.tensor<[5, 10], ui32>)
// CHECK: mogg.kernel(%[[ARG0]]: !mo.tensor<[50], ui32>, %[[ARG1]]: !mo.tensor<[5, 10], ui32>)

// CHECK: %[[ARR:.*]] = mogg.tensor.shape
// CHECK: %[[NEW_0:.*]] = mogg.call.view["mo.static.reshape"] (%[[ARG0]], %[[ARR]])
// CHECK: %[[NEW_1:.*]] = mogg.call.view["mo.static.reshape"] (%[[ARG1]], %[[ARR]])

// CHECK: mogg.bind (%[[NEW_0]], %[[NEW_1]]) elemwise_lambda
// CHECK: mogg.call.elementwise["mo.add"]


mo.graph @basicReshapeDyn<N, L>(%arg0 : !mo.tensor<[50], ui32>, %arg1 : !mo.tensor<[5, 10], ui32>) -> (!mo.tensor<[N, 5, L], ui32>) {
  %A = mo.static.reshape(%arg0) : !mo.tensor<[50], ui32> -> !mo.tensor<[N, 5, L], ui32>
  %B = mo.static.reshape(%arg1) : !mo.tensor<[5, 10], ui32> -> !mo.tensor<[N, 5, L], ui32>
  %res = mo.add(%A, %B) : !mo.tensor<[N, 5, L], ui32>
  mo.output %res : !mo.tensor<[N, 5, L], ui32>
}
// CHECK-LABEL: mo.graph @basicReshapeDyn
// CHECK: (%[[ARG0:.*]]: !mo.tensor<[50], ui32>, %[[ARG1:.*]]: !mo.tensor<[5, 10], ui32>)
// CHECK: mogg.kernel(%[[ARG0]]: !mo.tensor<[50], ui32>, %[[ARG1]]: !mo.tensor<[5, 10], ui32>)

// CHECK: %[[ARR:.*]] = mogg.tensor.shape
// CHECK: %[[NEW_0:.*]] = mogg.call.view["mo.static.reshape"] (%[[ARG0]], %[[ARR]])
// CHECK: %[[NEW_1:.*]] = mogg.call.view["mo.static.reshape"] (%[[ARG1]], %[[ARR]])

// CHECK: mogg.bind (%[[NEW_0]], %[[NEW_1]]) elemwise_lambda
// CHECK: mogg.call.elementwise["mo.add"]



mo.graph @multiReshape<N, L>(%arg0: !mo.tensor<[50], ui32>, %arg1: !mo.tensor<[5, 10], ui32>) -> !mo.tensor<[N, 5, L], ui32> {
    %0 = mo.static.reshape(%arg0) : !mo.tensor<[50], ui32> -> !mo.tensor<[N, 5, L], ui32>
    %1 = mo.add(%0, %0) : !mo.tensor<[N, 5, L], ui32>
    %2 = mo.add(%1, %0) : !mo.tensor<[N, 5, L], ui32>
    mo.output %2 : !mo.tensor<[N, 5, L], ui32>
}

// CHECK-LABEL: mo.graph @multiReshape
// CHECK: (%[[ARG0:.*]]: !mo.tensor<[50], ui32>, %[[ARG1:.*]]: !mo.tensor<[5, 10], ui32>)
// CHECK: mogg.kernel(%[[ARG0]] as %{{.*}}: !mo.tensor<[50], ui32>, %[[ARG0]] as %{{.*}}: !mo.tensor<[50], ui32>)

// CHECK: %[[ARR:.*]] = mogg.tensor.shape
// CHECK: %[[RESHAPED1:.*]] = mogg.call.view["mo.static.reshape"] (%{{.*}}, %[[ARR]])
// CHECK: %[[RESHAPED2:.*]] = mogg.call.view["mo.static.reshape"] (%{{.*}}, %[[ARR]])

// CHECK: mogg.bind (%[[RESHAPED1]], %[[RESHAPED2]]) elemwise_lambda
// CHECK: mogg.kernel(%{{.*}}: !mo.tensor<[N, 5, L], ui32>, %[[ARG0]]: !mo.tensor<[50], ui32>)


mo.graph @dontFuseCompound<N, M>(%arg0 : !mo.tensor<[M], ui32>, %arg1 : !mo.tensor<[N], ui32>) -> (!mo.tensor<[N], ui32>) {
  %1 = mo.rebind(%arg0) : !mo.tensor<[M], ui32> -> !mo.tensor<[add(N, 2)], ui32>
  %A = mo.static.reshape(%1) : !mo.tensor<[add(N, 2)], ui32> -> !mo.tensor<[N], ui32>
  %res = mo.add(%A, %arg1) : !mo.tensor<[N], ui32>
  mo.output %res : !mo.tensor<[N], ui32>
}

// CHECK-LABEL: mo.graph @dontFuseCompound<N, M>
// CHECK-NEXT: mogg.kernel(%arg0: !mo.tensor<[M], ui32>, %arg1: !mo.tensor<[N], ui32>) -> !mo.tensor<[N], ui32> {
// CHECK: mogg.tensor.shape
// CHECK: mogg.call.view["mo.static.reshape"] (%arg0, %{{.*}}) {kernel_param = {}}
// CHECK: mogg.bind (%{{.*}}, %arg1) elemwise_lambda

// Example with multiple view operations fused
mo.graph @in_place_slice<d0, d1, d2>(%arg0: !mo.tensor<[3, 3, 3], f32>, %arg1: !mo.tensor<[27], f32>) -> !mo.tensor<[3, 3, 3], f32> {
  %0 = mo.negative(%arg0) : !mo.tensor<[3, 3, 3], f32>
  %1 = mo.static.reshape(%0): !mo.tensor<[3, 3, 3], f32> -> !mo.tensor<[27], f32>
  %2 = mo.add(%1, %arg1) : !mo.tensor<[27], f32>
  %3 = mo.static.reshape(%2) : !mo.tensor<[27], f32> -> !mo.tensor<[3, 3, 3], f32>
  mo.output %3 : !mo.tensor<[3, 3, 3], f32>
}
// Just make sure this processes correctly with reshape view fusion
// CHECK-LABEL: @in_place_slice
// CHECK: mogg.kernel
