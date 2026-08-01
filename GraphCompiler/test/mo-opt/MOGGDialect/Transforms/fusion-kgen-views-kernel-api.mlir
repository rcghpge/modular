// RUN: mo-opt --mogg-import-packages --mo-to-mogg --mogg-fuse-views -cse %s | FileCheck %s

// CHECK-LABEL:   mo.graph @back2backBroadcastReshape
// CHECK: fusionInfo = View("mo.static.reshape":{{.*}}, "mo.add":{{.*}})
mo.graph @back2backBroadcastReshape<N, M, K, J>(%arg0: !mo.tensor<[K, M], f32>, %arg1: !mo.tensor<[], f32>) -> (!mo.tensor<[K, M], f32>) {
  %bcast = mo.static.broadcast_to(%arg1) : !mo.tensor<[], f32> -> !mo.tensor<[J], f32>
  %reshape = mo.static.reshape(%bcast) : !mo.tensor<[J], f32> -> !mo.tensor<[K, M], f32>
  %1 = mo.add(%arg0, %reshape) : !mo.tensor<[K, M], f32>
  mo.output %1 : !mo.tensor<[K, M], f32>
}

// A large chain of views to stress test pass. Suboptimal implementations may timeout.
// CHECK-LABEL:   mo.graph @largeChainOfViews
mo.graph @largeChainOfViews<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R>(%arg0: !mo.tensor<[], f32>) -> (!mo.tensor<[R], f32>) {
  %bcast1 = mo.static.broadcast_to(%arg0) : !mo.tensor<[], f32> -> !mo.tensor<[A], f32>
  %bcast2 = mo.static.broadcast_to(%bcast1) : !mo.tensor<[A], f32> -> !mo.tensor<[B], f32>
  %bcast3 = mo.static.broadcast_to(%bcast2) : !mo.tensor<[B], f32> -> !mo.tensor<[C], f32>
  %bcast4 = mo.static.broadcast_to(%bcast3) : !mo.tensor<[C], f32> -> !mo.tensor<[D], f32>
  %bcast5 = mo.static.broadcast_to(%bcast4) : !mo.tensor<[D], f32> -> !mo.tensor<[E], f32>
  %bcast6 = mo.static.broadcast_to(%bcast5) : !mo.tensor<[E], f32> -> !mo.tensor<[F], f32>
  %bcast7 = mo.static.broadcast_to(%bcast6) : !mo.tensor<[F], f32> -> !mo.tensor<[G], f32>
  %bcast8 = mo.static.broadcast_to(%bcast7) : !mo.tensor<[G], f32> -> !mo.tensor<[H], f32>
  %bcast9 = mo.static.broadcast_to(%bcast8) : !mo.tensor<[H], f32> -> !mo.tensor<[I], f32>

  %bcast10 = mo.static.broadcast_to(%bcast9) : !mo.tensor<[I], f32> -> !mo.tensor<[J], f32>
  %bcast11 = mo.static.broadcast_to(%bcast10) : !mo.tensor<[J], f32> -> !mo.tensor<[K], f32>
  %bcast12 = mo.static.broadcast_to(%bcast11) : !mo.tensor<[K], f32> -> !mo.tensor<[L], f32>
  %bcast13 = mo.static.broadcast_to(%bcast12) : !mo.tensor<[L], f32> -> !mo.tensor<[M], f32>
  %bcast14 = mo.static.broadcast_to(%bcast13) : !mo.tensor<[M], f32> -> !mo.tensor<[N], f32>
  %bcast15 = mo.static.broadcast_to(%bcast14) : !mo.tensor<[N], f32> -> !mo.tensor<[O], f32>
  %bcast16 = mo.static.broadcast_to(%bcast15) : !mo.tensor<[O], f32> -> !mo.tensor<[P], f32>
  %bcast17 = mo.static.broadcast_to(%bcast16) : !mo.tensor<[P], f32> -> !mo.tensor<[Q], f32>
  %bcast18 = mo.static.broadcast_to(%bcast17) : !mo.tensor<[Q], f32> -> !mo.tensor<[R], f32>

  mo.output %bcast18 : !mo.tensor<[R], f32>
}
