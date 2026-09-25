// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-elementwise=max-fusion-set-size=1000 | FileCheck %s --check-prefix=UNCAPPED
// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-elementwise=max-fusion-set-size=3 | FileCheck %s --check-prefix=CAPPED

// A residual chain of cheap (small-tensor) elementwise ops: each `%rN` is used
// twice -- by the next `tanh` and by the next residual `add` -- so it is a
// multi-use producer whose consumers are all elementwise foreaches. Input
// fusion duplicates such a producer into every consumer, and because the
// operand is itself a fused producer the whole prefix is recomputed at each
// step. Without a cap the entire chain collapses into a single foreach that
// recomputes far more `add`s than the five in the source (quadratic blow-up).
// The fusion-set cap bounds how many kernels land in one consumer, so the chain
// is split across several kernels instead of one over-fused one.

mo.graph @residual_chain(%a: !mo.tensor<[64], f32, gpu:0>) -> !mo.tensor<[64], f32, gpu:0> {
  %r0 = mo.exp(%a) : !mo.tensor<[64], f32, gpu:0>
  %d1 = mo.tanh(%r0) : !mo.tensor<[64], f32, gpu:0>
  %r1 = mo.add(%r0, %d1) : !mo.tensor<[64], f32, gpu:0>
  %d2 = mo.tanh(%r1) : !mo.tensor<[64], f32, gpu:0>
  %r2 = mo.add(%r1, %d2) : !mo.tensor<[64], f32, gpu:0>
  %d3 = mo.tanh(%r2) : !mo.tensor<[64], f32, gpu:0>
  %r3 = mo.add(%r2, %d3) : !mo.tensor<[64], f32, gpu:0>
  %d4 = mo.tanh(%r3) : !mo.tensor<[64], f32, gpu:0>
  %r4 = mo.add(%r3, %d4) : !mo.tensor<[64], f32, gpu:0>
  %d5 = mo.tanh(%r4) : !mo.tensor<[64], f32, gpu:0>
  %r5 = mo.add(%r4, %d5) : !mo.tensor<[64], f32, gpu:0>
  mo.output %r5 : !mo.tensor<[64], f32, gpu:0>
}

// Uncapped: the whole chain over-fuses into ONE foreach, with the residual adds
// duplicated well beyond the five in the source.
// UNCAPPED:          map.iter.foreach<
// UNCAPPED-COUNT-10: map.call.elementwise @{{.*}}.Add
// UNCAPPED-NOT:      map.iter.foreach<

// Capped: the cap forces the chain across several kernels rather than one.
// CAPPED-COUNT-4: map.iter.foreach<
