// RUN: mo-opt %s --mo-pipeline --mo-to-mogg --mogg-to-mgp -o /dev/null

// Regression test: HoistParameterExpressionsPass must not insert mo.rebind
// inside mo.parallel bodies.  The parallel body must remain kernel-only
// for lowerParallelOp.
//
// The graph feeds tensors with a complex expression dim (div(D0, 2)) into
// the parallel, so that HoistParameterExpressionsPass would normally try
// to rewrite the allreduce's types and insert rebinds.  The pass must
// skip ops inside mo.parallel.
//
// NOTE: Requires a system with at least 2 GPUs.

mo.graph @bundled_allreduce_complex_expr<D0>(
    %in0: !mo.tensor<[D0, 3], f32, gpu:0>,
    %in1: !mo.tensor<[D0, 3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[div(D0, 2), 3], f32>, !mo.tensor<[div(D0, 2), 3], f32>) {
  // Create tensors with a complex expression dim via rebind.
  %r0 = mo.rebind(%in0) : !mo.tensor<[D0, 3], f32, gpu:0> -> !mo.tensor<[div(D0, 2), 3], f32, gpu:0>
  %r1 = mo.rebind(%in1) : !mo.tensor<[D0, 3], f32, gpu:1> -> !mo.tensor<[div(D0, 2), 3], f32, gpu:1>
  %ch = mo.chain.create()
  // The parallel inputs have div(D0, 2) which is a ParamOperatorAttr.
  // HoistParameterExpressionsPass must NOT rewrite the allreduce inside.
  %res:2, %out_ch = mo.parallel (%arg, %sig) in
      ((%r0, %sig0) : (!mo.tensor<[div(D0, 2), 3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>),
       (%r1, %sig1) : (!mo.tensor<[div(D0, 2), 3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:1>))
      chain(%ch) {
    %out, %ch_out = mo.bundled.allreduce.sum(%arg, %sig, %ch)
        : (!mo.tensor<[div(D0, 2), 3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.chain)
       -> (!mo.tensor<[div(D0, 2), 3], f32, gpu:0>, !mo.chain)
    mo.yield %out : !mo.tensor<[div(D0, 2), 3], f32, gpu:0>
  }
  %h0, %ch3 = mo.transfer[%out_ch] %res#0 : !mo.tensor<[div(D0, 2), 3], f32, gpu:0> to <"cpu", 0>
  %h1, %ch4 = mo.transfer[%ch3] %res#1 : !mo.tensor<[div(D0, 2), 3], f32, gpu:1> to <"cpu", 0>
  mo.output %h0, %h1 : !mo.tensor<[div(D0, 2), 3], f32>, !mo.tensor<[div(D0, 2), 3], f32>
}
