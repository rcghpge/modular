// RUN: mo-opt %s --mo-pipeline --mo-to-mogg --mogg-to-mgp -o /dev/null

// Regression test: mo.parallel with expression-form dims in tensor types.
//
// The VisionPatchMerger in Qwen3-VL produces tensors with div(D0, 4) dims
// via reshape.  These flow into bundled allreduce as parallel inputs.
// The kernel stub infrastructure must handle expression-form dims
// (ParamExprAttr) in TensorInitOps, not just named refs (ParamDeclRefAttr).
//
// Additionally, HoistParameterExpressionsPass must not insert mo.rebind
// inside mo.parallel bodies (which would break lowerParallelOp).
//
// NOTE: Requires a system with at least 2 GPUs.

mo.graph @bundled_allreduce_expr_dim_inputs<D0>(
    %in0: !mo.tensor<[D0, 3], f32, gpu:0>,
    %in1: !mo.tensor<[D0, 3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[div(D0, 2), 3], f32>, !mo.tensor<[div(D0, 2), 3], f32>) {
  // Create tensors with expression-form dims (mimics VisionPatchMerger reshape).
  %r0 = mo.rebind(%in0) : !mo.tensor<[D0, 3], f32, gpu:0> -> !mo.tensor<[div(D0, 2), 3], f32, gpu:0>
  %r1 = mo.rebind(%in1) : !mo.tensor<[D0, 3], f32, gpu:1> -> !mo.tensor<[div(D0, 2), 3], f32, gpu:1>
  %ch = mo.chain.create()
  // Parallel inputs have div(D0, 2) -- an expression-form dim that kernel
  // stubs must provide runtime values for.
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
