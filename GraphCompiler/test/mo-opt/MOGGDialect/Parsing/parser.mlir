// RUN: mo-opt %s | mo-opt

// Test we can parse the custom mogg operations.

// Test MOGG_ShapeFuncResult type roundtrip with different ranks
func.func @shapeFuncResultTypeRoundTrip(%shape_rank3: !mogg.shape_func_result<3>,
                                        %shape_rank5: !mogg.shape_func_result<5>) -> (!mogg.shape_func_result<3>, !mogg.shape_func_result<5>) {
  return %shape_rank3, %shape_rank5 : !mogg.shape_func_result<3>, !mogg.shape_func_result<5>
}

mo.graph @basicMatmul<N, M, K>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[], f32>, %buffer: !mo.buffer<[], f32>) -> !mo.tensor<[M, N], f32> {
  %0 = mogg.kernel(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32, {layout = #mo.layout<KN>}>, %buffer: !mo.buffer<[], f32>) -> !mo.tensor<[M, N], f32> {
    %1 = mogg.output_placeholders : !mo.tensor<[M, N], f32>
    %2 = mogg.bind (%1) output_lambda (%arg5: f32, %arg6: !pop.array<2, index>) {
      // Checking that it can load from both tensors and buffers.
      %b = "mogg.tensor.load" (%buffer, %arg6) : (!mo.buffer<[], f32>, !pop.array<2, index>) -> f32
      %t = "mogg.tensor.load" (%arg0, %arg6) : (!mo.tensor<[M, K], f32>, !pop.array<2, index>) -> f32
      mogg.output %arg5, %arg6 : f32, !pop.array<2, index>
    }: (!mo.tensor<[M, N], f32>) -> !mo.tensor<[M, N], f32>
    %3 = mogg.device_context_placeholder {device = #M.device_ref<"cpu", 0>} : !mogg.context
    mogg.call.execute["mo.matmul"] inputs(%arg0, %arg1, %3 : !mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32, {layout = #mo.layout<KN>}>, !mogg.context) outputs(%2 : !mo.tensor<[M, N], f32>) {isThrows, kernel_param = {packed_b = false, transpose_b = false}}
  } fusionInfo = mo.matmul:4 {allocIndices = [], device = #M.device_ref<"cpu", 0>, hasReadEffect = false, hasWriteEffect = false, isAsync = true, outputParamDecls = #kgen<param.decls[]>}
  mo.output %0 : !mo.tensor<[M, N], f32>
}
