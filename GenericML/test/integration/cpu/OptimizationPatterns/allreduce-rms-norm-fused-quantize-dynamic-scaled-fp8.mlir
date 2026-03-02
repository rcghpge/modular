// COM: Test AllReduce + RMSNorm + FP8 dynamic quantization op fusion
// RUN: mo-opt %s --mogg-pattern-fusion | FileCheck %s

mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8<total_seq_len>(
  %input0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %input1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
  %input2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
  %input3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
  %sign_buff0: !mo.buffer<[537919488], ui8, gpu:0>,
  %sign_buff1: !mo.buffer<[537919488], ui8, gpu:1>,
  %sign_buff2: !mo.buffer<[537919488], ui8, gpu:2>,
  %sign_buff3: !mo.buffer<[537919488], ui8, gpu:3>,
  %c: !mo.chain,
  %residual: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %gamma: !mo.tensor<[16384], bf16, gpu:0>,
  %epsilon: !mo.tensor<[], bf16>,
  %weight_offset: !mo.tensor<[], bf16>
) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>, !mo.chain) {
    %allred:4, %out_chain = mo.distributed.allreduce.sum(%input0, %input1, %input2, %input3, %sign_buff0, %sign_buff1, %sign_buff2, %sign_buff3, %c) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.buffer<[537919488], ui8, gpu:2>, !mo.buffer<[537919488], ui8, gpu:3>, !mo.chain) -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.chain)
    %add = mo.add(%residual, %allred#0) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    %norm = mo.custom {device = #M.device_ref<"gpu", 0>, parameters = {multiply_before_cast = true}, symbol = "rms_norm"}(%add, %gamma, %epsilon, %weight_offset) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[16384], bf16, gpu:0>, !mo.tensor<[], bf16>, !mo.tensor<[], bf16>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    %scale_ub = mo.constant {value = #M.dense_array<1.200000e+03> : tensor<f32>} : !mo.tensor<[], f32>
    %norm_quant:2 = mo.custom {device = #M.device_ref<"gpu", 0>, parameters = {group_size_or_per_token = 16384 : index}, symbol = "mo.quantize_dynamic_scaled_float8"}(%norm, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>)
    mo.output %norm_quant#0, %norm_quant#1, %out_chain : !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>, !mo.chain
}
// CHECK-LABEL: mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8
// CHECK: mo.distributed.allreduce_add_rms_norm_quant_fp8(

mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8_residual_use<total_seq_len>(
  %input0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %input1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
  %input2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
  %input3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
  %sign_buff0: !mo.buffer<[537919488], ui8, gpu:0>,
  %sign_buff1: !mo.buffer<[537919488], ui8, gpu:1>,
  %sign_buff2: !mo.buffer<[537919488], ui8, gpu:2>,
  %sign_buff3: !mo.buffer<[537919488], ui8, gpu:3>,
  %c: !mo.chain,
  %residual: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %residual2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %gamma: !mo.tensor<[16384], bf16, gpu:0>,
  %epsilon: !mo.tensor<[], bf16>,
  %weight_offset: !mo.tensor<[], bf16>
) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.chain) {
    %allred:4, %out_chain = mo.distributed.allreduce.sum(%input0, %input1, %input2, %input3, %sign_buff0, %sign_buff1, %sign_buff2, %sign_buff3, %c) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.buffer<[537919488], ui8, gpu:2>, !mo.buffer<[537919488], ui8, gpu:3>, !mo.chain) -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.chain)
    %add = mo.add(%residual, %allred#0) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    %norm = mo.custom {device = #M.device_ref<"gpu", 0>, parameters = {multiply_before_cast = true}, symbol = "rms_norm"}(%add, %gamma, %epsilon, %weight_offset) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[16384], bf16, gpu:0>, !mo.tensor<[], bf16>, !mo.tensor<[], bf16>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    %scale_ub = mo.constant {value = #M.dense_array<1.200000e+03> : tensor<f32>} : !mo.tensor<[], f32>
    %norm_quant:2 = mo.custom {device = #M.device_ref<"gpu", 0>, parameters = {group_size_or_per_token = 16384 : index}, symbol = "mo.quantize_dynamic_scaled_float8"}(%norm, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>)
    %add2 = mo.add(%residual2, %add) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    mo.output %norm_quant#0, %norm_quant#1, %add2, %out_chain : !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.chain
}
// CHECK-LABEL: mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8_residual_use
// CHECK: %[[OUT_V:.*]], %[[OUT_S:.*]], %[[OUT_R:.*]], %[[OUT_C:.*]] = mo.distributed.allreduce_add_rms_norm_quant_fp8(
// CHECK: mo.add(%{{.*}}, %[[OUT_R]])

mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8_multi_gpu<total_seq_len>(
  %input0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %input1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
  %input2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>,
  %input3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>,
  %sign_buff0: !mo.buffer<[537919488], ui8, gpu:0>,
  %sign_buff1: !mo.buffer<[537919488], ui8, gpu:1>,
  %sign_buff2: !mo.buffer<[537919488], ui8, gpu:2>,
  %sign_buff3: !mo.buffer<[537919488], ui8, gpu:3>,
  %c: !mo.chain,
  %residual0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>,
  %residual1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>,
  %gamma0: !mo.tensor<[16384], bf16, gpu:0>,
  %gamma1: !mo.tensor<[16384], bf16, gpu:1>,
  %epsilon: !mo.tensor<[], bf16>,
  %weight_offset: !mo.tensor<[], bf16>
) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>, !mo.tensor<[1, total_seq_len], bf16, gpu:1>, !mo.chain) {
    %allred:4, %out_chain = mo.distributed.allreduce.sum(%input0, %input1, %input2, %input3, %sign_buff0, %sign_buff1, %sign_buff2, %sign_buff3, %c) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.buffer<[537919488], ui8, gpu:2>, !mo.buffer<[537919488], ui8, gpu:3>, !mo.chain) -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.chain)

    %scale_ub = mo.constant {value = #M.dense_array<1.200000e+03> : tensor<f32>} : !mo.tensor<[], f32>

    // Add + RMNSNorm + Quant (GPU:0)
    %add0 = mo.add(%residual0, %allred#0) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    %norm0 = mo.custom {device = #M.device_ref<"gpu", 0>, parameters = {multiply_before_cast = true}, symbol = "rms_norm"}(%add0, %gamma0, %epsilon, %weight_offset) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[16384], bf16, gpu:0>, !mo.tensor<[], bf16>, !mo.tensor<[], bf16>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
    %norm_quant0:2 = mo.custom {device = #M.device_ref<"gpu", 0>, parameters = {group_size_or_per_token = 16384 : index}, symbol = "mo.quantize_dynamic_scaled_float8"}(%norm0, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>)

    // Add + RMNSNorm + Quant (GPU:1)
    %add1 = mo.add(%residual1, %allred#1) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>
    %norm1 = mo.custom {device = #M.device_ref<"gpu", 1>, parameters = {multiply_before_cast = true}, symbol = "rms_norm"}(%add1, %gamma1, %epsilon, %weight_offset) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[16384], bf16, gpu:1>, !mo.tensor<[], bf16>, !mo.tensor<[], bf16>) -> !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>
    %norm_quant1:2 = mo.custom {device = #M.device_ref<"gpu", 1>, parameters = {group_size_or_per_token = 16384 : index}, symbol = "mo.quantize_dynamic_scaled_float8"}(%norm1, %scale_ub) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[], f32>) -> (!mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>, !mo.tensor<[1, total_seq_len], bf16, gpu:1>)

    mo.output %norm_quant0#0, %norm_quant0#1, %norm_quant1#0, %norm_quant1#1, %out_chain : !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:0>, !mo.tensor<[1, total_seq_len], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], f8e4m3fn, gpu:1>, !mo.tensor<[1, total_seq_len], bf16, gpu:1>, !mo.chain
}

// CHECK-LABEL: mo.graph @all_reduce_rms_norm_quantize_dynamic_scaled_fp8_multi_gpu
// CHECK: mo.distributed.allreduce_add_rms_norm_quant_fp8({{.*}}device = #M.device_ref<"gpu", 0>
// CHECK-NEXT: mo.distributed.allreduce_add_rms_norm_quant_fp8({{.*}}device = #M.device_ref<"gpu", 1>
