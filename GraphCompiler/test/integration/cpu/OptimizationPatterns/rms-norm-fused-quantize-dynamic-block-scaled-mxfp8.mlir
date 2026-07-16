// COM: Test RMSNorm + MXFP8 block-scaled dynamic quantization op fusion.
// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s

// A single-consumer rms_norm feeding an MXFP8 (E4M3 values + E8M0 rank-5 block
// scales) block-scaled quantize on GPU fuses into the composite op.
mo.graph @rms_norm_quantize_block_scaled_mxfp8(
  %input: !mo.tensor<[256, 128], bf16, gpu:0>,
  %gamma: !mo.tensor<[128], bf16, gpu:0>,
  %epsilon: !mo.tensor<[], bf16>,
  %weight_offset: !mo.tensor<[], bf16>
) -> (!mo.tensor<[256, 128], f8e4m3fn, gpu:0>, !mo.tensor<[2, 1, 32, 4, 4], f8e8m0fnu, gpu:0>) {
  %norm = mo.reduce.rms_norm(%input, %gamma, %epsilon, %weight_offset) {multiply_before_cast = true} : (!mo.tensor<[256, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>, !mo.tensor<[], bf16>, !mo.tensor<[], bf16>) -> !mo.tensor<[256, 128], bf16, gpu:0>
  %tensor_sf = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
  %quant:2 = mo.custom {device = #M.device_ref<"gpu", 0>, parameters = {SF_VECTOR_SIZE = 32 : index}, symbol = "mo.quantize.dynamic.block.scaled"}(%norm, %tensor_sf) : (!mo.tensor<[256, 128], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[256, 128], f8e4m3fn, gpu:0>, !mo.tensor<[2, 1, 32, 4, 4], f8e8m0fnu, gpu:0>)
  mo.output %quant#0, %quant#1 : !mo.tensor<[256, 128], f8e4m3fn, gpu:0>, !mo.tensor<[2, 1, 32, 4, 4], f8e8m0fnu, gpu:0>
}
// CHECK-LABEL: mo.graph @rms_norm_quantize_block_scaled_mxfp8
// CHECK: mo.composite.rms_norm_fused_quantize_dynamic_block_scaled
// CHECK-NOT: mo.reduce.rms_norm
// CHECK-NOT: "mo.quantize.dynamic.block.scaled"

// Multi-consumer norm: the bf16 normed tensor is also a graph output, so the
// norm result has two uses. The fused op does not emit the bf16 normed tensor,
// so the pattern must NOT fire.
mo.graph @rms_norm_quantize_block_scaled_mxfp8_multi_consumer(
  %input: !mo.tensor<[256, 128], bf16, gpu:0>,
  %gamma: !mo.tensor<[128], bf16, gpu:0>,
  %epsilon: !mo.tensor<[], bf16>,
  %weight_offset: !mo.tensor<[], bf16>
) -> (!mo.tensor<[256, 128], f8e4m3fn, gpu:0>, !mo.tensor<[2, 1, 32, 4, 4], f8e8m0fnu, gpu:0>, !mo.tensor<[256, 128], bf16, gpu:0>) {
  %norm = mo.reduce.rms_norm(%input, %gamma, %epsilon, %weight_offset) {multiply_before_cast = true} : (!mo.tensor<[256, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>, !mo.tensor<[], bf16>, !mo.tensor<[], bf16>) -> !mo.tensor<[256, 128], bf16, gpu:0>
  %tensor_sf = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
  %quant:2 = mo.custom {device = #M.device_ref<"gpu", 0>, parameters = {SF_VECTOR_SIZE = 32 : index}, symbol = "mo.quantize.dynamic.block.scaled"}(%norm, %tensor_sf) : (!mo.tensor<[256, 128], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[256, 128], f8e4m3fn, gpu:0>, !mo.tensor<[2, 1, 32, 4, 4], f8e8m0fnu, gpu:0>)
  mo.output %quant#0, %quant#1, %norm : !mo.tensor<[256, 128], f8e4m3fn, gpu:0>, !mo.tensor<[2, 1, 32, 4, 4], f8e8m0fnu, gpu:0>, !mo.tensor<[256, 128], bf16, gpu:0>
}
// CHECK-LABEL: mo.graph @rms_norm_quantize_block_scaled_mxfp8_multi_consumer
// CHECK: mo.reduce.rms_norm
// CHECK: "mo.quantize.dynamic.block.scaled"
// CHECK-NOT: mo.composite.rms_norm_fused_quantize_dynamic_block_scaled

// MXFP4 (packed uint8 values, E8M0 scales) shares the block-scaled quantize op
// but is not implemented by the fused MXFP8 kernel: the pattern must NOT fire.
mo.graph @rms_norm_quantize_block_scaled_mxfp4(
  %input: !mo.tensor<[256, 128], bf16, gpu:0>,
  %gamma: !mo.tensor<[128], bf16, gpu:0>,
  %epsilon: !mo.tensor<[], bf16>,
  %weight_offset: !mo.tensor<[], bf16>
) -> (!mo.tensor<[256, 64], ui8, gpu:0>, !mo.tensor<[2, 1, 32, 4, 4], f8e8m0fnu, gpu:0>) {
  %norm = mo.reduce.rms_norm(%input, %gamma, %epsilon, %weight_offset) {multiply_before_cast = true} : (!mo.tensor<[256, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>, !mo.tensor<[], bf16>, !mo.tensor<[], bf16>) -> !mo.tensor<[256, 128], bf16, gpu:0>
  %tensor_sf = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
  %quant:2 = mo.custom {device = #M.device_ref<"gpu", 0>, parameters = {SF_VECTOR_SIZE = 32 : index}, symbol = "mo.quantize.dynamic.block.scaled"}(%norm, %tensor_sf) : (!mo.tensor<[256, 128], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[256, 64], ui8, gpu:0>, !mo.tensor<[2, 1, 32, 4, 4], f8e8m0fnu, gpu:0>)
  mo.output %quant#0, %quant#1 : !mo.tensor<[256, 64], ui8, gpu:0>, !mo.tensor<[2, 1, 32, 4, 4], f8e8m0fnu, gpu:0>
}
// CHECK-LABEL: mo.graph @rms_norm_quantize_block_scaled_mxfp4
// CHECK: mo.reduce.rms_norm
// CHECK: "mo.quantize.dynamic.block.scaled"
// CHECK-NOT: mo.composite.rms_norm_fused_quantize_dynamic_block_scaled

// multiply_before_cast = false disagrees with the fused kernel contract: the
// pattern must NOT fire.
mo.graph @rms_norm_quantize_block_scaled_mxfp8_multiply_before_cast_wrong(
  %input: !mo.tensor<[256, 128], bf16, gpu:0>,
  %gamma: !mo.tensor<[128], bf16, gpu:0>,
  %epsilon: !mo.tensor<[], bf16>,
  %weight_offset: !mo.tensor<[], bf16>
) -> (!mo.tensor<[256, 128], f8e4m3fn, gpu:0>, !mo.tensor<[2, 1, 32, 4, 4], f8e8m0fnu, gpu:0>) {
  %norm = mo.reduce.rms_norm(%input, %gamma, %epsilon, %weight_offset) {multiply_before_cast = false} : (!mo.tensor<[256, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>, !mo.tensor<[], bf16>, !mo.tensor<[], bf16>) -> !mo.tensor<[256, 128], bf16, gpu:0>
  %tensor_sf = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
  %quant:2 = mo.custom {device = #M.device_ref<"gpu", 0>, parameters = {SF_VECTOR_SIZE = 32 : index}, symbol = "mo.quantize.dynamic.block.scaled"}(%norm, %tensor_sf) : (!mo.tensor<[256, 128], bf16, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[256, 128], f8e4m3fn, gpu:0>, !mo.tensor<[2, 1, 32, 4, 4], f8e8m0fnu, gpu:0>)
  mo.output %quant#0, %quant#1 : !mo.tensor<[256, 128], f8e4m3fn, gpu:0>, !mo.tensor<[2, 1, 32, 4, 4], f8e8m0fnu, gpu:0>
}
// CHECK-LABEL: mo.graph @rms_norm_quantize_block_scaled_mxfp8_multiply_before_cast_wrong
// CHECK: mo.reduce.rms_norm
// CHECK: "mo.quantize.dynamic.block.scaled"
// CHECK-NOT: mo.composite.rms_norm_fused_quantize_dynamic_block_scaled
