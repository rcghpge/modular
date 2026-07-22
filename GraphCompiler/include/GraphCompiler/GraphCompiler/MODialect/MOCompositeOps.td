//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines "composite" operations in the MO dialect: dedicated MO ops that
// represent a fused subgraph emitted for a pattern-matching fusion. They live in
// the `mo.composite.*` namespace to distinguish them from core MO operators.
// Pattern-based fusions should replace a matched subgraph with one of these ops
// (never an `mo.custom`); they can still be emitted directly.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOCOMPOSITEOPS_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOCOMPOSITEOPS_TD

include "GraphCompiler/GraphCompiler/MODialect/MOOps.td"

// Base class for composite ops. Forces the `mo.composite.` mnemonic namespace,
// emits the generated C++ classes into the `::M::MO::Composite` namespace (like
// `::M::MO::Static` for `mo.static.*`), and tags the op with the `MOComposite`
// marker trait so composite ops are easy to identify. Ops whose semantics need a
// more specialized base (e.g. `MO_DotBase`, `MO_DistributedOp`) instead spell
// out the `composite.` mnemonic prefix, set the `Composite` cppNamespace, and
// add `MO_CompositeOpTrait` to their trait list directly.
class MO_CompositeOp<string mnemonic, list<Trait> traits = []>
  : MO_TensorOp<"composite." # mnemonic, [MO_CompositeOpTrait] # traits> {
  string cppNamespace = MODialect.cppNamespace # "::Composite";
}

//===----------------------------------------------------------------------===//
// ReduceRMSNormRoPEOp
//===----------------------------------------------------------------------===//

def MOComposite_ReduceRMSNormRoPEOp : MO_CompositeOp<"rms_norm_rope", [
  Pure,
  MO_SingleDeviceWithHostOperands<["epsilon", "weightOffset"]>,
  MO_ParamDeclaration,
]> {
  let hasVerifier = 1;
  let summary = "Fused RMS normalization with Rotary Position Embedding";
  let description = [{
    Fused operation computing RMS normalization followed by Rotary Position
    Embedding (RoPE):

      normed = rms_norm(input, weight, epsilon, weight_offset)
      x1, x2 = split(normed, axis=-1)
      rotated = concat(-x2, x1, axis=-1)
      result = normed * cos_vals + rotated * sin_vals

    Example:

    ```mlir
      %result = mo.composite.rms_norm_rope(%input, %weight, %epsilon, %offset,
                                           %cos_vals, %sin_vals)
        {multiply_before_cast = false} :
        (!mo.tensor<[2, 3, 128], bf16, gpu:0>, !mo.tensor<[128], bf16, gpu:0>,
         !mo.tensor<[], bf16>, !mo.tensor<[], bf16>,
         !mo.tensor<[2, 3, 128], f32, gpu:0>, !mo.tensor<[2, 3, 128], f32, gpu:0>)
        -> !mo.tensor<[2, 3, 128], bf16, gpu:0>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$weight,
    MO_Tensor:$epsilon,
    MO_Tensor:$weightOffset,
    MO_Tensor:$cosVals,
    MO_Tensor:$sinVals,
    BoolAttr:$multiply_before_cast,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `{` `multiply_before_cast` `=` $multiply_before_cast `}` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// ReduceRMSNormFusedResidualAddOp
//===----------------------------------------------------------------------===//

def MOComposite_ReduceRMSNormFusedResidualAddOp : MO_CompositeOp<"rms_norm_fused_residual_add", [
  Pure,
  MO_SingleDeviceWithHostOperands<["epsilon1", "epsilon2", "weightOffset1", "weightOffset2"]>,
  MO_ParamDeclaration,
]> {
  let summary = "Fused RMS norm with residual add operation";
  let description = [{
    Fused operation computing:
      intermediate = rms_norm(input, gamma1, epsilon1, weight_offset1) + residual_input
      output = rms_norm(intermediate, gamma2, epsilon2, weight_offset2)

    Returns both the final output and the post-add intermediate tensor.

    Example:

    ```mlir
      %output, %intermediate = mo.composite.rms_norm_fused_residual_add(
          %input, %residual, %gamma1, %gamma2, %eps1, %eps2, %offset1, %offset2) {
          multiply_before_cast1 = false, multiply_before_cast2 = false} :
        (...) -> (!mo.tensor<[3, 2], f32>, !mo.tensor<[3, 2], f32>)
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$residualInput,
    MO_Tensor:$gamma1,
    MO_Tensor:$gamma2,
    MO_Tensor:$epsilon1,
    MO_Tensor:$epsilon2,
    MO_Tensor:$weightOffset1,
    MO_Tensor:$weightOffset2,
    BoolAttr:$multiply_before_cast,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$output, MO_Tensor:$intermediate);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `{` `multiply_before_cast` `=` $multiply_before_cast `}` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// RMSNormResidualAddOp
//===----------------------------------------------------------------------===//

def MOComposite_RMSNormResidualAddOp : MO_CompositeOp<"rms_norm_residual_add", [
  Pure,
  MO_SingleDeviceWithHostOperands<["epsilon", "weightOffset"]>,
  MO_ParamDeclaration,
]> {
  let summary = "Fused single-norm residual add + RMS norm operation";
  let description = [{
    Fused operation computing:
      intermediate = input + residual_input
      output = rms_norm(intermediate, gamma, epsilon, weight_offset)

    Returns both the final normalized output and the post-add intermediate
    tensor. This is the canonical transformer/mamba pre-norm boundary
    `rms_norm(residual + out)`, where the pre-add value is carried forward as
    the next block's residual. Unlike `rms_norm_fused_residual_add`, there is a
    single RMS norm (no inner norm on `input`).

    Example:

    ```mlir
      %output, %intermediate = mo.composite.rms_norm_residual_add(
          %input, %residual, %gamma, %eps, %offset) {
          multiply_before_cast = false} :
        (...) -> (!mo.tensor<[3, 2], f32>, !mo.tensor<[3, 2], f32>)
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$residualInput,
    MO_Tensor:$gamma,
    MO_Tensor:$epsilon,
    MO_Tensor:$weightOffset,
    BoolAttr:$multiply_before_cast,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$output, MO_Tensor:$intermediate);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `{` `multiply_before_cast` `=` $multiply_before_cast `}` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// FusedConcatSliceOp
//===----------------------------------------------------------------------===//

def MOComposite_FusedConcatSliceOp : MO_CompositeOp<"concat_slice", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  MO_SingleDevice
]> {
  let summary = "Fused Concatenation + slice operation";
  let description = [{
    This operation performs two operations at once:
    %concat = mo.concat[axis](inputs)
    %slice = mo.slice(%concat)
    And returns both the concat and the slice result.
  }];

  let arguments = (ins
    IndexAttr:$axis,
    Variadic<MO_Tensor>:$inputs,
    ArrayAttr:$static_starts,
    ArrayAttr:$static_steps,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$concatResult, MO_Tensor: $sliceResult);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls)
    `[` $axis `]`
    `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

//===----------------------------------------------------------------------===//
// FusedMatmulAddOp
//===----------------------------------------------------------------------===//

def MOComposite_FusedMatmulAddOp : MO_DotBase<"composite.matmul_add", [MO_CompositeOpTrait]> {
  string cppNamespace = MODialect.cppNamespace # "::Composite";
  let summary = "Fused matrix-matrix multiplication with element-wise add";
  let description = [{
    Computes C = A @ B (optionally transposed) + residual, fusing the matmul
    and the residual addition into a single kernel call.

    `residual` may be rank-2 (same shape as the output, element-wise add) or
    rank-1 (broadcast along the row dimension, i.e. a bias vector).

    This operation is currently only lowered for SM100 (B200) targets, where
    the residual is passed directly to the matmul kernel as an epilogue tensor.

    Example (2-D residual):

    ```mlir
      %res = mo.composite.matmul_add(%a, %b, %residual) {transpose_b = true} : (
        !mo.tensor<[4, 512], bf16>,
        !mo.tensor<[1536, 512], bf16>,
        !mo.tensor<[4, 1536], bf16>
      ) -> !mo.tensor<[4, 1536], bf16>
    ```

    Example (1-D bias broadcast):

    ```mlir
      %res = mo.composite.matmul_add(%a, %b, %bias) {transpose_b = true} : (
        !mo.tensor<[4, 512], bf16>,
        !mo.tensor<[1536, 512], bf16>,
        !mo.tensor<[1536], bf16>
      ) -> !mo.tensor<[4, 1536], bf16>
    ```
  }];

  let arguments = (
    ins MO_Rank2Tensor:$input_a,
    MO_Rank2Tensor:$input_b,
    MO_Tensor:$residual,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b
  );
  let results = (outs MO_Rank2Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// AllReduceAddRMSNormQuantFP8Op
//===----------------------------------------------------------------------===//

def MOComposite_AllReduceAddRMSNormQuantFP8Op : MO_DistributedOp<"composite.distributed.allreduce_add_rms_norm_quant_fp8", [
  MO_MultiDevice,
  MO_CompositeOpTrait
]> {
  string cppNamespace = MODialect.cppNamespace # "::Composite";
  let summary = "Fused AllreduceSum + Add + RMSNorm + Quant across N devices for a single device.";
  let description = [{
    Allreduce takes in inputs each coming from a different device with
    the same shape as the final output and performs a sum reduction
    across the devices. This op instance executes on a specific device
    (specified by the device attribute) and produces the output for that device.

    This op also applies a residual (add), then RMSNorm and dynamic FP8 quantization to the output of AllReduce.
    It returns both the quantized output value and the quantization scale.
    It also returns the intermediate output of the residual (add) op.
  }];
  // Variadic allreduce has a folder since it can take 1 input.
  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Variadic<MO_Buffer>:$signalBuffers,
    Variadic<MO_Tensor>:$residual,
    Variadic<MO_Tensor>:$gamma,
    Variadic<MO_RankedScalarFloatTensor>:$epsilon,
    Variadic<MO_RankedScalarFloatTensor>: $weightOffset,
    Variadic<MO_RankedScalarFloatTensor>: $scaleUb,
    MO_Chain:$inChain
  );
  let results = (outs
    Variadic<MO_Tensor>:$output,
    Variadic<MO_Tensor>:$outScale,
    Variadic<MO_Tensor>:$outResidual,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// BundledAllReduceAddRMSNormQuantFP8Op
//===----------------------------------------------------------------------===//

def MOComposite_BundledAllReduceAddRMSNormQuantFP8Op : MO_CompositeOp<
    "bundled.allreduce_add_rms_norm_quant_fp8", [
  MO_SingleDeviceWithPeerInputs<["epsilon", "weightOffset", "scaleUb"]>,
  DeclareOpInterfaceMethods<MO_MutableOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  MO_SameVariadicOperandSize,
  MO_BundledCollective
]> {
  let summary = "Per-device fused allreduce.sum + add + rms_norm + fp8 "
                "quantize, for use inside mo.parallel.";
  let description = [{
    Per-device entry point for the fused `allreduce.sum` +
    residual add + RMS norm + dynamic-scaled FP8 quantize chain, used inside
    an `mo.parallel` region.  Takes N peer tensor inputs (from
    `mo.bundled.expand`), N signal buffers (captured from graph scope),
    per-device residual and gamma tensors, the epsilon / weight offset /
    scale-upper-bound scalars, and a chain.

    Returns the FP8 quantized output for this device, its scale tensor, the
    intermediate residual (post-add) tensor, and an output chain.  This is
    the bundled analog of
    `mo.composite.distributed.allreduce_add_rms_norm_quant_fp8`.
  }];
  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Variadic<MO_Buffer>:$signalBuffers,
    MO_Tensor:$residual,
    MO_Tensor:$gamma,
    MO_RankedScalarFloatTensor:$epsilon,
    MO_RankedScalarFloatTensor:$weightOffset,
    MO_RankedScalarFloatTensor:$scaleUb,
    MO_Chain:$inChain
  );
  let results = (outs
    MO_Tensor:$output,
    MO_Tensor:$outScale,
    MO_Tensor:$outResidual,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Flash attention ops
//===----------------------------------------------------------------------===//
//
// The multi-head-attention fusion produces one of these device/mask-specific
// ops, each a 1:1 wrapper around an existing flash-attention kernel (the kernel
// is unchanged, just registered under the op name). Splitting per (device,
// mask) keeps every kernel's existing signature intact and avoids any dispatch
// in the graph compiler.

def MOComposite_NoMaskFlashAttentionCPUOp
    : MO_CompositeOp<"no_mask_flash_attention_cpu", [
  MO_SingleDeviceWithHostOperands<["scale"]>,
]> {
  let summary = "Fused multi-head (flash) attention on CPU, without a mask";
  let description = [{
    Fused scaled-dot-product attention (`softmax(Q @ K^T * scale) @ V`) on CPU
    with no attention mask. `scale` is a host scalar (`f32`). Lowers 1:1 to the
    `no_mask_flash_attention_cpu` kernel.
  }];
  let arguments = (ins
    MO_Tensor:$query,
    MO_Tensor:$key,
    MO_Tensor:$value,
    MO_Tensor:$scale
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOComposite_MaskedFlashAttentionCPUOp
    : MO_CompositeOp<"masked_flash_attention_cpu", [
  MO_SingleDeviceWithHostOperands<["scale"]>,
]> {
  let summary = "Fused multi-head (flash) attention on CPU, with a mask";
  let description = [{
    Fused scaled-dot-product attention (`softmax(Q @ K^T * scale + mask) @ V`)
    on CPU. `scale` is a host scalar (`f32`). Lowers 1:1 to the
    `with_mask_flash_attention_cpu` kernel.
  }];
  let arguments = (ins
    MO_Tensor:$query,
    MO_Tensor:$key,
    MO_Tensor:$value,
    MO_Tensor:$mask,
    MO_Tensor:$scale
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOComposite_MaskedFlashAttentionGPUOp
    : MO_CompositeOp<"masked_flash_attention_gpu", [
  MO_SingleDeviceWithHostOperands<["scale"]>,
]> {
  let summary = "Fused multi-head (flash) attention on GPU, with a mask";
  let description = [{
    Fused scaled-dot-product attention (`softmax(Q @ K^T * scale + mask) @ V`)
    on GPU. `scale` is a host scalar (`f32`). Lowers 1:1 to the
    `masked_flash_attention_gpu` kernel.
  }];
  let arguments = (ins
    MO_Tensor:$query,
    MO_Tensor:$key,
    MO_Tensor:$value,
    MO_Tensor:$mask,
    MO_Tensor:$scale
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// RMSNormFusedQuantizeDynamicScaledFP8Op
//===----------------------------------------------------------------------===//

def MOComposite_RMSNormFusedQuantizeDynamicScaledFP8Op
    : MO_CompositeOp<"rms_norm_fused_quantize_dynamic_scaled_fp8", [
  Pure,
  MO_SingleDeviceWithHostOperands<["epsilon", "weightOffset", "scaleUb"]>,
]> {
  let summary = "Fused RMS norm + dynamic-scaled FP8 quantize";
  let description = [{
    Fused operation computing token-wise dynamic-scaled FP8 quantization of an
    RMS-normalized input:

      normed = rms_norm(input, weight, epsilon, weight_offset)
                 {multiply_before_cast = true}
      output, scale = quantize_dynamic_scaled_float8(normed, scale_ub)

    Returns the FP8 quantized output and its per-token scale tensor. Produced by
    the RMS-norm + quantize fusion pattern (GPU only).
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$weight,
    MO_Tensor:$epsilon,
    MO_Tensor:$weightOffset,
    MO_Tensor:$scaleUb
  );
  let results = (outs MO_Tensor:$output, MO_Tensor:$scale);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// RMSNormFusedQuantizeDynamicBlockScaledOp
//===----------------------------------------------------------------------===//

def MOComposite_RMSNormFusedQuantizeDynamicBlockScaledOp
    : MO_CompositeOp<"rms_norm_fused_quantize_dynamic_block_scaled", [
  Pure,
  MO_SingleDeviceWithHostOperands<["epsilon", "weightOffset"]>,
]> {
  let summary = "Fused RMS norm + dynamic block-scaled MXFP8 quantize";
  let description = [{
    Fused operation computing block-scaled MXFP8 quantization of an
    RMS-normalized input:

      normed = rms_norm(input, weight, epsilon, weight_offset)
                 {multiply_before_cast = true}
      output, scale = quantize.dynamic.block.scaled(normed)
                        {SF_VECTOR_SIZE}

    Returns the E4M3 quantized output and its E8M0 per-block scale tensor in the
    rank-5 interleaved layout the SM100 block-scaled GEMM consumes. `output` and
    `scale` are numerically identical to the unfused
    `rms_norm -> quantize.dynamic.block.scaled` (MXFP8) pipeline. The fusion
    pattern attaches the block size as an `SF_VECTOR_SIZE` attribute (32 for
    MXFP8), which the MOGG lowering forwards to the kernel as a compile-time
    parameter. Produced by the RMS-norm + MXFP8-block-quantize fusion pattern
    (GPU only).
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$weight,
    MO_Tensor:$epsilon,
    MO_Tensor:$weightOffset
  );
  let results = (outs MO_Tensor:$output, MO_Tensor:$scale);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// MatmulFusedPartialRMSNormOp
//===----------------------------------------------------------------------===//

def MOComposite_MatmulFusedPartialRMSNormOp
    : MO_CompositeOp<"matmul_fused_partial_rms_norm", [
  Pure,
  MO_SingleDeviceWithHostOperands<["epsilon", "weightOffset"]>,
]> {
  let summary = "Fused matmul + partial RMS norm";
  let description = [{
    Fused operation computing a matmul followed by an RMS norm of the result,
    returning both the normalized output and the un-normalized matmul output:

      unnormed = input_a @ input_b   {transpose_b = true}
      normed   = rms_norm(unnormed, gamma, epsilon, weight_offset)
                   {multiply_before_cast = true}

    The kernel hardcodes `transpose_b = true` and `multiply_before_cast = true`.
    Produced by the matmul + partial-RMS-norm fusion pattern (GPU only).
  }];

  let arguments = (ins
    MO_Rank2Tensor:$input_a,
    MO_Rank2Tensor:$input_b,
    MO_Tensor:$gamma,
    MO_Tensor:$epsilon,
    MO_Tensor:$weightOffset
  );
  let results = (outs MO_Tensor:$normed, MO_Tensor:$unnormed);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// DistributedMatmulReduceScatterSumOp
//===----------------------------------------------------------------------===//

def MOComposite_DistributedMatmulReduceScatterSumOp : MO_DistributedOp<"composite.distributed.matmul_reduce_scatter.sum", [
  MO_MultiDeviceFrom<"inputsA">,
  MO_CompositeOpTrait
]> {
  string cppNamespace = MODialect.cppNamespace # "::Composite";
  let summary = "Fused Matmul + ReduceScatter sum across N devices.";
  let description = [{
    Each device computes a matmul (A_i @ B_i^T) and the results are
    reduce-scattered across devices along the M axis. On Blackwell GPUs
    this maps to a fused kernel that overlaps compute with cross-GPU TMA
    reductions; on other targets it falls back to separate matmul + reduce-scatter.

    All A inputs must be rank-2 with the same shape, and all B inputs must be
    rank-2 with the same shape. The matmul is always performed with B transposed
    (K-major layout).

    When `has_residual` is true, the `residual` tensor is added to the matmul
    output on the device it resides on before the reduce-scatter communication.
    The `residual_peer` attribute identifies which input index (and thus
    device) holds the residual; the add is applied only on that peer so the
    reduce-scatter sum yields `sum_j(A_j @ B_j) + residual`.
    This fuses the `add(residual, matmul) -> reduce_scatter` pattern from
    tensor-parallel attention (e.g. DeepseekV3/KimiK2.5 with TP+EP).
    When `has_residual` is false, `residual` and `residual_peer` are ignored.
  }];
  let hasVerifier = 1;
  let arguments = (ins
    Variadic<MO_Tensor>:$inputsA,
    Variadic<MO_Tensor>:$inputsB,
    MO_Tensor:$residual,
    DefaultValuedAttr<BoolAttr, "false">:$has_residual,
    DefaultValuedAttr<I64Attr, "0">:$residual_peer,
    Variadic<MO_Buffer>:$signalBuffers,
    MO_Chain:$inChain
  );
  let results = (outs
    Variadic<MO_Tensor>:$outputs,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// DistributedReduceScatterRMSNormOp
//===----------------------------------------------------------------------===//

def MOComposite_DistributedReduceScatterRMSNormOp : MO_DistributedOp<"composite.distributed.reduce_scatter_rms_norm", [
  MO_MultiDevice,
  MO_CompositeOpTrait
]> {
  string cppNamespace = MODialect.cppNamespace # "::Composite";
  let summary = "Fused ReduceScatter sum + RMSNorm across N devices for a single device.";
  let description = [{
    ReduceScatter takes in inputs each coming from a different device and
    partitions the reduction so each device receives a disjoint row shard of the
    sum. This op keeps that shard's sum in f32 registers and RMSNorm-normalizes
    it in the same launch, avoiding the HBM round-trip of a separate norm kernel.

    It returns both the normed shard (`output`, fed to the next layer) and the
    reduce-scatter sum shard (`outResidual`, the residual stream). The norm is
    inherently `multiply_before_cast=true`: gamma is folded in f32 and the value
    is cast to the input dtype once, last. bf16 in/out only (no quantization).
  }];
  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Variadic<MO_Buffer>:$signalBuffers,
    Variadic<MO_Tensor>:$gamma,
    Variadic<MO_RankedScalarFloatTensor>:$epsilon,
    Variadic<MO_RankedScalarFloatTensor>:$weightOffset,
    MO_Chain:$inChain
  );
  let results = (outs
    Variadic<MO_Tensor>:$output,
    Variadic<MO_Tensor>:$outResidual,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

// Grouped-matmul MoE FFN composite ops (NVFP4 expert-parallel path)
//===----------------------------------------------------------------------===//
//
// First-class composite versions of the two grouped-matmul custom ops that make
// up an NVFP4 MoE FFN: the gate-up GEMM + SwiGLU + bf16->nvfp4 quant (L1), and
// the down GEMM (L2). These are emitted directly by the model graph (in place of
// the `mo.composite.grouped_matmul_swiglu_nvfp4` /
// `mo.composite.grouped_matmul_block_scaled` `mo.custom` ops) so the MegaFFN
// fusion below can match typed ops with named
// operands instead of string-keyed customs. Each lowers 1:1 to the same kernel,
// re-registered under the composite op name.
//
// Routing operands (`expert_start_indices`, `expert_ids`, `a_scale_offsets`) and
// the expert-usage scalars are produced on-device by the EP gate; the kernel's
// host contract (front-packed active experts, `num_experts <= 64`,
// `ids[e] >= 0 => tokens[e] > 0`) is an upstream guarantee, not verified here.

def MOComposite_GroupedMatmulSwigluNvfp4Op
    : MO_CompositeOp<"grouped_matmul_swiglu_nvfp4", [
  Pure,
  MO_SingleDeviceWithHostOperands<[
    "estimated_total_m", "num_active_experts", "swiglu_alpha",
    "swiglu_limit"]>,
]> {
  let summary = "Grouped gate-up GEMM + SwiGLU + bf16->nvfp4 quantize (MoE L1)";
  let description = [{
    The gate-up leg of an NVFP4 MoE FFN: a per-expert (ragged) grouped matmul of
    the packed-NVFP4 activations by the gate-up weights, followed by a SwiGLU
    activation and a bf16->nvfp4 quantization of the result. Returns the packed
    NVFP4 activations for the down leg (`c_packed`) and their per-expert SwiGLU
    scale tile (`c_swiglu_scales`).

    `swiglu_alpha`/`swiglu_limit` are host scalars parameterizing the clamped
    SwiGLU (swigluoai) activation, enabled by the `clamp_activation` attribute.

    Composite form of the `mo.composite.grouped_matmul_swiglu_nvfp4` kernel;
    lowers 1:1.
  }];

  let arguments = (ins
    MO_Tensor:$hidden_states,
    MO_Tensor:$weight,
    MO_Tensor:$a_scales,
    MO_Tensor:$b_scales,
    MO_Tensor:$expert_start_indices,
    MO_Tensor:$expert_ids,
    MO_Tensor:$a_scale_offsets,
    MO_Tensor:$expert_scales,
    MO_Tensor:$c_input_scales,
    MO_Tensor:$estimated_total_m,
    MO_Tensor:$num_active_experts,
    MO_Tensor:$swiglu_alpha,
    MO_Tensor:$swiglu_limit,
    DefaultValuedAttr<BoolAttr, "false">:$clamp_activation
  );
  let results = (outs MO_Tensor:$c_packed, MO_Tensor:$c_swiglu_scales);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOComposite_GroupedMatmulBlockScaledOp
    : MO_CompositeOp<"grouped_matmul_block_scaled", [
  Pure,
  MO_SingleDeviceWithHostOperands<["estimated_total_m", "num_active_experts"]>,
]> {
  let summary = "Grouped block-scaled down GEMM (MoE L2)";
  let description = [{
    The down leg of an NVFP4 MoE FFN: a per-expert (ragged) block-scaled grouped
    matmul of the packed-NVFP4 activations by the down weights, producing the
    bf16 hidden-state output.

    Composite form of the `mo.composite.grouped_matmul_block_scaled` kernel;
    lowers 1:1.
  }];

  let arguments = (ins
    MO_Tensor:$hidden_states,
    MO_Tensor:$weight,
    MO_Tensor:$a_scales,
    MO_Tensor:$b_scales,
    MO_Tensor:$expert_start_indices,
    MO_Tensor:$expert_ids,
    MO_Tensor:$a_scale_offsets,
    MO_Tensor:$expert_scales,
    MO_Tensor:$estimated_total_m,
    MO_Tensor:$num_active_experts
  );
  let results = (outs MO_Tensor:$output);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// MegaFFNNvfp4Op
//===----------------------------------------------------------------------===//

def MOComposite_MegaFFNNvfp4Op : MO_CompositeOp<"mega_ffn_nvfp4", [
  // NOT `Pure`: the op writes the persistent `arrival_count` pool-slot buffer
  // in place (cross-CTA scheduler hand-off), so it carries a memory effect and
  // threads a chain. The `MO_MutableOpInterface` + chain interface methods
  // auto-derive from the `inChain`/`outChain`-named operand/result (see
  // MOInterfaces.td default impls); `MemoryEffectsOpInterface::getEffects` is
  // hand-written in MOOps.cpp (a Write on `arrival_count`).
  DeclareOpInterfaceMethods<MO_MutableOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  MO_SingleDeviceWithHostOperands<[
    "estimated_total_m", "gate_up_num_active_experts",
    "down_num_active_experts"]>,
]> {
  let summary = "Fused NVFP4 MoE FFN (gate-up + SwiGLU + quant + down) in one "
                "kernel";
  let description = [{
    Single-kernel NVFP4 MoE FFN: collapses the gate-up GEMM + SwiGLU +
    bf16->nvfp4 quantize leg (`mo.composite.grouped_matmul_swiglu_nvfp4`) and the
    block-scaled down GEMM leg (`mo.composite.grouped_matmul_block_scaled`) into
    one launch with an in-kernel cross-CTA hand-off between the two legs.
    Produced by the MegaFFN fusion pattern; the intermediate packed activations
    and SwiGLU scales are kept entirely on-chip and are not graph values.

    B200 (SM100) only. The fusion pattern fires on SM100 targets when enabled;
    on any other target the unfused two-composite chain is preserved.

    Operands are the union of the two legs' inputs (minus the dead intermediate):
    the gate-up leg's `hidden_states`/`weight`/`a_scales`/`b_scales`/
    `expert_scales`/`c_input_scales`, the down leg's `weight`/`b_scales`/
    `expert_scales`, the shared routing tensors (`expert_start_indices`,
    `expert_ids`, `a_scale_offsets`, `estimated_total_m`), and each leg's
    `num_active_experts`.

    The `arrival_count` buffer holds the cross-CTA pool-slot arrival counters
    the kernel uses to hand off between the two legs. It is a PERSISTENT device
    buffer minted once by the fusion pattern (`mo.buffer.create` with a zero
    `initValue`) and initialized once at model load; under the kernel's
    `POST_SELF_CLEAN_UP` policy every touched slot is reset in band, so the
    buffer is all-zero at every launch boundary (correct under single-stream
    serialization). `inChain`/`outChain` sequence the in-place write (required
    for any op with an `mo.buffer` operand; the chain is GC-internal and is
    dropped before the kernel-arg binding, so `arrival_count` binds to the
    registration's `MutableInputTensor` arg).

    The clamped-SwiGLU (`swigluoai`) activation is selected by the discardable
    `clamp_activation` attribute; the kernel supplies the canonical swigluoai
    alpha/limit constants when it is set.
  }];

  let arguments = (ins
    MO_Tensor:$hidden_states,
    MO_Tensor:$gate_up_weight,
    MO_Tensor:$gate_up_a_scales,
    MO_Tensor:$gate_up_b_scales,
    MO_Tensor:$down_weight,
    MO_Tensor:$down_b_scales,
    MO_Tensor:$expert_start_indices,
    MO_Tensor:$expert_ids,
    MO_Tensor:$a_scale_offsets,
    MO_Tensor:$gate_up_expert_scales,
    MO_Tensor:$down_expert_scales,
    MO_Tensor:$c_input_scales,
    MO_Tensor:$estimated_total_m,
    MO_Tensor:$gate_up_num_active_experts,
    MO_Tensor:$down_num_active_experts,
    MO_Buffer:$arrival_count,
    MO_Chain:$inChain
  );
  let results = (outs MO_Tensor:$output, MO_Chain:$outChain);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOCOMPOSITEOPS_TD
