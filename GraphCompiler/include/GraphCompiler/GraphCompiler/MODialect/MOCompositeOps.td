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

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOCOMPOSITEOPS_TD
