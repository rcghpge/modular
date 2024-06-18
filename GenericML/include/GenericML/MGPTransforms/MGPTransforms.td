//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPTRANSFORMS_TD
#define GENERICML_MGPDIALECT_MGPTRANSFORMS_TD

include "mlir/Pass/PassBase.td"
include "GenericML/MGPDialect/MGPOps.td"

//===----------------------------------------------------------------------===//
// AsyncDataflowDependenciesTestPass
//===----------------------------------------------------------------------===//

def AsyncDataflowDependenciesTestPass : Pass<"async-dataflow-dependencies-test", "MGP::ModelOp"> {
  let summary = "Prints the sameAs and impliesReady relations for all pairs of "
                "values in the input model, leaving the model unchanged. "
                "For testing only.";
}

//===----------------------------------------------------------------------===//
// TensorExtractAnalysisTestPass
//===----------------------------------------------------------------------===//

def TensorExtractAnalysisTestPass : Pass<"tensor-extract-analysis-test", "MGP::ModelOp"> {
  let summary = "Prints the isTensorExtract predicate for all values in the "
                "input module, leaving the module unchanged. For unit testing "
                "only.";
}

//===----------------------------------------------------------------------===//
// CheckOverhangsPass
//===----------------------------------------------------------------------===//

def CheckOverhangsPass : Pass<"check-overhangs", "MGP::ModelOp"> {
  let summary = "Checks that all ops will have finished execution when model "
                "results are ready.";
  let description = [{
    This pass checks that every op will be definitely finished when a model's
    results are ready. That is, execution of an op cannot 'overhang' the
    apparent completion of the model itself.

    Overhanging ops are not memory safe since context resources they depend
    on may be torn down before the op has completed, or may be mutated by a
    subsequent call to 'execute'. Overhanging ops also make memory planning
    difficult since there's no indication of when a buffer operand is no longer
    in use.

    The most common cause of overhangs is ignoring the chain result of an op,
  }];
}

//===----------------------------------------------------------------------===//
// CheckBufferReadWritePass
//===----------------------------------------------------------------------===//

def CheckBufferReadWritePass : Pass<"check-buffer-read-write", "MGP::ModelOp"> {
  let summary = "Checks that the same buffer can never be concurrently read or "
                "written while another write is in progress.";
}

//===----------------------------------------------------------------------===//
// SimplifyChainsPass
//===----------------------------------------------------------------------===//

def SimplifyChainsPass : Pass<"simplify-chains", "MGP::ModelOp"> {
  let summary = "Removed unnecessary chains from grt.chain.create ops.";
  let description = [{
    Given:
    ```mlir
      %a = grt.chain.create (%x, %y, %z)
    ```
    where AsyncValueDependencies::impliesReady(%x, %y) then there's no need
    to include %y and we can rewrite to:
    ```mlir
      %a = grt.chain.create (%x, %z)
    ```

    If in addition AsyncValueDependencies::impliesReady(%x, %z) then we can
    simply replace %a for %x and remove the merge.

    If any operand is known to be isAlwaysReady then it can be removed.
  }];
}

//===----------------------------------------------------------------------===//
// StaticMemoryPlanerPass
//===----------------------------------------------------------------------===//

def StaticMemoryPlannerPass : Pass<"static-memory-planner", "ModuleOp"> {
  let summary = "Replace mgp.buffer.alloc.static.deprecated ops with "
                "mgp.buffer.section.static.borrow ops using statically "
                "determined offsets.";
  let description = [{
    This pass replaces mgp.buffer.alloc.{static,spec} ops with
    mgp.buffer.section.{static,dynamic}.borrow ops from (pre-allocated)
    shared buffers. The size of the shared buffers, and the offset to use for
    each mgp.buffer.section.{static,dynamic}.borrow op, is chosen by
    this pass so as to ensure no active buffers will ever overlap in their
    borrowed memory.

    Heuristics are used to decide when planning is worthwhile, and to choose
    offsets within the shared buffers.
  }];

  let options = [
    Option<"minBuffers", "min-buffers", "size_t", "10",
           "The minimum number of buffers within a planning group to "
           "consider static memory planning worthwhile. If zero then "
           "static memory planning is disabled. If one then no restriction.">,
    Option<"maxHighWaterMarkPercentage", "max-highwatermark-percentage",
           "double", "50.0",
           "The maximum high-watermark percentage to consider planning "
           "worthwhile. This percentage is the reduction in maximum memory "
           "usage using memory planned buffers as compared with simply "
           "pre-allocating every buffer at the beginning of execution. "
           "If zero then no restriction.">,
  ];
}

//===----------------------------------------------------------------------===//
// RemoveRedundantTensorExtractsPass
//===----------------------------------------------------------------------===//

def RemoveRedundantTensorExtractsPass : Pass<"remove-redundant-tensor-extracts", "ModuleOp"> {
  let summary = "Removes unnecessary tensor create/extract operations from "
                "the graph.";
  let description = [{
    This pass avoid unnecessary mgp.tensor.{create,extract} operations in
    two situations:

    1. If the result of a (mgp.cpu/cuda).tensor.create op is exclusively consumed by
       tensor.extract ops, remove them in favor of the original buffer,
       tensor spec and chain used by the tensor.create. Eg:
    ```mlir
          %tensor0 = mgp.tensor.create.deprecated %ch0, %inputBuf0, %inputSpec0
          %tensor1 = mgp.tensor.create.deprecated %ch1, %inputBuf1, %inputSpec1
          %buf0, %spec0 = mgp.tensor.extract.deprecated %tensor0
          %buf1, %spec1 = mgp.tensor.extract.deprecated %tensor1
          %ch3 = my.op(%ch2, ..., %buf0, %spec0, %buf1, %spec1)
          // where %buf0, %buf1 are in 'read' positions
    ```
       becomes:
    ```mlir
          %ch4 = grt.chain.create %ch2, %ch0, %ch1
          %ch3 = my.op(%ch4, ...,
                       %inputBuf0, %inputSpec0,  %inputBuf1, %inputSpec1)
    ```
       Assumes:
        - All buffers passed to tensor.create will not be further
          mutated once the chain to that op is ready. This is currently true
          for the GraphCompiler, but need not be true in general.
        - Ops have buffer annotations indicating which are in 'write' positions.

    2. Ensures values of type !(mgp.cpu/cuda).tensor not otherwise the result of an
       tensor.create op (eg block arguments or the results of other
       primitives) are extracted at most once. Eg:
    ```mlir
          execute(%tensor : !mgp.tensor) {
            %buf, %spec = mgp.tensor.extract.deprecated %tensor
            my.op(..., %buf, %spec, ...)
            %buf2, %spec2 = mgp.tensor.extract.deprecated %tensor
            my.op(..., %buf2, %spec2, ...)
          }
    ```
       becomes:
    ```mlir
          execute(%tensor : !mgp.tensor) {
            %buf, %spec = mgp.tensor.extract.deprecated %tensor
            my.op(..., %buf, %spec, ...)
            my.op(..., %buf, %spec, ...)
          }
    ```

    These rules apply across control flow boundaries. Eg:
    ```mlir
          %tensor0 = mgp.tensor.create.deprecated %ch0, %inputBuf0, %inputSpec0
          %ch1, %tensor1 = grt.if %cond (%ch0, %tensor0) -> ... {
            %buf0, %spec0 = mgp.tensor.extract.deprecated %tensor0
            %tensor1 = mgp.tensor.create.deprecated %ch0, %buf0, %spec0
            mef.output %ch0, %tensor1
          } else {
            ...
          }
          %buf1, %spec1 = mgp.tensor.extract.deprecated %tensor1
          %ch3 = my.op(%ch2, ..., %buf1, %spec1)
          // where %buf1 is in 'read' position
    ```
    becomes:
    ```mlir
          %ch1, %buf1, %spec1 = grt.if %cond (%ch0, %inputBuf0, %inputSpec0) -> ... {
            mef.output %ch0, %inputBuf0, %inputSpec0
          } else {
            ...
          }
          %ch4 = grt.chain.create %ch2, %ch1
          %ch3 = my_op(%ch4, ..., %buf1, %spec1)
          // where %buf1 is in 'read' position
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DeferAllocsPass
//===----------------------------------------------------------------------===//

def DeferAllocsPass : Pass<"defer-allocs", "MGP::ModelOp"> {
  let summary = "Defers allocations until the producing op is ready to execute.";
  let description = [{
    Ensures mgp.buffer.alloc.* operations are deferred till they are
    about to be written to. Eg:
    ```mlir
      %buf = mgp.buffer.alloc.static.deprecated %bufChain, 24
      my_op(%opChain, %buf, ...)
      // where %buf is in an 'output' position
    ```
    becomes:
    ```
      %ch = grt.chain.create (%opChain, %bufChain)
      %buf = mgp.buffer.alloc.static.deprecated %ch, 24
      my_op(%ch, %buf, ...)
    ```
    Assumes:
      - Ops with buffer operands have "numOutputBuffers" attributes.
  }];
}

//===----------------------------------------------------------------------===//
// DeferTensorCreateExecPass
//===----------------------------------------------------------------------===//

def DeferTensorCreateExecPass : Pass<"defer-tensor-create-exec", "MGP::ModelOp"> {
  let summary = "Defers execution of `mgp.tensor.create` until its buffer has "
                "no uses.";
  let description = [{
    Ensures `mgp.tensor.create` operations which are operands to the
    `mef.output` are deferred until their buffers have no other uses. In this
    way we avoid an additional and unnecessary buffer copy.

    For example:
    ```mlir
      %buf = mgp.buffer.alloc.static ...
      %ch1 = op1 (... out: %buf, ...)
      %tensor = mgp.tensor.create %ch1, %buf ...
      %ch2 = op2 (... in: %buf, ...)
      mef.output ... %tensor ...
    ```
    becomes:
    ```
      %buf = mgp.buffer.alloc.static ...
      %ch1 = op1 (... out: %buf, ...)
      %ch2 = op2 (... in: %buf, ...)
      %ch3 = mgp.chain.create (%ch1, %ch2)
      %tensor = mgp.tensor.create %ch3, %buf ...
      mef.output ... %tensor ...
    ```

    CAUTION: Does not look inside control flow ops, and does not track all
             possible buffer uses.
  }];
}

//===----------------------------------------------------------------------===//
// StripBufferAttributesPass
//===----------------------------------------------------------------------===//

def StripBufferAttributesPass : Pass<"strip-buffer-attributes", "MGP::ModelOp"> {
  let summary = "Removes numOutputBuffers attributes.";
  let description = [{
    Removes numOutputBuffers attributes. These may be added during lowering
    to help downstream passes distinguish between input and output buffer
    arguments, but are not needed thereafter.
  }];
}

//===----------------------------------------------------------------------===//
// ConcatInPlacePass
//===----------------------------------------------------------------------===//

def ConcatInPlacePass : Pass<"concat-in-place", "MGP::ModelOp"> {
  let summary = "Replaces `mgp.buffer.concat` with `mgp.buffer.split.borrow`.";
  let description = [{
    This pass replaces `mgp.buffer.concat` with `mgp.buffer.split.borrow`,
    provided all writers to the operand buffers of the concat can be rewritten
    to instead write to the result buffers of the split.

    For example:
    ```mlir
    %in1_buf = mgp.buffer.alloc.static ... : !mgp.buffer<size1>
    %in1_ch = ... writer to %in1_buf ...
    %in2_bytes = ...
    %in2_buf = mgp.buffer.alloc.dynamic ..., %in2_bytes
    %in2_ch = ... writer to %in2_buf ...
    %out_bytes = ...
    %out_buf = mgp.buffer.alloc.dynamic ..., %out_bytes
    %ch1 = mgp.chain.create (%in1_ch, %in2_ch)
    %ch2 = mgp.buffer.concat %ch1, %out_buf, %in1_buf, %in2_buf : ...
    // where these are the only uses of %in1_buf and %in2_buf
    ```
    is rewritten to:
    ```mlir
    %out_bytes = ...
    %out_buf = mgp.buffer.alloc.dynamic %root_ch, %out_bytes
    %in2_spec = ...
    %in2_size = mgp.tensor_spec.size %in2_spec
    %in1_buf, %in2_buf = mgp.buffer.split %root_ch, %out_buf, %in2_size : ...
    %in1_ch = ... writer to %in1_buf ...
    %in2_ch = ... writer to %in2_buf ...
    %ch2 = mgp.chain.create (%in1_ch, %in2_ch)
    ```

    Note that this pass simply assumes that the argument chain to the
    concat op will only be ready once all argument buffers are ready. In
    this way we simply replace uses of %ch2 with %ch1.
  }];
}

//===----------------------------------------------------------------------===//
// ExecInvariantCodeMotionPass
//===----------------------------------------------------------------------===//

def ExecInvariantCodeMotionPass : Pass<"exec-invariant-code-motion", "MGP::ModelOp"> {
  let summary = "Moves code from the 'execute' to the 'init' blocks.";
  let description = [{
    This pass moves computations which do not depend on the 'execute' block
    arguments (directly or indirectly) to the 'init' block, using slots in
    the `!mgp.context` to forward the results, subject to heuristics to:
      - avoid moving 'trivial' code, where the cost of caching and retrieving
        the result outweighs the cost of the computation itself;
      - avoid moving code who's result is awkward to forward within the
        `!mgp.context`.
      - avoid caching results which are now only needed within the 'init' block.

    Notes:
      1. We want to lift buffer-producing ops into the 'init' block. In order
         to reference such lifted buffers in the 'execute' block, we need to
         leverage the buffer cache mechanism, i.e., cache in the 'init' block,
         and look up the cache in the 'execute' block. Note that
          (a). If all users of a lifted buffer are also lifted, we don't cache
               such buffers, thereby minimizing the number of buffer cache ops
               inserted.
          (b). If no user of a buffer op is lifted, we don't lift the buffer op
               because it merely adds more caching without saving us any
               computation.

      2. We want to lift ops that modifies the lifted buffers into the init
         phase (e.g., matmul weights pre-packing). Usually such ops produces a
         chain; in order to preserve control dependency, we must:
          (a). Update the 'init' block to output a chain that accounts for the
               lifted chain.
          (b). Update the 'execute' block to use the input chain instead of the
               lifted chain.

      3. We want to lift ops that produce non-buffer outputs. In order to
         reference such lifted values in the 'execute' block, we need to perform
         re-computation, i.e., leave a clone of the computation inside the
         'execute' block, and lift the original computation (unless we introduce
         caching mechanism for these output types too). For instance, the output
         spec of matmul weight packing is used by both the lifted packing
         computation, and the actual matmul computation in the exec phase.

    For example:
    ```mlir
    mgp.model @example : mof init {
      %ch0 = grt.chain.create ()
      %ctx = mgp.context.create 1
      mef.output %ch0, %ctx : !mef.chain, !mgp.context

    } execute (%ch0: !mef.chain, %ctx: !mgp.context, %arg: !mgp.tensor<?xf32>) -> (!mef.chain, !mgp.tensor<?xf32>) {
      %const_buf     : !mgp.buffer
      %const_spec    : !mgp.tensor_spec<?xf32>
      %new_const_buf : !mgp.tensor_spec<?xf32>
      %new_const_spec = spec_update_op(%const_spec) : (!mgp.tensor_spec<?xf32>) -> !mgp.tensor_spec<?xf32>
      %kernel_ch1 = kernel_op1(%ch0, %const_buf, %const_spec, %new_const_buf, %new_const_spec) : ... -> mef.chain

      %out_spec : !mgp.tensor_spec<?xf32>
      %out_buf  : !mgp.buffer
      %kernel_ch2 = kernel_op2(%kernel_ch1, %arg_buf, %arg_spec, %new_const_buf, %new_const_spec, %out_buf, %out_spec) : ...
      %out_tensor = mgp.tensor.create.deprecated %kernel_ch2, %out_buf, %out_spec  : !mgp.tensor<?xf32>
      mef.output %ch0, %out_tensor : !mef.chain, !mgp.tensor<?xf32>
    }
    ```
    is rewritten to (note all the clarifying comments):
    ```mlir
    mgp.model @example : mof init {
      %ch0 = grt.chain.create ()
      // number of slots increased, to store lifted buffers
      %ctx = mgp.context.create 2

      // moved from exec phase
      %const_buf     : !mgp.buffer
      %const_spec    : !mgp.tensor_spec<?xf32>
      %new_const_buf : !mgp.tensor_spec<?xf32>
      %new_const_spec = spec_update_op(%const_spec) : ...
      %kernel_ch1 = kernel_op1(%ch0, %const_buf, %const_spec, %new_const_buf, %new_const_spec) : ... -> mef.chain

      // inserted; buffer.cache may perform a deep copy, so we must wait for the
      // actions on buffers to finish first.
      %buffer_action_ch = grt.merge (%kernel_ch1) // may have more action chain in general
      %cache_ch = mgp.buffer.cache %buffer_action_ch, %ctx[1], %new_const_buf : !mgp.buffer

      // out chain changed
      %out_ch = grt.merge (%cache_ch) // may have more cache_ch in general
      mef.output %out_ch, %ctx : !mef.chain, !mgp.context

    } execute (%ch0: !mef.chain, %ctx: !mgp.context, %arg: !mgp.tensor<?xf32>) -> (!mef.chain, !mgp.tensor<?xf32>) {
      // cloned
      %const_spec : !mgp.tensor_spec<?xf32>
      %new_const_spec = spec_update_op(%const_spec) : (!mgp.tensor_spec<?xf32>) -> !mgp.tensor_spec<?xf32>

      // inserted
      %new_const_buf = mgp.buffer.get_cached %ch0, %ctx[1] : !mgp.buffer

      // references to `%kernel_ch1` become `%ch0`
      %out_spec : !mgp.tensor_spec<?xf32>
      %out_buf  : !mgp.buffer
      %kernel_ch2 = kernel_op2(%ch0, %arg_buf, %arg_spec, %new_const_buf, %new_const_spec, %out_buf, %out_spec) : ... -> mef.chain
      %out_tensor = mgp.tensor.create.deprecated %kernel_ch2, %out_buf, %out_spec  : !mgp.tensor<?xf32>
      mef.output %ch0, %out_tensor : !mef.chain, !mgp.tensor<?xf32>
    }
    ```

    TODO(#24751): Possible improvements to this pass are tracked in this issue.
  }];
}

//===----------------------------------------------------------------------===//
// AddDeviceContextProfilingPass
//===----------------------------------------------------------------------===//

def AddDeviceContextProfilingPass : Pass<"device-context-profiling", "MGP::ModelOp"> {
  let summary = "If enabled, enable profiling using the Device Context";
  let description = [{
    This pass is intended as a stop-gap to allow device-context profiling to be controlled explicitly in the IR. This allows
    the pathways which won't use the max-driver to be able to dump statistics from the deviceContext. 
  }];
  let options = [
    Option<"enabled", "enabled", "bool", "false",
           "Profiling is by default off and inserted only if enabled by this specific flag">,
    Option<"filePath", "filepath", "std::string", "\"./prof.log\"",
           "Profiling log file-path, it is up to the deviceContext to append device-specific appends to filepath">,
  ];
}


//===----------------------------------------------------------------------===//
// FixupOverhangsPass
//===----------------------------------------------------------------------===//

def FixupOverhangsPass : Pass<"fixup-overhangs", "MGP::ModelOp"> {
  let summary = "If necessary, inserts additional chain dependences to ensure "
                "no overhanging chains in the graph.";
  let description = [{
    This pass checks for overhanging chains (i.e., chains that have no uses in
    the graph) and inserts dependencies on the graph output to ensure that the
    output waits on all previously hanging chains.
  }];
}

//===----------------------------------------------------------------------===//
// PredicateAssertDependencies
//===----------------------------------------------------------------------===//

def PredicateAssertDependenciesPass : Pass<"predicate-asserts-deps", "MGP::ModelOp"> {
  let summary = "Predicates the execution of operations that directly or "
                "indirectly depend the result of an assert operation.";
  let description = [{
    For each block in the IR, any operation that takes in a chain and uses a
    value that contributed to at least one assert will have to depend on the
    chains of those asserts.

    Before
    ------
    %0 = some_op
    %1 = mgp.assert %0
    some_user %ch, %0

    After
    ------
    %0 = some_op
    %1 = mgp.assert %0
    %ch2 = grt.chain.create %ch, %1
    some_user %ch2, %0

    Operations that do not take in a chain will ignore asserts meaning they no
    additional dependency is inserted. The result of those operations could
    still be dependent on the assert which would be invalid if the assertion
    fails.
  }];
}

//===----------------------------------------------------------------------===//
// VerifyModelArgumentsPass
//===----------------------------------------------------------------------===//

def VerifyModelArgumentsPass : Pass<"verify-model-arguments", "MGP::ModelOp"> {
  let summary = "Ensures that user provided model arguments match the static "
                "expectations";
  let description = [{
    This pass will add runtime shape comparison checks to dynamically verify
    that the tensors a user provides match the statically known expectations of
    the model arguments.

    Example:
    ```mlir
    %valid = mgp.tensor_spec.equal.static %arg2 : !mgp.tensor<2x4xf32>
    %ch = mgp.assert %valid, "Tensor type mismatch for argument 0"
    ```
  }];
}

//===----------------------------------------------------------------------===//
// PrepareDevicePrimitivesPass
//===----------------------------------------------------------------------===//

def PrepareDevicePrimitivesPass : Pass<"prepare-device-primitives", "MGP::ModelOp"> {
  let summary = "Prepares device-generic MGP primitives for execution by the "
                "GraphRT.";
  let description = [{
    This pass will:
      - rename device-generic MGP primitives to include the label
        of the primitive's 'main' device in the primitive name. In this way the
        GraphRT primitive dispatch mechanism can be used to support multiple
        devices without the need for any additional indirection.
      - replaces #M.device_ref attributes with just the device id. In this way
        we avoid reconstructing device labels at runtime which are ignored
        anyway.

    For example:
    ```mlir
      %buf = mgp.buffer.alloc.static %ch, %ctx[1] : !mgp.buffer<<"cuda", 3>, 124>
    ```
    becomes:
    ```mlir
      %buf = "mgp.buffer.alloc.static.cuda" (%ch, %ctx)
               { aRuntimeSlot = 1 : ui64,
                 bSize = 124 : ui64,
                 cRawAlign = kUnknownSize,
                 dDeviceId = 3 : ui64 }
               : (!mgp.chain<<"cuda", 3>>, !mgp.context) ->
                 !mgp.buffer<<"cuda", 3>, 124>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ExplicitBufferLifetimesPass
//===----------------------------------------------------------------------===//

def ExplicitBufferLifetimesPass : Pass<"buffer-lifetimes", "MGP::ModelOp"> {
  let summary = "Inserts mgp.buffer.free ops based on the results of MGPBufferLifetimeAnalysis";
  let description = [{
    This pass tries to insert an mgp.buffer.free op for each mgp.alloc.dynamic
    at a location in the graph where it is guaranteed that there will be no more
    accesses to the buffer. If there is no such location before the end of the
    graph, no free will be inserted.

    Note that the buffer.free ops do not actually free the buffer. Buffer memory
    management is handled by refcounting in the runtime. This pass exists purely
    as a way to debug and validate BufferLifetimeAnalysisPass.
  }];
}

#endif // GENERICML_MGPDIALECT_MGPTRANSFORMS_TD
