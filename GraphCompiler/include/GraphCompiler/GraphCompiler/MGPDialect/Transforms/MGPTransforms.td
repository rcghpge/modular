//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPTRANSFORMS_TD
#define GENERICML_MGPDIALECT_MGPTRANSFORMS_TD

include "mlir/Pass/PassBase.td"
include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPOps.td"

//===----------------------------------------------------------------------===//
// AsyncDataflowDependenciesTestPass
//===----------------------------------------------------------------------===//

def AsyncDataflowDependenciesTestPass : Pass<"mgp-async-dataflow-dependencies-test", "MGP::ModelOp"> {
  let summary = "Prints the sameAs and impliesReady relations for all pairs of "
                "values in the input model, leaving the model unchanged. "
                "For testing only.";
}

//===----------------------------------------------------------------------===//
// LowerAlignmentsPass
//===----------------------------------------------------------------------===//

def LowerAlignmentsPass : Pass<"mgp-lower-alignments", "MGP::ModelOp"> {
  let summary = "Lowers alignment information to the kgen op from MGP";
  let description = [{
    Alignment can be modified by kernel or operations. This pass will
    have heuristics which help determine when alignment from MGP buffers
    are potentially modified and provide additional information in kgen for this.
  }];
}

//===----------------------------------------------------------------------===//
// SimplifyExtendLifetimesPass
//===----------------------------------------------------------------------===//

def SimplifyExtendLifetimesPass : Pass<"mgp-simplify-extend-lifetimes", "MGP::ModelOp"> {
  let summary = "Simplify extend lifetime ops";
  let description = [{
    Given:
    ```mlir
      %ch1 = mogg.generic.compiled.region(...) -> !mgp.chain
      %ch2 = mgp.extend.lifetime %ch1, %t1, ... %tn
      op(%ch2)
    ```
    We can fuse these into simply:
    ```mlir
      %ch1 = mogg.generic.compiled.region(..., %t1, ... %tn) -> !mgp.chain
      op(%ch1)
    ```
    Provided they are not then removed for being unused. So we must run this pass last.
  }];
}

//===----------------------------------------------------------------------===//
// SimplifyChainsPass
//===----------------------------------------------------------------------===//

def SimplifyChainsPass : Pass<"mgp-simplify-chains", "MGP::ModelOp"> {
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

  let statistics = [
    Statistic<"numEliminatedChains", "num-eliminated-chains",
              "the number of chain operands eliminated">,
  ];
}

//===----------------------------------------------------------------------===//
// DeferAllocsPass
//===----------------------------------------------------------------------===//

def DeferAllocsPass : Pass<"mgp-defer-allocs", "MGP::ModelOp"> {
  let summary = "Defers allocations until the producing op is ready to execute.";
  let description = [{
    Ensures mgp.buffer.alloc.* operations are deferred till they are
    about to be written to. Eg:
    ```mlir
      %buf = mgp.buffer.alloc %bufChain, 24
      my_op(%opChain, %buf, ...)
      // where %buf is in an 'output' position
    ```
    becomes:
    ```
      %ch = grt.chain.create (%opChain, %bufChain)
      %buf = mgp.buffer.alloc %ch, 24
      my_op(%ch, %buf, ...)
    ```
    Assumes:
      - Ops with buffer operands have "numOutputBuffers" attributes.
  }];
}

//===----------------------------------------------------------------------===//
// ConcatInPlacePass
//===----------------------------------------------------------------------===//

def ConcatInPlacePass : Pass<"mgp-concat-in-place", "MGP::ModelOp"> {
  let summary = "Replaces `mgp.buffer.concat` with `mgp.buffer.split.borrow`.";
  let description = [{
    This pass replaces `mgp.buffer.concat` with `mgp.buffer.split.borrow`,
    provided all writers to the operand buffers of the concat can be rewritten
    to instead write to the result buffers of the split.

    For example:
    ```mlir
    %in1_buf = mgp.buffer.alloc ... : !mgp.buffer<size1>
    %in1_ch = ... writer to %in1_buf ...
    %in2_bytes = ...
    %in2_buf = mgp.buffer.alloc ..., %in2_bytes
    %in2_ch = ... writer to %in2_buf ...
    %out_bytes = ...
    %out_buf = mgp.buffer.alloc ..., %out_bytes
    %ch1 = mgp.chain.create (%in1_ch, %in2_ch)
    %ch2 = mgp.buffer.concat %ch1, %out_buf, %in1_buf, %in2_buf : ...
    // where these are the only uses of %in1_buf and %in2_buf
    ```
    is rewritten to:
    ```mlir
    %out_bytes = ...
    %out_buf = mgp.buffer.alloc %root_ch, %out_bytes
    %in2_spec = ...
    %in2_size = ...
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

def ExecInvariantCodeMotionPass : Pass<"mgp-exec-invariant-code-motion", "MGP::ModelOp"> {
  let summary = "Moves code from the 'execute' to the 'init' blocks.";
  let description = [{
    This pass moves computations which do not depend on the 'execute' block
    arguments (directly or indirectly) to the 'init' block, using slots in
    the `!mgp.context` to forward the results, subject to heuristics to:
      - avoid moving 'trivial' code, where the cost of caching and retrieving
        the result outweighs the cost of the computation itself;
      - avoid moving code whose result is awkward to forward within the
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

      2. We want to lift ops that modify the lifted buffers into the init
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
// FixupOverhangsPass
//===----------------------------------------------------------------------===//

def FixupOverhangsPass : Pass<"mgp-fixup-overhangs", "MGP::ModelOp"> {
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

def PredicateAssertDependenciesPass : Pass<"mgp-predicate-asserts-deps", "MGP::ModelOp"> {
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

def VerifyModelArgumentsPass : Pass<"mgp-verify-model-arguments", "MGP::ModelOp"> {
  let summary = "Ensures that user provided model arguments match the static "
                "expectations";
  let description = [{
    This pass will add runtime shape comparison checks to dynamically verify
    that the tensors a user provides match the statically known expectations of
    the model arguments.

    Example:
    ```mlir
    %known = index.constant 4
    %dim = mgp.tensor_spec.get_dim[0] %arg2 : !mgp.tensor_spec<4xf32>
    %valid = index.cmp eq(%dim, %known)
    %ch = mgp.assert %valid, "Tensor type mismatch for argument 0"
    ```
  }];
  let dependentDialects = ["KGEN::POP::POPDialect"];
}

//===----------------------------------------------------------------------===//
// PrepareDevicePrimitivesPass
//===----------------------------------------------------------------------===//

def PrepareDevicePrimitivesPass : Pass<"mgp-prepare-device-primitives"> {
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
      %buf = mgp.buffer.alloc %ch, %size: !mgp.buffer<<"gpu", 3>>
    ```
    becomes:
    ```mlir
      %buf = "mgp.buffer.alloc.gpu" (%ch, %ctx)
               { aRuntimeSlot = 1 : ui64,
                 bRawAlign = kUnknownSize,
                 cDevice = 3 : ui64 }
               : (!mgp.chain<<"gpu", 3>>, !mgp.context) ->
                 !mgp.buffer<<"gpu", 3>>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ExplicitBufferLifetimesPass
//===----------------------------------------------------------------------===//

def ExplicitBufferLifetimesPass : Pass<"mgp-buffer-lifetimes", "MGP::ModelOp"> {
  let summary = "Inserts mgp.buffer.free ops based on the results of MGPBufferLifetimeAnalysis";
  let description = [{
    This pass tries to insert an mgp.buffer.free op for each mgp.alloc
    at a location in the graph where it is guaranteed that there will be no more
    accesses to the buffer. If there is no such location before the end of the
    graph, no free will be inserted.

    Note that the buffer.free ops do not actually free the buffer. Buffer memory
    management is handled by refcounting in the runtime. This pass exists purely
    as a way to debug and validate BufferLifetimeAnalysisPass.
  }];
}

//===----------------------------------------------------------------------===//
// SimpleBufferReusePass
//===----------------------------------------------------------------------===//

def SimpleBufferReusePass : Pass<"mgp-simple-buffer-reuse", "MGP::ModelOp"> {
  let summary = "Reuses allocated buffers in simple cases";
  let description = [{
    This pass replaces a mgp.buffer.alloc op B with a
    mgp.buffer.alloc op A if the following conditions are met:
      1) all uses of A are guaranteed to execute before the first use of B
      2) A has the same runtime size SSA value as B
      3) A has the same input chain as B
      4) A has the same device as B

    For example:
    ```mlir
      %input = ...
      %inChain = ...
      %size = ...
      %buf0 = mgp.buffer.alloc %inChain, %size, ...
      %buf1 = mgp.buffer.alloc %inChain, %size, ...
      %buf2 = mgp.buffer.alloc %inChain, %size, ...

      %outChain_0 = use(%inChain, %input, %buf0)
      %outChain_1 = use(%outChain_0, %buf0, %buf1)
      %outChain_2 = use(%outChain_1, %buf1, %buf2)
    ```
    becomes:
    ```mlir
      %input = ...
      %inChain = ...
      %size = ...
      %buf0 = mgp.buffer.alloc %inChain, %size, ...
      %buf1 = mgp.buffer.alloc %inChain, %size, ...

      %outChain_0 = use(%inChain, %input, %buf0)
      %outChain_1 = use(%outChain_0, %buf0, %buf1)
      %outChain_2 = use(%outChain_1, %buf1, %buf0)
    ```

  }];
}

//===----------------------------------------------------------------------===//
// DynamicMemoryPlanningPass
//===----------------------------------------------------------------------===//

def DynamicMemoryPlanningPass : Pass<"mgp-dynamic-memory-planning", "MGP::ModelOp"> {
  let summary = "Dynamic (runtime) memory planning";
  let description = [{
    Uses BufferLifetimeAnalysisPass to compute lifetimes of all buffers, then
    inserts BufferPlanOp to do runtime memory planning. The high watermark
    returned by BufferPlanOp is used to make a single large allocation, and all
    former allocations are replaced with slices (BufferSliceOp) into the single
    large allocation.
  }];
}

//===----------------------------------------------------------------------===//
// AttachPythonStackTraces
//===----------------------------------------------------------------------===//

def AttachPythonStackTraces : Pass<"mgp-attach-python-stack-traces", "mlir::ModuleOp"> {
  let summary = "Attaches Python stack traces from OpaqueLoc to ops and asserts";
  let description = [{
    Walks all ops in the module. For each op with an OpaqueLoc wrapping a
    PythonFrameAttr, encodes the Python stack trace as FusedLoc metadata
    (so it survives downstream lowering) and strips the OpaqueLoc. Also
    embeds the trace directly into the message of any mgp.assert op for
    better runtime diagnostics.
  }];
}

//===----------------------------------------------------------------------===//
// GlobalizeSlotIndicesPass
//===----------------------------------------------------------------------===//

def GlobalizeSlotIndicesPass : Pass<"mgp-globalize-slot-indices", "mlir::ModuleOp"> {
    let summary = "Globalizes slot indices across all models in the module";
    let description = [{
     Globalizes all slot indices used across all `mgp.model` operations within the module.
     This pass iterates through the models in their definition order, assigns a unique
     range of slot indices to each model based on the cumulative count of slots from
     previously processed models, and updates the slot indices used within each model
     accordingly. Finally, it updates the `num_slots` attribute of *all*
     `mgp.context.create` operations to reflect the total number of slots required
     across the entire module.

     This ensures that slot indices are globally unique, preventing conflicts when models
     are potentially combined or inlined later.

     Example:

     BEFORE:
     ```mlir
     module {
       mgp.model @sub1 init {
         %ctx = mgp.context.create 2 // Uses slots 0, 1
         %rt = mgp.runtime.create %chain, %ctx[0]
         %c = mgp.buffer.cache %chain, %ctx[1], %buf1
         ...
       } execute {
         %b = mgp.buffer.get_cached %chain, %ctx[1]
         ...
       }
       mgp.model @sub2 init {
         %ctx = mgp.context.create 1 // Uses slot 0
         %rt = mgp.runtime.create %chain, %ctx[0]
         ...
       } execute { ... }
     }
     ```

     AFTER:
     ```mlir
     module {
       mgp.model @sub1 init {
         // Total slots updated to 3 (2 from @sub1 + 1 from @sub2)
         %ctx = mgp.context.create 3
         // Slots for @sub1 start at 0 (no offset)
         %rt = mgp.runtime.create %chain, %ctx[0]
         %c = mgp.buffer.cache %chain, %ctx[1], %buf1
         ...
       } execute {
         %b = mgp.buffer.get_cached %chain, %ctx[1]
         ...
       }
       mgp.model @sub2 init {
         // Total slots updated to 3
         %ctx = mgp.context.create 3
         // Slots for @sub2 start at 2 (offset by @sub1's 2 slots)
         %rt = mgp.runtime.create %chain, %ctx[2]
         ...
       } execute { ... }
     }
     ```
    }];
}

//===----------------------------------------------------------------------===//
// InlineMGPCallsPass
//===----------------------------------------------------------------------===//

def InlineMGPCallsPass : Pass<"mgp-inline-mgp-calls", "mlir::ModuleOp"> {
  let summary = "Inlines the `execute` block of subgraphs at `mgp.call` sites";
  let description = [{
    Inline `mgp.call` operations into their calling `mgp.model` operations.

    This pass constructs a call graph of `mgp.model` operations within a module.
    It performs a post-order traversal (reverse topological sort) of the call
    graph to determine a safe inlining order, ensuring that callees are inlined
    before their callers. Recursion is detected and reported as an error, as it
    is not supported for inlining.

    For each `mgp.call` operation encountered during the traversal of the calling
    model's `init` and `execute` regions:
    1.  **Init Block Inlining (Once per Callee):** If the callee's `init` block
        has not been inlined into the caller's `init` block yet, its operations
        (excluding `mgp.context.create` and the terminator) are cloned into the
        caller's `init` block before the terminator. Arguments (weights registry,
        devices, context) are mapped appropriately.
    2.  **Execute Block Inlining:** Operations from the callee's `execute` block
        (excluding the terminator) are cloned at the call site. Operands of the
        call are mapped to the callee's block arguments, and results are mapped
        back.
    3.  The `mgp.call` operation is replaced with the results of the inlined
        execute block and then erased.

    After all possible calls have been inlined, the pass attempts to erase any
    `mgp.model` operations that were targets of inlining and now have no remaining
    uses.
  }];
}

//===----------------------------------------------------------------------===//
// OutlineModelsToCallables
//===----------------------------------------------------------------------===//

def OutlineModelsToCallables : Pass<"mgp-outline-models-to-callables", "mlir::ModuleOp"> {
  let summary = "Converts all MGP::ModelOps into component MEF::CallableOps and wrapping MEF::ModelOps";
  let description = [{
    Lowers all `MGP::ModelOps` into a structure directly executable by GraphRT,
    including subgraphs.

    This pass has two major conceptual steps:
    1) Propagate prefix handling through model call graph
    2) Take the (now, and progressively moreso) denormalized `mgp.model` ops
        and lower them fully to a series of:
        a) `mef.model` ops that represent entry points to the module,
            and that package the init and exec regions in a form
            our executor can interpret without too much magic, and
        b) `mef.callable` ops holding the component regions
            of the model, which can be directly called by
            `grt.invoke` ops in other regions.

    This allows for reuse of subgraphs without fully inlining and
    monomorphising their contents. Instead, subgraphs are
    reused at runtime by the executor across all their invocations.

    A `mef.model` is used to encapsulate this new form with a relaxed
    relationship between `init` and `exec`. The only requirement
    that is placed on that relationship is that the output
    of init be a prefix to the input to exec, and that the
    remainder of the input to exec match the input signature
    for the model itself.

    Example:

    BEFORE:
    ```mlir
    module {
      mgp.model @exp subgraph : mof ...
      init (/* weights registry, device contexts */...) {
        %0 = grt.chain.create ()
        %1 = mgp.context.create 2
        %2 = mgp.runtime.create %0, %1[0]
        %3 = mgp.module.load %2, %1[0] [0], ...
        %4 = mgp.string.create "foo" : !mgp.opaque<"mojo_string">
        %5 = mgp.buffer.constant.external %3, %arg0[%4, 80, align 16 placeholder true] ...
        %6 = mgp.buffer.cache %0, %1[1], %5
        mef.output %6, %1, %arg0, %arg1
      } execute (/* implicits */..., %arg4: !mgp.tensor<4x5xf32>, %arg5: index)
      -> (!mef.chain, !mgp.tensor<4x5xf32>) {
        ...
        mef.output %arg0, %3 : !mef.chain, !mgp.tensor<4x5xf32>
      }
      mgp.model @main : mof
        {argument_device_indices = [0 : ui32], result_device_indices = [0 : ui32]}
        devices [#M.device_ref<"cpu", 0>]
        init (/* weights registry, device contexts */)
      {
        %0 = grt.chain.create ()
        %1 = mgp.context.create 1
        %2 = mgp.runtime.create %0, %1[0]
        %3 = mgp.module.load %2, %1[0] [0], ...
        mef.output %3, %1, %arg0, %arg1
      } execute (/* implicits */..., %arg4: !mgp.tensor<4x5xf32>)
      -> (!mef.chain, !mgp.tensor<4x5xf32>) {
        ...
        %1:2 = mgp.call @exp(%arg0, %arg1, %arg2, %arg3, %arg4, %0) {..., prefix = "prefix"} ...
        mef.output %1#0, %1#1 : !mef.chain, !mgp.tensor<4x5xf32>
      }
    }
    ```

    AFTER:
    ```mlir
    module {
      mef.callable @exp$init(..., %arg2: !mgp.opaque<"mojo_string">) -> (
        !mef.chain,
        !mgp.context,
        !mgp.opaque<"weights_registry">,
        !mgp.device_context<<"cpu", 0>>,
        !mgp.opaque<"mojo_string">
      ) {
        %0 = grt.chain.create ()
        %1 = mgp.context.create 2
        %2 = mgp.runtime.create %0, %1[0] // note that subgraphs still
                                          // get their own context and runtime
        %3 = mgp.module.load %2, %1[0] [0], ...
        %4 = mgp.string.create "foo"
        %5 = mgp.string.concat %arg2, %4 // NEW
        %6 = mgp.buffer.constant.external %3,
            %arg0[%5, 80, align 16 placeholder true]
            // MODIFIED: uses concat'd prefix from %5
        %7 = mgp.buffer.cache %0, %1[1], %6
        mef.output %7, %1, %arg0, %arg1, %arg2
              : !mef.chain,
                !mgp.context,
                !mgp.opaque<"weights_registry">,
                !mgp.device_context<<"cpu", 0>>,
                !mgp.opaque<"mojo_string"> // added context to pass to exec
      }
      mef.callable @exp$exec(/* implicits */...,
        %arg4: !mgp.opaque<"mojo_string">,
        %arg5: !mgp.tensor<4x5xf32>,
        %arg6: index)
      -> (!mef.chain, !mgp.tensor<4x5xf32>) {
        ...
        mef.output %arg0, %3 : !mef.chain, !mgp.tensor<4x5xf32>
      }
      mef.callable @main$init(
        %arg0: !mgp.opaque<"weights_registry">,
        %arg1: !mgp.device_context<<"cpu", 0>>)
      -> (...) {
        %0 = grt.chain.create ()
        %1 = mgp.context.create 1
        %2 = mgp.runtime.create %0, %1[0]
        %3 = mgp.module.load %2, %1[0] [0], ...
        %4 = mgp.string.create "prefix" : !mgp.opaque<"mojo_string">
        %5:5 = grt.invoke @exp$init(%arg0, %arg1, %4)
        mef.output %3, %1, %arg0, %arg1, %5#0, %5#1, %5#2, %5#3, %5#4
            : !mef.chain,
              !mgp.context,
              !mgp.opaque<"weights_registry">,
              !mgp.device_context<<"cpu", 0>>, // main's forwarded implicits
              !mef.chain,
              !mgp.context,
              !mgp.opaque<"weights_registry">,
              !mgp.device_context<<"cpu", 0>>, // exp's forwarded implicits
              !mgp.opaque<"mojo_string"> // exp's SSA prefix for all placeholder
                                         // accesses to weights registry
      }
      mef.callable @main$exec(
        /* main's forwarded implicits */...,
        /* subgraph's forwarded implicits */...,
        %arg9: !mgp.tensor<4x5xf32>)
      -> (...) {
        %0 = "mgp.generic.execute"(%arg1) ...
        %1:2 = grt.invoke @exp$exec(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %0) ...
        mef.output %1#0, %1#1 : !mef.chain, !mgp.tensor<4x5xf32>
      }
      mef.model @main : mof ... {
        %0:9 = grt.invoke @main$init(%arg0, %arg1) : ...
        mef.output %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7, %0#8 : ...
      } execute (...) -> (...) {
        %0:2 = grt.invoke @main$exec(
          arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : ...
        mef.output %0#0, %0#1 ...
      }
    }
    ```
  }];
}

//===----------------------------------------------------------------------===//
// StageDeviceGraphsPass
//===----------------------------------------------------------------------===//

def StageDeviceGraphsPass :
    Pass<"mgp-stage-device-graphs", "MGP::ModelOp"> {
  let summary = "Stage the execute body of an `isDeviceGraph` model into a "
                "`mgp.device_graph.staging` op.";
  let description = [{
    Part of device-graph synthesis. For each `mgp.model` marked `isDeviceGraph`,
    moves every `mogg.generic.compiled.region` in the execute block into a single
    `mgp.device_graph.staging` op. The regions' free values become the op's
    inputs (entering the body as `!mogg.async<T>` block args, unpacked inside),
    and values produced inside but consumed outside become the op's results
    (packed for the async yield, external uses rewired). The staging op is an
    `IsolatedFromAbove`, transient container that later synthesis passes lower
    into `mgp.device_graph.create` / `mgp.device_graph.execute`.

    For models without `isDeviceGraph`, the pass is a no-op.
  }];
}

#endif // GENERICML_MGPDIALECT_MGPTRANSFORMS_TD
