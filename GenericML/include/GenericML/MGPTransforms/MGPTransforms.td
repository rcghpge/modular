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
// SimplifyChainsPass
//===----------------------------------------------------------------------===//

def SimplifyChainsPass : Pass<"simplify-chains", "MGP::ModelOp"> {
  let summary = "Removed unnecessary chains from grt.chain.merge ops.";
  let description = [{
    Given:
    ```mlir
      %a = grt.chain.merge (%x, %y, %z)
    ```
    where AsyncValueDependencies::impliesReady(%x, %y) then there's no need
    to include %y and we can rewrite to:
    ```mlir
      %a = grt.chain.merge (%x, %z)
    ```

    If in addition AsyncValueDependencies::impliesReady(%x, %z) then we can
    simply replace %a for %x and remove the merge.

    If any operand is known to be isAlwaysReady then it can be removed.
  }];
}

//===----------------------------------------------------------------------===//
// RemoveRedundantTensorExtractsPass
//===----------------------------------------------------------------------===//

def RemoveRedundantTensorExtractsPass : Pass<"remove-redundant-tensor-extracts", "MGP::ModelOp"> {
  let summary = "Removes redundant tensor extract operations from the graph.";
  let description = [{
    This pass avoid redundant mgp.cpu.tensor.extract operations in two
    situations:

    1. Ensures 'execute' block argument tensors are fed to at most one
       mgp.cpu.tensor.extract operation. Eg:
    ```mlir
          execute(%tensor : !mgp.cpu.tensor):
            %buf, %spec = mgp.cpu.tensor.extract %tensor
            my_op(..., %buf, %spec, ...)
            %buf2, %spec2 = mgp.cpu.tensor.extract %tensor
            my_op(..., %buf2, %spec2, ...)
    ```
       becomes:
    ```mlir
          execute(%tensor : !mgp.cpu.tensor):
            %buf, %spec = mgp.cpu.tensor.extract %tensor
            my_op(..., %buf, %spec, ...)
            my_op(..., %buf, %spec, ...)
    ```

    2. Removes mgp.cpu.tensor.create ops which are only subsequently consumed
       by mgp.cpu.tensor.extract ops. Eg:
    ```mlir
          %tensor0 = mgp.cpu.tensor.create %inputChain0, %inputBuf0, %inputSpec0
          %tensor1 = mgp.cpu.tensor.create %inputChain1, %inputBuf1, %inputSpec1
          %buf0, %spec0 = mgp.cpu.tensor.extract %tensor0
          %buf1, %spec1 = mgp.cpu.tensor.extract %tensor1
          %resultChain = my_op(%opChain, ..., %buf0, %spec0, %buf1, %spec1)
          // where %buf0, %buf1 are in 'input' positions
    ```
       becomes:
    ```mlir
          %ch = grt.chain.merge (%opChain, %inputChain0, %inputChain1)
          %resultChain = my_op(%ch, ...,
                               %inputBuf0, %inputSpec0,
                               %inputBuf1, %inputSpec1)
    ```
       Assumes:
        - All buffers passed to mgp.cpu.tensor.create will not be further
          mutated once the chain to that op is ready. This is currently true
          for the GraphCompiler, but need not be true in general.
        - Ops with buffer operands have "numOutputBuffers" attributes.
  }];
}

//===----------------------------------------------------------------------===//
// DeferAllocsPass
//===----------------------------------------------------------------------===//

def DeferAllocsPass : Pass<"defer-allocs", "MGP::ModelOp"> {
  let summary = "Defers allocations until the producing op is ready to execute.";
  let description = [{
    Ensures mgp.cpu.buffer.alloc.* operations are deferred till they are
    about to be written to. Eg:
    ```mlir
      %buf = mgp.cpu.buffer.alloc.static %bufChain, 24
      my_op(%opChain, %buf, ...)
      // where %buf is in an 'output' position
    ```
    becomes:
    ```
      %ch = grt.chain.merge (%opChain, %bufChain)
      %buf = mgp.cpu.buffer.alloc.static %ch, 24
      my_op(%ch, %buf, ...)
    ```
    Assumes:
      - Ops with buffer operands have "numOutputBuffers" attributes.
  }];
}

//===----------------------------------------------------------------------===//
// LookupCachedConstantsPass
//===----------------------------------------------------------------------===//

def LookupCachedConstantsPass : Pass<"lookup-cached-constants", "MGP::ModelOp"> {
  let summary = "Replace uses of #cache.constant_hash<TAG> "
                "with mgp.cas.lookup, mgp.cpu.buffer.cache and "
                "mgp.cpu.buffer.get_cached";
  let description = [{
    This pass replace uses of #cache.constant_hash<"TAG> with mgp.cas.lookup,
    mgp.cpu.buffer.cache and mgp.cpu.buffer.get_cached.

    For example:
    ```mlir
    %buf = mgp.cpu.buffer.constant #cache.constant_hash<"tag"> : !mgp.cpu.buffer<256>
    ```
    becomes, in the 'init' block:
    ```mlir
    %ch1 = mgp.cas.create_instance %ch0, %ctx[0] at "cacheDir"
    %buf = mgp.cas.lookup %ch1, %ctx[0], "tag" : !mgp.cpu.buffer<256>
    %ch2 = mgp.cpu.buffer.cache %ch1, %ctx[5], %buf : !mgp.cpu.buffer<256>
    ```
    and in the 'execute' block:
    ```mlir
    %buf = mgp.cpu.buffer.get_cached %ch, %ctx[5] : !mgp.cpu.buffer<256>
    ```
  }];

  let options = [
    Option<"cacheDir", "cache-dir", "std::string", "",
           "The cache directory to use.">,
  ];

  let dependentDialects = ["Cache::CacheDialect"];
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
  let summary = "Replaces mgp.cpu.buffer.concat with "
                "mcp.cpu.buffer.split.borrow.";
  let description = [{
    This pass replaces mgc.cpu.buffer.concat with
    mcp.cpu.buffer.split.borrow, provided all writers to the buffer
    operands to the concat can be rewritten to instead write to the buffer
    results of split.

    For example:
    ```mlir
    %in1_buf = mgp.cpu.buffer.alloc.static ..., <size1>
    %in1_ch = ... writer to %in1_buf ...
    %in2_spec = ...
    %in2_buf = mgp.cpu.buffer.alloc.spec ..., %in2_spec
    %in2_ch = ... writer to %in2_buf ...
    %out_spec = ...
    %out_buf = mgp.cpu.buffer.alloc.spec ..., %out_spec
    %ch1 = grt.chain.merge (%in1_ch, %in2_ch)
    %ch2 = mgp.cpu.buffer.concat %ch1, %out_buf, %in1_buf, %in2_buf : ...
    // where these are the only uses of %in1_buf and %in2_buf
    ```
    (where %ch0 must be the argument chain to the 'execute' block)
    is rewritten to:
    ```mlir
    %out_spec = ...
    %out_buf = mgp.cpu.buffer.alloc.spec %root_ch, %out_spec
    %in2_spec = ...
    %in2_size = mgp.cpu.tensor_spec.size %in2_spec
    %in1_buf, %in2_buf = mgp.cpu.buffer.split %root_ch, %out_buf, %in2_size : ...
    %in1_ch = ... writer to %in1_buf ...
    %in2_ch = ... writer to %in2_buf ...
    %ch2 = grt.chain.merge (%in1_ch, %in2_ch)
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
  let summary = "Lifts execute-phase invariant code to init-phase";
  let description = [{
    This pass examines computations in the exec phase, identifies the ones that
    are invariant over multiple runs of the exec phase (e.g., with different
    input arguments), and lift them into the init phase.

    To enable aggressive lifting, it does the following:
    1. insert minimal buffer cache ops so that ops in the exec phase can still
       refer to the lifted buffers (which can be allocated buffer or constant).
    2. wire up ops that consumes/produces chains so that control dependencies
       are preserved.
    3. duplicate certain computations to enable more lifting (e.g., to lift a
       simple spec computation that has unliftable users, we must duplicate the
       computation because we can't cache spec).

    For example:
    ```mlir
    mgp.model @example : tf init {
      %ch0 = grt.chain.create
      %ctx = mgp.context.create 1
      mef.output %ch0, %ctx : !mef.chain, !mgp.context

    } execute (%ch0: !mef.chain, %ctx: !mgp.context, %arg: !mgp.cpu.tensor<?xf32>) -> (!mef.chain, !mgp.cpu.tensor<?xf32>) {
      %const_buf     : !mgp.cpu.buffer
      %const_spec    : !mgp.cpu.tensor_spec<?xf32>
      %new_const_buf : !mgp.cpu.tensor_spec<?xf32>
      %new_const_spec = spec_update_op(%const_spec) : (!mgp.cpu.tensor_spec<?xf32>) -> !mgp.cpu.tensor_spec<?xf32>
      %kernel_ch1 = kernel_op1(%ch0, %const_buf, %const_spec, %new_const_buf, %new_const_spec) : ... -> mef.chain

      %out_spec : !mgp.cpu.tensor_spec<?xf32>
      %out_buf  : !mgp.cpu.buffer
      %kernel_ch2 = kernel_op2(%kernel_ch1, %arg_buf, %arg_spec, %new_const_buf, %new_const_spec, %out_buf, %out_spec) : ...
      %out_tensor = mgp.cpu.tensor.create %kernel_ch2, %out_buf, %out_spec  : !mgp.cpu.tensor<?xf32>
      mef.output %ch0, %out_tensor : !mef.chain, !mgp.cpu.tensor<?xf32>
    }
    ```
    is rewritten to (note all the clarifying comments):
    ```mlir
    mgp.model @example : tf init {
      %ch0 = grt.chain.create
      // number of slots increased, to store lifted buffers
      %ctx = mgp.context.create 2

      // moved from exec phase
      %const_buf     : !mgp.cpu.buffer
      %const_spec    : !mgp.cpu.tensor_spec<?xf32>
      %new_const_buf : !mgp.cpu.tensor_spec<?xf32>
      %new_const_spec = spec_update_op(%const_spec) : ...
      %kernel_ch1 = kernel_op1(%ch0, %const_buf, %const_spec, %new_const_buf, %new_const_spec) : ... -> mef.chain

      // inserted; buffer.cache may perform a deep copy, so we must wait for the
      // actions on buffers to finish first.
      %buffer_action_ch = grt.merge (%kernel_ch1) // may have more action chain in general
      %cache_ch = mgp.cpu.buffer.cache %buffer_action_ch, %ctx[1], %new_const_buf : !mgp.cpu.buffer

      // out chain changed
      %out_ch = grt.merge (%cache_ch) // may have more cache_ch in general
      mef.output %out_ch, %ctx : !mef.chain, !mgp.context

    } execute (%ch0: !mef.chain, %ctx: !mgp.context, %arg: !mgp.cpu.tensor<?xf32>) -> (!mef.chain, !mgp.cpu.tensor<?xf32>) {
      // cloned
      %const_spec : !mgp.cpu.tensor_spec<?xf32>
      %new_const_spec = spec_update_op(%const_spec) : (!mgp.cpu.tensor_spec<?xf32>) -> !mgp.cpu.tensor_spec<?xf32>

      // inserted
      %new_const_buf = mgp.cpu.buffer.get_cached %ch0, %ctx[1] : !mgp.cpu.buffer

      // references to `%kernel_ch1` become `%ch0`
      %out_spec : !mgp.cpu.tensor_spec<?xf32>
      %out_buf  : !mgp.cpu.buffer
      %kernel_ch2 = kernel_op2(%ch0, %arg_buf, %arg_spec, %new_const_buf, %new_const_spec, %out_buf, %out_spec) : ... -> mef.chain
      %out_tensor = mgp.cpu.tensor.create %kernel_ch2, %out_buf, %out_spec  : !mgp.cpu.tensor<?xf32>
      mef.output %ch0, %out_tensor : !mef.chain, !mgp.cpu.tensor<?xf32>
    }
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ReallocBufferPass
//===----------------------------------------------------------------------===//

def ReallocBuffersPass : Pass<"realloc-buffers", "MGP::ModelOp"> {
  let summary = "Replaces calls to mgp.cpu.buffer.alloc.* with corresponding "
                "realloc versions";
  let description = [{
    This pass replaces mgp.cpu.buffer.alloc.* calls in the 'execute' block
    with mgp.cpu.buffer.realloc.* calls, also in the 'execute' block. Each
    alloc call gets a dedicated 'default' slot created in the 'init' block.
    In this way buffer allocations will be cached between calls to 'execute'
    unless the required buffer size changes. That reduces dynamic memory
    management overhead at the expense of memory high water mark.

    Note that we do not attempt to move mgp.cpu.buffer.alloc.static allocations
    to the init phase itself, though that may be worthwhile. Currently there's
    no op to allocate into a slot but return only a chain.
  }];
}

#endif // GENERICML_MGPDIALECT_MGPTRANSFORMS_TD
