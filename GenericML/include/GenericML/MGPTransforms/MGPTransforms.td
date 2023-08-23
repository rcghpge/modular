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
  let summary = "Replace mgp.cpu.buffer.alloc.static ops with "
                "mgp.cpu.buffer.section.static.borrow ops using statically "
                "determined offsets.";
  let description = [{
    This pass replaces mgp.cpu.buffer.alloc.{static,spec} ops with
    mgp.cpu.buffer.section.{static,dynamic}.borrow ops from (pre-allocated)
    shared buffers. The size of the shared buffers, and the offset to use for
    each mgp.cpu.buffer.section.{static,dynamic}.borrow op, is chosen by
    this pass so as to ensure no active buffers will ever overlap in their
    borrowed memory.

    The mgp.cpu.buffer.alloc.spec ops can be replaced only when their shape has
    a single dynamic dimension (eg for a dynamic batch size). Planning is done
    in groups for each SSA value supplying the single runtime dynamic dimension.
    The mgp.cpu.buffer.alloc.static ops are planning in their own group.

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
    This pass avoid unnecessary mgp.cpu.tensor.{create,extract} operations in
    two situations:

    1. If the result of a mgp.cpu.tensor.create op is exclusively consumed by
       mgp.cpu.tensor.extract ops, remove them in favor of the original buffer,
       tensor spec and chain used by the mgp.cpu.tensor.create. Eg:
    ```mlir
          %tensor0 = mgp.cpu.tensor.create %ch0, %inputBuf0, %inputSpec0
          %tensor1 = mgp.cpu.tensor.create %ch1, %inputBuf1, %inputSpec1
          %buf0, %spec0 = mgp.cpu.tensor.extract %tensor0
          %buf1, %spec1 = mgp.cpu.tensor.extract %tensor1
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
        - All buffers passed to mgp.cpu.tensor.create will not be further
          mutated once the chain to that op is ready. This is currently true
          for the GraphCompiler, but need not be true in general.
        - Ops have buffer annotations indicating which are in 'write' positions.

    2. Ensures values of type !mgp.cpu.tensor not otherwise the result of an
       mgp.cpu.tensor.create op (eg block arguments or the results of other
       primitives) are extracted at most once. Eg:
    ```mlir
          execute(%tensor : !mgp.cpu.tensor) {
            %buf, %spec = mgp.cpu.tensor.extract %tensor
            my.op(..., %buf, %spec, ...)
            %buf2, %spec2 = mgp.cpu.tensor.extract %tensor
            my.op(..., %buf2, %spec2, ...)
          }
    ```
       becomes:
    ```mlir
          execute(%tensor : !mgp.cpu.tensor) {
            %buf, %spec = mgp.cpu.tensor.extract %tensor
            my.op(..., %buf, %spec, ...)
            my.op(..., %buf, %spec, ...)
          }
    ```

    These rules apply across control flow boundaries. Eg:
    ```mlir
          %tensor0 = mgp.cpu.tensor.create %ch0, %inputBuf0, %inputSpec0
          %ch1, %tensor1 = grt.if %cond (%ch0, %tensor0) -> ... {
            %buf0, %spec0 = mgp.cpu.tensor.extract %tensor0
            %tensor1 = mgp.cpu.tensor.create %ch0, %buf0, %spec0
            mef.output %ch0, %tensor1
          } else {
            ...
          }
          %buf1, %spec1 = mgp.cpu.tensor.extract %tensor1
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
    Ensures mgp.cpu.buffer.alloc.* operations are deferred till they are
    about to be written to. Eg:
    ```mlir
      %buf = mgp.cpu.buffer.alloc.static %bufChain, 24
      my_op(%opChain, %buf, ...)
      // where %buf is in an 'output' position
    ```
    becomes:
    ```
      %ch = grt.chain.create (%opChain, %bufChain)
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
    %ch1 = grt.chain.create (%in1_ch, %in2_ch)
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
    %ch2 = grt.chain.create (%in1_ch, %in2_ch)
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
    mgp.model @example : mof init {
      %ch0 = grt.chain.create ()
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
    mgp.model @example : mof init {
      %ch0 = grt.chain.create ()
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

#endif // GENERICML_MGPDIALECT_MGPTRANSFORMS_TD
