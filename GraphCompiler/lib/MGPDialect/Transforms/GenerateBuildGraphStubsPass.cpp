//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

using namespace M::MGP;
using namespace M::MOGG;
using namespace mlir;

namespace {

/// The producer dispatch in a kernel stub body, with the launch context it
/// runs against. `mogg.with_trace` is not part of the contract -- it is an
/// out-param every later use null-guards. What is required is one dispatch and
/// a context. The MAP ops are not terminators and share no interface with the
/// legacy ones, so the match is by op rather than by interface.
struct ProducerDispatch {
  Operation *op = nullptr;
  Value context;

  explicit operator bool() const { return op != nullptr; }
};

ProducerDispatch matchProducerDispatch(Operation *op) {
  if (isa<CallExecuteOp, CallForeachOp, CallMaterializeOp>(op))
    return {op, cast<KernelTerminator>(op).getDeviceContext()};

  if (auto opaque = dyn_cast<CallOpaqueOp>(op)) {
    ValueRange contexts = opaque.getDeviceContexts();
    return {op, contexts.empty() ? Value() : contexts.front()};
  }

  if (auto foreachRegion = dyn_cast<CallForeachRegionOp>(op))
    return {op, foreachRegion.getContext()};

  return {};
}

ProducerDispatch findProducerDispatch(KernelStubOp stub,
                                      WithTraceOp &withTrace) {
  ProducerDispatch result;
  stub.walk([&](Operation *op) {
    ProducerDispatch match = matchProducerDispatch(op);
    if (!match)
      return WalkResult::advance();

    result = match;
    withTrace = op->getParentOfType<WithTraceOp>();
    return WalkResult::interrupt();
  });

  return result;
}

/// Name and launch-context argument position of a generated recording sibling
/// stub. The position tells the call-site rewrite which operand to bridge to a
/// builder.
struct RecordingStub {
  StringAttr name;
  unsigned contextArgIndex = 0;
};

/// Build the sibling recording stub `@<original>_build_graph` from `original`
/// and insert it at module scope via `symtab`. The clone mirrors the original
/// body with its launch arg retyped from `!mogg.context` to
/// `!mgp.device_graph.builder`, and how the producer dispatch (a
/// `mogg.call.execute` or `mogg.call.materialize` inside `mogg.with_trace`)
/// records depends on the producer:
///   - A `mogg.call.execute` whose kernel has a node-free `build_graph` method:
///     the dispatch becomes a `mogg.call.build_graph` recording into the
///     builder directly (the `mogg.with_trace` is dropped).
///   - An execute-only kernel (`mogg.call.execute` with no `build_graph`) or a
///     `mogg.call.materialize` view copy: the producer is kept, but its launch
///     context is repointed to a recording device context viewed off the
///     builder via `mgp.device_graph.recording_context`, so the ordinary
///     dispatch records into the graph.
/// Either way the stub returns nothing -- node ordering is carried by the
/// enclosing `mgp.device_graph.dependencies` op, not threaded through the stub.
/// Returns the sibling's symbol name and the launch-context argument position.
FailureOr<RecordingStub> createRecordingStub(SymbolTable &symtab,
                                             KernelStubOp original) {
  MLIRContext *ctx = original.getContext();
  OwningOpRef<KernelStubOp> clone = cast<KernelStubOp>(original->clone());

  WithTraceOp withTrace;
  ProducerDispatch producer = findProducerDispatch(*clone, withTrace);
  if (!producer) {
    return original.emitError(
        "device-graph recording stub generation: kernel stub has no producer "
        "dispatch");
  }

  // The launch context the producer dispatches against must be one of the
  // stub's block arguments (in any position) -- that is the slot the builder
  // takes over.
  Block &body = clone->getBodyBlock();
  auto context = dyn_cast<BlockArgument>(producer.context);
  if (!context || context.getOwner() != &body) {
    return original.emitError(
        "device-graph recording stub generation: expected the producer launch "
        "context to be an argument of the kernel stub");
  }

  // Only a `mogg.call.execute` whose kernel declares a node-free `build_graph`
  // method records through `mogg.call.build_graph`. Everything else -- an
  // execute-only kernel or a `mogg.call.materialize` view copy -- records by
  // dispatching the producer against a recording device context.
  auto execute = dyn_cast<CallExecuteOp>(producer.op);
  auto opaque = dyn_cast<CallOpaqueOp>(producer.op);
  KernelDeclOp decl;
  if (execute)
    decl =
        symtab.lookup<KernelDeclOp>(execute.getKernelSymNameAttr().getValue());
  else if (opaque)
    decl = symtab.lookup<KernelDeclOp>(opaque.getCallee());
  bool useBuildGraph = decl && decl.getBuildGraph().has_value();

  // `mogg.call.build_graph` has no fusion flags and its emission never asks,
  // so recording a fused kernel through it would drop the fusion silently.
  if (useBuildGraph && opaque &&
      (opaque.getHasEpilogueFusion() || opaque.getHasPrologueFusion())) {
    return original.emitError(
        "device-graph recording stub generation: cannot record a fused "
        "`mogg._call.opaque` through `mogg.call.build_graph`");
  }

  // Snapshot the producer's operands/attrs *before* retyping the launch arg:
  // `getKernelInputs()` keys off the context type to drop it, so it must run
  // while the arg is still `!mogg.context`. Only the `build_graph` path
  // rebuilds the dispatch, so this is unneeded otherwise.
  SmallVector<Value> kernelInputs, outputs;
  StringAttr kernelSymName, moOpName;
  UnitAttr isThrows;
  DictionaryAttr kernelParams;
  if (useBuildGraph && execute) {
    kernelInputs = execute.getKernelInputs();
    outputs = llvm::to_vector(execute.getOutputs());
    kernelSymName = execute.getKernelSymNameAttr();
    moOpName = execute.getMoOpNameAttr();
    isThrows = execute.getIsThrowsAttr();
    kernelParams = execute.getKernelParamAttr();
  } else if (useBuildGraph) {
    // `useBuildGraph` implies a decl, which only an execute or an opaque
    // resolves, so this branch is the opaque.
    assert(opaque && "a build_graph producer is an execute or an opaque");
    // `_call.opaque` keeps inputs, outputs and contexts in separate operand
    // groups, so there is no context to filter out. `moOpName` is provenance
    // that nothing reads back, so the decl's registration stands in.
    kernelInputs = llvm::to_vector(opaque.getInputs());
    outputs = llvm::to_vector(opaque.getOutputs());
    kernelSymName = StringAttr::get(ctx, opaque.getCallee());
    moOpName = StringAttr::get(ctx, decl.getRegistration());
    // The raises bit has to come off `build_graph`, since that is what this
    // dispatch calls. `CallOpaqueOp::isThrows()` cannot serve: it resolves the
    // decl by symbol lookup, which finds nothing from inside a detached clone
    // and would silently answer false.
    auto buildGraphSignature = decl.getBuildGraph()->getSignature();
    isThrows = buildGraphSignature.isThrows() ? UnitAttr::get(ctx) : UnitAttr();
    kernelParams = opaque.getKernelParamAttr();
  }

  // Retype the launch arg: `!mogg.context` -> `!mgp.device_graph.builder`.
  BlockArgument launchArg = context;
  launchArg.setType(DeviceGraphBuilderType::get(ctx));

  Operation &insertionPoint = withTrace ? *withTrace : *producer.op;

  if (useBuildGraph) {
    OpBuilder builder(&insertionPoint);
    // Replace `with_trace { call.execute }` with a `call.build_graph` recording
    // into the builder. `call.build_graph` is a plain value op (not a kernel
    // terminator), so the wrapping `with_trace` is dropped.
    CallBuildGraphOp::create(builder,
                             /*location=*/producer.op->getLoc(),
                             /*kernelSymName=*/kernelSymName,
                             /*moOpName=*/moOpName,
                             /*isThrows=*/isThrows,
                             /*inputs=*/kernelInputs,
                             /*outputs=*/outputs,
                             /*graphBuilder=*/launchArg,
                             /*kernel_param=*/kernelParams);
    // Legacy wraps the producer in `with_trace`, so erasing that removes the
    // dispatch too; a MAP producer stands alone and has to go on its own.
    if (withTrace)
      withTrace.erase();
    else
      producer.op->erase();
  } else {
    auto builder = OpBuilder::atBlockBegin(&clone->getBodyBlock());
    // Execute-only kernel or view copy: keep the producer dispatch, but repoint
    // its launch context from the (now builder-typed) block arg to a recording
    // device context viewed off the builder. The dispatch then records into the
    // graph instead of running eagerly.
    auto recordingContext = DeviceGraphRecordingContextOp::create(
        builder, producer.op->getLoc(), M::MOGG::ContextType::get(ctx),
        launchArg);
    launchArg.replaceUsesWithIf(
        recordingContext.getContext(),
        [&](OpOperand &use) { return use.getOwner() == producer.op; });
  }

  // The recording is node-free: ordering is expressed by the enclosing
  // `mgp.device_graph.dependencies` op (which lowers to
  // `DeviceGraphBuilder.region`), so no `!mgp.device_graph.node` is threaded
  // through the stub. Leave the `mogg.output` terminator operand-free and give
  // the stub a void result; only the launch arg changes (context -> builder).
  clone->setFunctionTypeAttr(
      TypeAttr::get(FunctionType::get(ctx, body.getArgumentTypes(), {})));

  // Name and register the sibling alongside the original (kept for the
  // direct-execution / init path). The name is unique, so `insert` preserves
  // it.
  clone->setSymNameAttr(
      StringAttr::get(ctx, original.getSymName() + "_build_graph"));

  // Insert the operation just after the original stub, transferring ownership
  // now that the op lives in the module.
  StringAttr newName =
      symtab.insert(clone.release(), std::next(original->getIterator()));
  return RecordingStub{newName, context.getArgNumber()};
}

/// The builder recording the region that contains `op`: the body block argument
/// of the innermost `mgp.device_graph.dependencies`, or the container body's
/// builder for an op sitting directly in a `staging`/`collective` region.
Value enclosingGraphBuilder(Operation *op) {
  if (auto deps = op->getParentOfType<DeviceGraphDependenciesOp>())
    return deps.getBuilderBlockArgument();

  if (auto staging = op->getParentOfType<DeviceGraphStagingOp>())
    return staging.getBodyBuilder();

  auto collective = op->getParentOfType<DeviceGraphCollectiveOp>();
  ASSERT_STREAM(collective,
                "reached an operation not inside a device graph region");

  Region *opRegion = op->getParentRegion();
  for (DeviceGraphCollectiveOp::CollectiveRegion region : collective.graphs()) {
    auto blockRegion = region.bodyBlock->getParent();
    if (blockRegion->isAncestor(opRegion))
      return region.builder;
  }

  llvm_unreachable("unable to determine body region");
}

/// Retarget a `mogg.call.stub @stub_N` inside a device-graph container to its
/// sibling `@stub_N_build_graph`.
///
/// The sibling takes a `!mgp.device_graph.builder` in the launch-context
/// position where the original took a launch context; all other operands
/// (including any other context-typed ones) pass through unchanged.
LogicalResult rewriteGraphCall(CallStubOp call, const RecordingStub &sibling) {
  SmallVector<Value> operands = llvm::to_vector(call.getInputs());
  if (sibling.contextArgIndex >= operands.size() ||
      !isa<DeviceContextType, M::MOGG::ContextType>(
          operands[sibling.contextArgIndex].getType())) {
    return call.emitError("device-graph recording stub call: expected operand ")
           << sibling.contextArgIndex << " to be the launch context";
  }

  IRRewriter rewriter(call);

  // Take the builder directly, as `rewriteBuilderAllocs` does; the launch
  // context chain this leaves dead is canonicalized away before finalization.
  operands[sibling.contextArgIndex] = enclosingGraphBuilder(call);

  // The sibling stub is node-free (void result), so the retargeted call keeps
  // the original call's result types unchanged.
  auto newCall = CallStubOp::create(rewriter, call.getLoc(),
                                    call.getResultTypes(), sibling.name,
                                    operands, call.getAffinityDeviceIdAttr());

  rewriter.replaceAllUsesWith(call.getResults(), newCall.getResults());

  call.erase();

  return success();
}

/// Replace each `mgp.buffer.alloc` in `container` with
/// `mgp.device_graph.alloc`, so every address the graph records comes from the
/// builder's memory pool and stays reserved for the graph's lifetime.
void rewriteBuilderAllocs(Operation *container) {
  container->walk([&](BufferAllocOp alloc) {
    IRRewriter rewriter(alloc);
    rewriter.replaceOpWithNewOp<DeviceGraphAllocOp>(
        alloc,
        /*outBuffer=*/alloc.getOutBuffer().getType(),
        /*inChain=*/alloc.getInChain(),
        /*runtimeSize=*/alloc.getRuntimeSize(),
        /*graphBuilder=*/enclosingGraphBuilder(alloc),
        /*rawAlign=*/alloc.getRawAlign());
  });
}

/// For each `mogg.call.stub @stub_N` inside `container` (the body of a
/// `staging` or `collective`), generate a sibling recording stub
/// `@stub_N_build_graph` (inserted at module scope alongside `@stub_N` via
/// `symtab`) and retarget the call to it.
LogicalResult generateBuildGraphStubs(SymbolTable &symtab,
                                      Operation *container) {
  // TODO: We may need to expand this to support `CallStubAsyncOp` in the
  // future.
  SmallVector<CallStubOp> calls;
  container->walk([&](CallStubOp call) { calls.push_back(call); });

  if (calls.empty())
    return success();

  // One sibling per distinct callee; all calls to it are retargeted.
  llvm::DenseMap<StringAttr, RecordingStub> calleeToSibling;
  for (CallStubOp call : calls) {
    StringAttr callee = call.getNameAttr().getRootReference();

    RecordingStub &sibling = calleeToSibling[callee];
    if (!sibling.name) {
      auto original = symtab.lookup<KernelStubOp>(callee.getValue());
      if (!original)
        return call.emitError("device-graph recording stub call: '")
               << callee.getValue() << "' does not name a kernel stub";

      FailureOr<RecordingStub> recording =
          createRecordingStub(symtab, original);
      if (failed(recording))
        return failure();

      sibling = std::move(*recording);
    }

    if (failed(rewriteGraphCall(call, sibling)))
      return failure();
  }
  return success();
}

} // namespace

namespace M::MGP {

#define GEN_PASS_DEF_GENERATEBUILDGRAPHSTUBSPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

class GenerateBuildGraphStubsPass final
    : public impl::GenerateBuildGraphStubsPassBase<
          GenerateBuildGraphStubsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Operate on the module so the generated recording stubs can be inserted at
    // module scope. Use the module's cached `SymbolTableAnalysis`:
    // `mogg-to-mgp` registers the kernel stubs through it, so it is current. We
    // keep it current too (sibling stubs are added via `symtab.insert`), hence
    // preserve.
    SymbolTable &symtab =
        getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

    // Device-graph synthesis; runs after
    // `mgp-compute-device-graph-dependencies`. Scope is each device-graph
    // container (`mgp.device_graph.staging` and `mgp.device_graph.collective`)
    // in the execute body. Collect them first, then process them (generating
    // stubs mutates the module's top-level op list, so we must not be
    // mid-iteration over it).
    SmallVector<DeviceGraphCollectiveOp> graphContainers;
    for (ModelOp model : module.getOps<ModelOp>()) {
      if (!model.getIsDeviceGraph())
        continue;

      model.getExecuteBlock()->walk<WalkOrder::PreOrder>(
          [&](DeviceGraphCollectiveOp collective) {
            graphContainers.push_back(collective);
            // Device-graph containers do not nest.
            return WalkResult::skip();
          });
    }

    for (Operation *container : graphContainers)
      if (failed(generateBuildGraphStubs(symtab, container)))
        return signalPassFailure();

    // Route in-body allocations through the builder: rewrite
    // `mgp.buffer.alloc(builder-cast)` to `mgp.device_graph.alloc(builder)`.
    for (Operation *container : graphContainers)
      rewriteBuilderAllocs(container);
  }
};

} // namespace
} // namespace M::MGP
