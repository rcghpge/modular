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

/// Locate the producer dispatch in a kernel stub body: the `mogg.call.execute`
/// terminating a `mogg.with_trace`.
CallExecuteOp findProducerDispatch(KernelStubOp stub, WithTraceOp &withTrace) {
  CallExecuteOp result;
  stub.walk([&](CallExecuteOp execute) {
    result = execute;
    withTrace = execute->getParentOfType<WithTraceOp>();
    return WalkResult::interrupt();
  });
  return result;
}

/// Build the sibling recording stub `@<original>_build_graph` from `original`
/// and insert it at module scope via `symtab`. The clone mirrors the original
/// body, but its producer dispatch (`mogg.call.execute` inside
/// `mogg.with_trace`) becomes a node-free `mogg.call.build_graph` recording
/// into a `!mgp.device_graph.builder` (the retyped launch arg). The stub
/// returns nothing -- node ordering is carried by the enclosing
/// `mgp.device_graph.dependencies` op, not threaded through the stub.
/// Returns the sibling's symbol name.
FailureOr<StringAttr> createRecordingStub(SymbolTable &symtab,
                                          KernelStubOp original) {
  MLIRContext *ctx = original.getContext();
  OwningOpRef<KernelStubOp> clone = cast<KernelStubOp>(original->clone());

  WithTraceOp withTrace;
  CallExecuteOp execute = findProducerDispatch(*clone, withTrace);
  if (!execute) {
    return original.emitError(
        "device-graph recording stub generation: kernel stub has no "
        "`mogg.with_trace { mogg.call.execute }` producer dispatch");
  }

  // The launch context the producer dispatches against must be the stub's last
  // block argument -- that is the slot the builder takes over.
  Block &body = clone->getBodyBlock();
  Value context = execute.getDeviceContext();
  if (body.getNumArguments() == 0 || context != body.getArguments().back()) {
    return original.emitError(
        "device-graph recording stub generation: expected the kernel stub's "
        "last argument to be the producer launch context");
  }

  // Snapshot the producer's operands/attrs *before* retyping the launch arg:
  // `getKernelInputs()` keys off the context type to drop it, so it must run
  // while the arg is still `!mogg.context`.
  SmallVector<Value> kernelInputs = execute.getKernelInputs();
  SmallVector<Value> outputs = llvm::to_vector(execute.getOutputs());
  StringAttr kernelSymName = execute.getKernelSymNameAttr();
  StringAttr moOpName = execute.getMoOpNameAttr();
  UnitAttr isThrows = execute.getIsThrowsAttr();
  DictionaryAttr kernelParams = execute.getKernelParamAttr();

  // Retype the launch arg: `!mogg.context` -> `!mgp.device_graph.builder`.
  BlockArgument launchArg = body.getArguments().back();
  launchArg.setType(DeviceGraphBuilderType::get(ctx));

  Operation &insertionPoint = withTrace ? *withTrace : *execute;

  // Replace `with_trace { call.execute }` with a `call.build_graph` recording
  // into the builder. `call.build_graph` is a plain value op (not a kernel
  // terminator), so the wrapping `with_trace` is dropped.
  OpBuilder builder(&insertionPoint);
  CallBuildGraphOp::create(builder,
                           /*location=*/execute.getLoc(),
                           /*kernelSymName=*/kernelSymName,
                           /*moOpName=*/moOpName,
                           /*isThrows=*/isThrows,
                           /*inputs=*/kernelInputs,
                           /*outputs=*/outputs,
                           /*graphBuilder=*/launchArg,
                           /*kernel_param=*/kernelParams);

  if (withTrace)
    withTrace.erase();

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
  return newName;
}

/// Retarget a staged `mogg.call.stub @stub_N` to its sibling
/// `@stub_N_build_graph`.
///
/// The sibling takes a `!mgp.device_graph.builder` where the original took a
/// launch context.
LogicalResult rewriteStagedCall(CallStubOp call, StringAttr siblingName) {
  MLIRContext *ctx = call.getContext();
  SmallVector<Value> operands = llvm::to_vector(call.getInputs());
  if (operands.empty() ||
      !isa<DeviceContextType, M::MOGG::ContextType>(operands.back().getType()))
    return call.emitError("device-graph recording stub call: expected the last "
                          "operand to be the launch context");

  IRRewriter rewriter(call);

  Value launch = operands.back();
  // Fallback: bridge the launch context to a builder.
  operands.back() = UnrealizedConversionCastOp::create(
                        rewriter, call.getLoc(),
                        /*resultTypes=*/DeviceGraphBuilderType::get(ctx),
                        /*operands=*/launch)
                        .getResult(0);

  // The sibling stub is node-free (void result), so the retargeted call keeps
  // the original call's result types unchanged.
  auto newCall =
      CallStubOp::create(rewriter, call.getLoc(), call.getResultTypes(),
                         siblingName, operands, call.getAffinityDeviceIdAttr());

  rewriter.replaceAllUsesWith(call.getResults(), newCall.getResults());

  call.erase();
  return success();
}

/// Replace each `mgp.buffer.alloc` in `staging` with `mgp.device_graph.alloc`.
void rewriteBuilderAllocs(DeviceGraphStagingOp staging) {
  staging.getBody().walk([&](BufferAllocOp alloc) {
    IRRewriter rewriter(alloc);
    auto graphBuilderType = rewriter.getType<DeviceGraphBuilderType>();
    auto cast = UnrealizedConversionCastOp::create(
        rewriter, alloc.getLoc(), graphBuilderType, alloc.getDeviceContext());

    rewriter.replaceOpWithNewOp<DeviceGraphAllocOp>(
        alloc,
        /*outBuffer=*/alloc.getOutBuffer().getType(),
        /*inChain=*/alloc.getInChain(),
        /*runtimeSize=*/alloc.getRuntimeSize(),
        /*graphBuilder=*/cast.getResult(0),
        /*rawAlign=*/alloc.getRawAlign());
  });
}

/// For each `mogg.call.stub @stub_N` inside `staging` (the staged execute
/// body), generate a sibling recording stub `@stub_N_build_graph` (inserted at
/// module scope alongside `@stub_N` via `symtab`) and retarget the call to it.
LogicalResult generateBuildGraphStubs(SymbolTable &symtab,
                                      DeviceGraphStagingOp staging) {
  // TODO: We may need to expand this to support `CallStubAsyncOp` in the
  // future.
  SmallVector<CallStubOp> calls;
  staging.getBody().walk([&](CallStubOp call) { calls.push_back(call); });

  if (calls.empty())
    return success();

  // One sibling per distinct callee; all calls to it are retargeted.
  llvm::DenseMap<StringAttr, StringAttr> calleeToSibling;
  for (CallStubOp call : calls) {
    StringAttr callee = call.getNameAttr().getRootReference();

    StringAttr &sibling = calleeToSibling[callee];
    if (!sibling) {
      auto original = symtab.lookup<KernelStubOp>(callee.getValue());
      if (!original)
        return call.emitError("device-graph recording stub call: '")
               << callee.getValue() << "' does not name a kernel stub";

      FailureOr<StringAttr> siblingName = createRecordingStub(symtab, original);
      if (failed(siblingName))
        return failure();

      sibling = *siblingName;
    }

    if (failed(rewriteStagedCall(call, sibling)))
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
    // `mgp-compute-device-graph-dependencies`. Scope is the staged execute
    // body: collect the `mgp.device_graph.staging` op(s) of each
    // `isDeviceGraph` model first, then process them (generating stubs mutates
    // the module's top-level op list, so we must not be mid-iteration over it).
    SmallVector<DeviceGraphStagingOp> stagingOps;
    for (ModelOp model : module.getOps<ModelOp>()) {
      if (!model.getIsDeviceGraph())
        continue;

      model.getExecuteBlock()->walk<WalkOrder::PreOrder>(
          [&](DeviceGraphStagingOp staging) {
            stagingOps.push_back(staging);
            // Staging ops do not nest.
            return WalkResult::skip();
          });
    }

    for (DeviceGraphStagingOp staging : stagingOps)
      if (failed(generateBuildGraphStubs(symtab, staging)))
        return signalPassFailure();

    // Route in-body allocations through the builder: rewrite
    // `mgp.buffer.alloc(builder-cast)` to `mgp.device_graph.alloc(builder)`.
    for (DeviceGraphStagingOp staging : stagingOps)
      rewriteBuilderAllocs(staging);
  }
};

} // namespace
} // namespace M::MGP
