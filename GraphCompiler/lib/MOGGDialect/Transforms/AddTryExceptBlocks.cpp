//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Interfaces/LoweringState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOpInterfaces.h"
#include "GraphCompiler/MOGGDialect/MOGGInterfaces.h" // IWYU: needed for type declarations used via MOGG::
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOGGDialect/Support/ProfileScopeLabels.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_ADDTRYEXCEPTBLOCKS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

namespace {

struct AddTryExceptBlocks
    : public MOGG::impl::AddTryExceptBlocksBase<AddTryExceptBlocks> {
  using AddTryExceptBlocksBase::AddTryExceptBlocksBase;

  /// Helper to set the builder insertion point for AsyncDelOp operations.
  /// Sets insertion point before the first AsyncReadyOp if one exists in the
  /// block, otherwise before the fallback operation (or at block end if
  /// nullptr).
  void setDelInsertionPoint(OpBuilder &builder, Block &block,
                            Operation *fallback) {
    auto asyncReadyOps = block.getOps<MOGG::AsyncReadyOp>();
    if (!asyncReadyOps.empty()) {
      builder.setInsertionPoint((*asyncReadyOps.begin()).getOperation());
    } else if (fallback) {
      builder.setInsertionPoint(fallback);
    } else {
      builder.setInsertionPointToEnd(&block);
    }
  }

  /// Helper to insert AsyncDelOp for all kernel arguments at the current
  /// insertion point.
  void insertAsyncDelOps(OpBuilder &builder, Location loc,
                         ArrayRef<BlockArgument> args) {
    if (!args.empty())
      MOGG::AsyncDelOp::create(builder, loc, args);
  }

  /// Synthesize profiler ranges from contiguous ops sharing the
  /// same profile scope.
  void addProfileScopeTracing(MOGG::KernelStubOp kernelOp,
                              bool traceUnlabelled) {
    // Use the kernel op locations since MergeIdenticalKernel fuses
    // a single location into KernelStubOp.
    auto kernelScopeLayers = collectProfileScopeLayers(kernelOp->getLoc());
    if (kernelScopeLayers.size() > 1) {
      llvm::SetVector<ProfileScopeLayer> observedLayers;
      for (Operation &op : kernelOp.getBodyBlock().getOperations())
        for (ProfileScopeLayer &layer : collectProfileScopeLayers(op.getLoc()))
          observedLayers.insert(layer);
      if (observedLayers.size() < kernelScopeLayers.size())
        return;
    }

    auto ctx = kernelOp->getContext();
    auto &kernelBody = kernelOp.getBodyBlock();
    Location loc = kernelOp->getLoc();
    auto rangeTy = MOGG::ProfileScopeRangeType::get(ctx);
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(&kernelBody);

    SmallVector<Operation *> traceable;
    for (Operation &op : kernelBody.getOperations()) {
      // Async lifetime/signaling bookkeeping and the region's output packing
      // belong to no user scope, so they neither open a run nor break one.
      if (isa<MOGG::AsyncDelOp, MOGG::AsyncReadyOp>(op) ||
          op.hasTrait<mlir::OpTrait::IsTerminator>())
        continue;
      traceable.push_back(&op);
    }

    if (traceable.empty())
      return;

    SmallVector<ProfileScopeLayerSet> layerSets;
    for (Operation *op : traceable) {
      // Ops that will never be individually traced, such as host-side
      // bookkeeping like buffer lookups or tensor-view construction must never
      // define or break a run boundary.
      auto opIt = mlir::Block::iterator(op);
      if (!opDoesSignificantWork(op)) {
        if (!layerSets.empty())
          layerSets.back().last = opIt;
        continue;
      }
      auto layers = collectProfileScopeLayers(op->getLoc());
      if (layers.empty()) {
        // Unlabeled ops are glue: extend an open run, otherwise ignore.
        if (!layerSets.empty())
          layerSets.back().last = opIt;
        continue;
      }
      if (!layerSets.empty() && layerSets.back().layers == layers) {
        layerSets.back().last = opIt;
        continue;
      }
      layerSets.push_back({opIt, opIt, std::move(layers)});
    }

    // If the whole body is unlabeled, optionally emit one fallback range over
    // the entire stub body. Default is off.
    if (layerSets.empty() && traceUnlabelled)
      layerSets.push_back({mlir::Block::iterator(traceable.front()),
                           mlir::Block::iterator(traceable.back()),
                           {}});

    // Drop insignificant ranges
    llvm::SmallVector<ProfileScopeLayerSet> filteredLayerSets;
    for (ProfileScopeLayerSet &layerSet : layerSets)
      for (Operation &op : layerSet.ops())
        if (opDoesSignificantWork(&op)) {
          filteredLayerSets.push_back(std::move(layerSet));
          break;
        }
    layerSets = std::move(filteredLayerSets);

    std::string regionName = kernelOp.getSymName().str();
    for (ProfileScopeLayerSet &layerSet : layerSets) {
      std::string name = formatProfileScope(layerSet.layers, true) + regionName;
      auto beginOp = MOGG::ProfileScopeBeginOp::create(
          builder, loc, rangeTy, StringAttr::get(ctx, name));
      beginOp->moveBefore(&*layerSet.first);
      auto endOp =
          MOGG::ProfileScopeEndOp::create(builder, loc, beginOp.getRange());
      endOp->moveAfter(&*layerSet.last);
    }
  }

  /// Heuristic to separate ops that should be part of a traced range from those
  /// should not, like host-side bookkeeping such as buffer lookups,
  /// tensor-view construction, shape math, etc.
  static bool opDoesSignificantWork(Operation *op) {
    if (isa<StubCallOpInterface, M::MGP::MGPHasDeviceOpInterface>(op))
      return true;
    return false;
  }

  void runOnOperation() override {
    auto kernelOp = getOperation();
    OpBuilder builder(kernelOp->getContext());
    auto &kernelBody = kernelOp.getBodyBlock();
    auto terminator = kernelBody.getTerminator();
    Location loc = kernelOp->getLoc();

    // Handle MOGG ABI stubs: wrap all ops (except TryOp and OutputOp) in a
    // TryOp. The try block contains the setup ops and WithTraceOp (added by
    // createKernelStub). The except block contains a RaiseOp that formats
    // the error with full kernel context (name, fusion info, traceback).
    if (kernelOp.getAbi() == MOGG::ABIKind::MOGG) {
      if (!kernelOp.isThrows())
        return;

      builder.setInsertionPointToStart(&kernelBody);
      auto tryOp = MOGG::TryOp::create(builder, loc, TypeRange{});
      Block &tryBlock = tryOp.getTryRegion().emplaceBlock();
      for (Operation &op :
           llvm::make_early_inc_range(kernelBody.getOperations())) {
        if (!isa<MOGG::TryOp, MOGG::OutputOp>(op))
          op.moveBefore(&tryBlock, tryBlock.end());
      }
      builder.setInsertionPointToEnd(&tryBlock);
      MOGG::YieldOp::create(builder, loc, ValueRange{});

      Block &exceptBlock = tryOp.getExceptRegion().emplaceBlock();
      Value errArg =
          exceptBlock.addArgument(MOGG::ErrorType::get(loc->getContext()), loc);
      builder.setInsertionPointToStart(&exceptBlock);
      MOGG::RaiseOp::create(builder, loc, errArg);
      return;
    }

    if (kernelOp.getAbi() != MOGG::ABIKind::MGP)
      return;

    // Collect arguments that need cleanup. We need to do this before moving
    // ops around.
    SmallVector<BlockArgument> argsToDelete(kernelBody.getArguments().begin(),
                                            kernelBody.getArguments().end());

    // Profile-scope range bracketing runs before try/except wrapping: the
    // begin/end Trace calls are raising, so they need the wrapping below to
    // keep the MGP-ABI entry point non-raising.
    if (enableProfileScopeTracing)
      addProfileScopeTracing(kernelOp, profileScopeTracingTraceUnlabelled);

    // For non-throwing kernels, insert AsyncDelOp before AsyncReadyOp.
    // AsyncReadyOp signals that the output chain is ready, which tells
    // downstream consumers they can start. By inserting cleanup before
    // AsyncReadyOp, we ensure that when await() returns for downstream
    // consumers, all cleanup from this kernel has completed.
    //
    // Note: We specifically look for AsyncReadyOp and not AsyncPackOp because
    // pack operations may still reference values derived from arguments (like
    // memory storage handles from cached buffers). The chain ready is what
    // triggers downstream execution.
    if (!kernelOp.isThrows()) {
      setDelInsertionPoint(builder, kernelBody, terminator);
      insertAsyncDelOps(builder, loc, argsToDelete);
      return;
    }

    // For throwing kernels, we need AsyncDelOp in BOTH the try and except
    // blocks to ensure cleanup happens before signaling ready/error to
    // downstream consumers, regardless of whether an exception is thrown.

    // Move all ops inside the Try clause.
    builder.setInsertionPointToStart(&kernelBody);
    auto tryOp =
        MOGG::TryOp::create(builder, loc, terminator->getOperandTypes());
    Block &tryBlock = tryOp.getTryRegion().emplaceBlock();
    for (Operation &op :
         llvm::make_early_inc_range(kernelBody.getOperations())) {
      if (!isa<MOGG::TryOp, MOGG::OutputOp>(op))
        op.moveBefore(&tryBlock, tryBlock.end());
    }

    // Insert AsyncDelOp in try block before AsyncReadyOp (or at the end if
    // no AsyncReadyOp found).
    setDelInsertionPoint(builder, tryBlock, /*fallback=*/nullptr);
    insertAsyncDelOps(builder, loc, argsToDelete);

    builder.setInsertionPointToEnd(&tryBlock);
    auto tryYieldOp =
        MOGG::YieldOp::create(builder, loc, terminator->getOperands());

    // The except block returns a chain error and dummy values for other
    // outputs.
    Block &exceptBlock = tryOp.getExceptRegion().emplaceBlock();
    Value errArg =
        exceptBlock.addArgument(MOGG::ErrorType::get(loc->getContext()), loc);
    builder.setInsertionPointToStart(&exceptBlock);

    // Insert AsyncDelOp in except block too - cleanup must happen before
    // signaling error to downstream.
    insertAsyncDelOps(builder, loc, argsToDelete);

    // Create individual AsyncErrorOp for each output (chain and non-chain).
    // Each AsyncErrorOp represents setting an error on one specific output.
    SmallVector<Value> exceptYieldValues;
    exceptYieldValues.reserve(tryYieldOp.getOperands().size());
    for (Value tryYieldOperand : tryYieldOp.getOperands()) {
      Value asyncError = MOGG::AsyncErrorOp::create(
          builder, loc, tryYieldOperand.getType(), errArg);
      exceptYieldValues.push_back(asyncError);
    }

    MOGG::YieldOp::create(builder, loc, exceptYieldValues);

    // Update the final terminator outputs to the try op results.
    terminator->setOperands(tryOp.getOutputs());
  }
};

} // namespace

} // namespace M::MOGG
