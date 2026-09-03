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
#include "GraphCompiler/MOGGDialect/MOGGTypes.h"
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

    // Also bracket each ancestor scope level a leaf run's layer belongs to,
    // appended before the leaf runs so a shared start/end anchor between an
    // ancestor range and its first/last child orders correctly (ancestor
    // starts first, ends last) purely from insertion order below -- see
    // addAncestorProfileScopeRuns.
    size_t maxLayerLen = 0;
    for (const ProfileScopeLayerSet &layerSet : layerSets)
      if (layerSet.layers.size() == 1)
        maxLayerLen = std::max(maxLayerLen, layerSet.layers.front().size());

    llvm::SmallVector<ProfileScopeLayerSet> allLayerSets;
    for (size_t depth = 0; depth + 1 < maxLayerLen; ++depth)
      addAncestorProfileScopeRuns(layerSets, depth, allLayerSets);
    allLayerSets.append(layerSets.begin(), layerSets.end());

    std::string regionName = kernelOp.getSymName().str();
    for (ProfileScopeLayerSet &layerSet : allLayerSets) {
      if (!llvm::any_of(layerSet.ops(), [](Operation &op) {
            return opDoesSignificantWork(&op);
          }))
        continue;

      std::string name = formatProfileScope(layerSet.layers, true) + regionName;
      auto colors = collectRunColors(layerSet.ops());
      auto resolvedColor = resolveProfileScopeColor(colors);
      mlir::StringAttr colorAttr;
      if (resolvedColor)
        colorAttr = StringAttr::get(ctx, *resolvedColor);
      auto beginOp = MOGG::ProfileScopeBeginOp::create(
          builder, loc, rangeTy, StringAttr::get(ctx, name), colorAttr);
      beginOp->moveBefore(&*layerSet.first);
      auto endOp =
          MOGG::ProfileScopeEndOp::create(builder, loc, beginOp.getRange());
      endOp->moveAfter(&*layerSet.last);
    }
  }

  /// Collects every distinct profile scope color in range. Used to decide
  /// which color (if any) the range spanning these ops should claim.
  static llvm::SmallVector<std::string>
  collectRunColors(llvm::iterator_range<mlir::Block::iterator> range) {
    llvm::SmallSetVector<std::string, 4> colors;
    for (Operation &op : range) {
      auto opColors = collectProfileScopeColors(op.getLoc());
      for (std::string &color : opColors)
        colors.insert(std::move(color));
    }
    return colors.takeVector();
  }

  /// Heuristic to separate ops that should be part of a traced range from those
  /// should not, like host-side bookkeeping such as buffer lookups,
  /// tensor-view construction, shape math, etc.
  static bool opDoesSignificantWork(Operation *op) {
    if (isa<StubCallOpInterface, M::MGP::MGPHasDeviceOpInterface>(op))
      return true;
    return false;
  }

  /// Collects every distinct `Graph.profile_scope(..., color=...)` color
  /// reachable from any op in `[first, last]` (inclusive), deduplicated and
  /// sorted for discovery-order independence. Used to decide which color
  /// (if any) the range spanning these ops should claim.
  static llvm::SmallVector<std::string, 2> collectRunColors(Operation *first,
                                                            Operation *last) {
    llvm::SmallVector<std::string, 2> colors;
    for (Operation *op = first; op; op = op->getNextNode()) {
      auto opColors = collectProfileScopeColors(op->getLoc());
      for (std::string &color : opColors)
        if (!llvm::is_contained(colors, color))
          colors.push_back(std::move(color));
      if (op == last)
        break;
    }
    llvm::sort(colors);
    return colors;
  }

  /// Group adjacent single-layer entries of `leafLayerSets` that share the
  /// same length-`(depth + 1)` layer prefix into coarser ancestor ranges,
  /// appending one `ProfileScopeLayerSet` per group of two or more leaf layer
  /// sets to `outLayerSets`.
  ///
  /// A peer set (`layers.size() > 1`, ops merged in from an unrelated scope by
  /// fusion) has no meaningful ancestor and never participates. A leaf layer
  /// set whose layer is *exactly* `depth + 1` long doesn't participate at that
  /// depth either: its depth-`d` prefix would just be a second copy of the
  /// leaf range that already covers the identical span.
  static void addAncestorProfileScopeRuns(
      llvm::ArrayRef<ProfileScopeLayerSet> leafLayerSets, size_t depth,
      llvm::SmallVector<ProfileScopeLayerSet> &outLayerSets) {
    const ProfileScopeLayerSet *groupFirst = nullptr;
    const ProfileScopeLayerSet *groupLast = nullptr;
    llvm::ArrayRef<std::string> groupPrefix;
    size_t groupSize = 0;

    auto flush = [&] {
      if (groupSize > 1)
        outLayerSets.push_back({groupFirst->first,
                                groupLast->last,
                                {MOGG::ProfileScopeLayer(groupPrefix.begin(),
                                                         groupPrefix.end())}});
      groupFirst = nullptr;
      groupSize = 0;
    };

    for (const ProfileScopeLayerSet &layerSet : leafLayerSets) {
      bool participates = layerSet.layers.size() == 1 &&
                          layerSet.layers.front().size() > depth + 1;
      if (participates) {
        llvm::ArrayRef<std::string> prefix =
            llvm::ArrayRef<std::string>(layerSet.layers.front())
                .take_front(depth + 1);
        if (groupFirst && prefix == groupPrefix) {
          groupLast = &layerSet;
          ++groupSize;
          continue;
        }
        flush();
        groupFirst = groupLast = &layerSet;
        groupPrefix = prefix;
        groupSize = 1;
        continue;
      }
      flush();
    }
    flush();
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
