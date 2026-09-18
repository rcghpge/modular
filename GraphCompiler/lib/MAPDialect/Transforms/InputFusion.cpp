//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Support/InputFusion.h"

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/Support/DataflowUtils.h"

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEINPUT
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

bool isViewForeach(IterForeachOp op) {
  return op.getBody()
      .walk([&](IndexTransformOpInterface) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

bool producesThroughView(StoreOp store) {
  return dataflowGraphContains<WalkDirection::Producers>(
      store.getValue().getDefiningOp(),
      [](Operation *o) { return isa<IndexTransformOpInterface>(o); });
}

namespace {

// TODO(GEX-4184): support multiple stores/outputs. Kernels are single-output
// today (`findEligibleProducer` guards `getNumResults() == 1`); multi-output
// only arises via sibling fusion, where this would fuse per-store instead.
StoreOp producerStoreOf(IterForeachOp src) {
  return getUniqueStore(src.getBody().front());
}

// Whether the index `load` reads at runs through a `map.index.reshape`. A
// reshape changes rank, so `LoopInvariantViewMotion` can't hoist a view fused
// atop such a load into a stride view: a reshape is the terminal view of the
// chain it sits on, and nothing may fuse into a load already past one.
bool loadIndexCrossesReshape(LoadOp load) {
  return dataflowGraphContains<WalkDirection::Producers>(
      load.getIndex().getDefiningOp(),
      [](Operation *o) { return isa<ReshapeIndexOp>(o); });
}

// Guards against fusing across a chain-carrying kernel (e.g. mo.mutable.store)
// when doing so could change effective ordering relative to another consumer of
// the same chain.
bool checkMutableKernelFusible(Value srcInChain, Value srcOutChain,
                               Value dstInChain, Value dstOutChain) {
  if (srcInChain && srcOutChain && dstInChain && dstOutChain) {
    if (dstInChain != srcOutChain)
      return false;
    if (!srcOutChain.hasOneUse())
      return false;
  }
  if (dstInChain && !dstInChain.hasOneUse())
    return false;
  return true;
}

// The existing prologue capturing `dst`'s `argIdx`'th input and that capture's
// own index, or null when the input isn't captured by a prologue, is captured
// with something else, or its capture is read more than once -- any of which
// leaves no single, safely-replaceable load site. Input fusion targets
// prologues only; an epilogue's input capture is reached emergently (a view
// fuses into the consumer foreach, which then output-fuses into the epilogue).
struct ClosureCapture {
  CapturingFusionClosure closure = nullptr;
  unsigned captureIdx = 0;
};

ClosureCapture findExistingPrologueCapture(IterOpaqueOp dst, unsigned argIdx) {
  BlockArgument dstArg = dst.getInputBlockArgs()[argIdx];
  if (!dstArg.hasOneUse())
    return {};
  auto closure = dyn_cast<CapturingFusionClosure>(*dstArg.getUsers().begin());
  if (!closure || !isa<IterPrologueOp>(closure.getOperation()))
    return {};
  auto captureIt = llvm::find(closure.getCaptures(), dstArg);
  if (captureIt == closure.getCaptures().end())
    return {};
  unsigned captureIdx =
      static_cast<unsigned>(captureIt - closure.getCaptures().begin());
  BlockArgument capArg = closure.getCaptureBlockArgs()[captureIdx];
  if (!capArg.hasOneUse())
    return {};
  return {closure, captureIdx};
}

// A producer may only fuse into a consumer that runs on the same device (a
// foreach carries its own device; an opaque carries none, so the result's
// tensor type -- where the consumer reads the value -- stands in).
bool sameDevice(IterForeachOp src, Operation *dst) {
  auto srcDev = MO::getDeviceOrDefault(src);
  if (auto dstForeach = dyn_cast<IterForeachOp>(dst))
    return MO::getDeviceOrDefault(dstForeach) == srcDev;
  auto tensorTy = dyn_cast<MO::TensorType>(src.getResult(0).getType());
  return tensorTy && MO::getDeviceOrDefault(tensorTy) == srcDev;
}

// The first load of any consumer block arg fed by `src`'s result, standing in
// for the fusion site the wrapper's `shouldFuse` inspects. Null if none.
LoadOp firstConsumerLoad(IterForeachOp dst, IterForeachOp src) {
  Block &body = dst.getBody().front();
  for (auto [i, input] : llvm::enumerate(dst.getInputs())) {
    if (input != src.getResult(0))
      continue;
    for (Operation *user : body.getArgument(i).getUsers())
      if (auto load = dyn_cast<LoadOp>(user))
        return load;
  }
  return nullptr;
}

// Splices `src`'s body into `dst` at each of its load site(s), using each
// load's own (possibly transformed) index -- what lets a view fuse. Stamps View
// for a view producer, else Elementwise.
void spliceIntoForeach(IterForeachOp dst, IterForeachOp src,
                       PatternRewriter &rewriter) {
  bool isView = producesThroughView(producerStoreOf(src));
  Block &dstBody = dst.getBody().front();

  SmallVector<unsigned> fusionIndices;
  for (auto [i, input] : llvm::enumerate(dst.getInputs()))
    if (input == src.getResult(0))
      fusionIndices.push_back(i);
  assert(!fusionIndices.empty() && "src must be a current input of dst");

  for (unsigned argIdx : llvm::reverse(fusionIndices)) {
    unsigned numSrcInputs = src.getInputs().size();
    dst.insertInputsAt(argIdx, src.getInputs());
    ValueRange newDstArgs = dstBody.getArguments().slice(argIdx, numSrcInputs);
    BlockArgument oldTensorArg = dstBody.getArgument(argIdx + numSrcInputs);

    for (Operation *user : llvm::to_vector(oldTensorArg.getUsers())) {
      auto load = cast<LoadOp>(user);
      Value loadIndex = load.getIndex();

      rewriter.cloneRegionBefore(src.getBody(), *dstBody.getParent(),
                                 dstBody.getParent()->begin());
      Block &srcClone = dstBody.getParent()->front();

      auto storeOp = getUniqueStore(srcClone);
      Value fusedVal = storeOp.getValue();
      storeOp.erase();
      srcClone.eraseArgument(numSrcInputs);

      // Inline right before `load`: its index may be produced by an earlier op
      // in dst's own body (dst's own transform chain), which only dominates
      // from this point on.
      SmallVector<Value> fusedArgs(newDstArgs.begin(), newDstArgs.end());
      fusedArgs.push_back(loadIndex);
      rewriter.inlineBlockBefore(&srcClone, load, fusedArgs);

      load.getResult().replaceAllUsesWith(fusedVal);
      load.erase();
    }

    dst.eraseInputsAt(argIdx + numSrcInputs);
  }

  if (isView)
    dst.setFusionInfoAttr(MOGG::createFusionInfo<MOGG::FusionViewAttr>(
        src->getContext(), {src, dst}));
  else
    dst.setFusionInfoAttr(MOGG::createFusionInfo<MOGG::FusionElementwiseAttr>(
        src->getContext(), {src, dst}));
  dst->setLoc(
      mlir::FusedLoc::get(src->getContext(), {src->getLoc(), dst->getLoc()}));
  dst.setHasReadEffect(src.getHasReadEffect() || dst.getHasReadEffect());
  dst.setHasWriteEffect(src.getHasWriteEffect() || dst.getHasWriteEffect());
}

// Splices `src`'s body into `dst`'s existing prologue at the load site behind
// the capture(s) `src` feeds. Stamps Prologue.
void spliceIntoClosure(IterOpaqueOp dst, IterForeachOp src,
                       PatternRewriter &rewriter) {
  SmallVector<unsigned> fusionIndices;
  for (auto [i, input] : llvm::enumerate(dst.getInputs()))
    if (input == src.getResult(0))
      fusionIndices.push_back(i);
  assert(!fusionIndices.empty() && "src must be a current input of dst");

  unsigned numSrcInputs = src.getInputs().size();

  for (unsigned argIdx : llvm::reverse(fusionIndices)) {
    ClosureCapture capture = findExistingPrologueCapture(dst, argIdx);
    assert(capture.closure && "findEligibleProducer should have verified this");
    CapturingFusionClosure closure = capture.closure;
    // Captured before cloning inserts a second block ahead of it.
    Block &closureBody = closure.getBody().front();

    BlockArgument oldCap = closure.getCaptureBlockArgs()[capture.captureIdx];
    auto oldLoad = cast<LoadOp>(*oldCap.getUsers().begin());
    Value loadIndex = oldLoad.getIndex();

    rewriter.cloneRegionBefore(src.getBody(), closure.getBody(),
                               closure.getBody().begin());
    Block &srcClone = closure.getBody().front();

    auto storeOp = getUniqueStore(srcClone);
    Value fusedVal = storeOp.getValue();
    storeOp.erase();
    srcClone.eraseArgument(numSrcInputs);

    SmallVector<Value> fusedArgs;
    for (Value input : src.getInputs()) {
      unsigned dstInputIdx = dst.getInputs().size();
      dst.insertInputsAt(dstInputIdx, input);
      Value dstInputArg = dst.getInputBlockArgs()[dstInputIdx];
      closure.appendCaptureOperand(dstInputArg);
      fusedArgs.push_back(
          closureBody.addArgument(input.getType(), input.getLoc()));
    }
    fusedArgs.push_back(loadIndex);

    rewriter.inlineBlockBefore(&srcClone, oldLoad, fusedArgs);

    oldLoad.getResult().replaceAllUsesWith(fusedVal);
    oldLoad.erase();
    closure.eraseCaptureAt(capture.captureIdx);
    dst.eraseInputsAt(argIdx);

    dst.setFusionInfoAttr(MOGG::createFusionInfo<MOGG::FusionPrologueAttr>(
        src->getContext(), {src, dst}, {argIdx}));
    dst->setLoc(
        mlir::FusedLoc::get(src->getContext(), {src->getLoc(), dst->getLoc()}));
  }
}

// Whether `dstOp` has a site to read `src`'s result through if it were cloned
// in: a foreach loads it in its body, an opaque only through an existing
// prologue capture.
bool hasFusionSite(IterForeachOp src, Operation *dstOp) {
  if (src->getBlock() != dstOp->getBlock())
    return false;

  if (auto dst = dyn_cast<IterForeachOp>(dstOp))
    return firstConsumerLoad(dst, src) != nullptr;

  auto dst = dyn_cast<IterOpaqueOp>(dstOp);
  if (!dst)
    return false;
  bool feedsInput = false;
  for (auto [i, in] : llvm::enumerate(dst.getInputs())) {
    if (in != src.getResult(0))
      continue;
    if (!findExistingPrologueCapture(dst, i).closure)
      return false;
    feedsInput = true;
  }
  return feedsInput;
}

// A multi-use producer fuses only if the config allows duplicating it (the
// splice clones it into each consumer); a single-use producer always fuses.
// Cloning pays by removing the materialization, so every user needs a site to
// absorb the clone -- `mo.output`, or an opaque with no prologue on that input,
// leaves the result written to memory anyway. Legacy refuses the same case.
bool duplicationAllowed(IterForeachOp src, const InputFusionConfig &config) {
  if (src.getResult(0).hasOneUse())
    return true;
  for (Operation *user : src.getResult(0).getUsers())
    if (!hasFusionSite(src, user))
      return false;
  return config.canDuplicate(src);
}

// Finds an eligible producer feeding `dstOp` under `config`, or null. A foreach
// consumer fuses the producer into its body; an opaque consumer into an
// existing prologue capture. A producer feeding more than one input must clear
// every one of them (the splice fuses them all).
IterForeachOp findEligibleProducer(Operation *dstOp,
                                   const InputFusionConfig &config) {
  if (auto dst = dyn_cast<IterForeachOp>(dstOp)) {
    Block &body = dst.getBody().front();
    DenseSet<Operation *> tried;
    for (Value input : dst.getInputs()) {
      auto src = input.getDefiningOp<IterForeachOp>();
      if (!src || src.getNumResults() != 1 || !tried.insert(src).second)
        continue;
      // Fusion stays within a block: never across a control-flow boundary.
      if (src->getBlock() != dstOp->getBlock())
        continue;
      if (!sameDevice(src, dst) || !duplicationAllowed(src, config))
        continue;
      if (!checkMutableKernelFusible(src.getInputChain(), src.getOutputChain(),
                                     dst.getInputChain(), dst.getOutputChain()))
        continue;
      // Reshape barrier: a view cannot fuse into a load already past a reshape.
      bool blocked = false;
      for (auto [i, in] : llvm::enumerate(dst.getInputs())) {
        if (in != src.getResult(0))
          continue;
        for (Operation *user : body.getArgument(i).getUsers())
          if (auto load = dyn_cast<LoadOp>(user))
            if (loadIndexCrossesReshape(load)) {
              blocked = true;
              break;
            }
        if (blocked)
          break;
      }
      if (blocked)
        continue;
      LoadOp load = firstConsumerLoad(dst, src);
      if (load && config.shouldFuse(producerStoreOf(src), load))
        return src;
    }
    return nullptr;
  }

  auto dst = cast<IterOpaqueOp>(dstOp);
  DenseSet<Operation *> tried;
  // Highest operand index first: `spliceIntoClosure` records each fused
  // producer's operand index and then `eraseInputsAt`s it, which shifts every
  // lower operand down. Draining high-to-low means an erase never shifts an
  // operand still to be recorded, so the stamped `operandIndices` stay
  // faithful.
  for (Value input : llvm::reverse(dst.getInputs())) {
    auto src = input.getDefiningOp<IterForeachOp>();
    if (!src || src.getNumResults() != 1 || !tried.insert(src).second)
      continue;
    if (src->getBlock() != dstOp->getBlock())
      continue;
    if (!sameDevice(src, dst) || !duplicationAllowed(src, config))
      continue;
    // Every input `src` feeds must resolve to a prologue capture whose load
    // does not already cross a reshape, and pass the wrapper's narrowing.
    bool ok = true;
    LoadOp firstLoad = nullptr;
    for (auto [i, in] : llvm::enumerate(dst.getInputs())) {
      if (in != src.getResult(0))
        continue;
      ClosureCapture capture = findExistingPrologueCapture(dst, i);
      if (!capture.closure) {
        ok = false;
        break;
      }
      BlockArgument capArg =
          capture.closure.getCaptureBlockArgs()[capture.captureIdx];
      auto load = cast<LoadOp>(*capArg.getUsers().begin());
      if (loadIndexCrossesReshape(load)) {
        ok = false;
        break;
      }
      if (!firstLoad)
        firstLoad = load;
    }
    if (ok && firstLoad && config.shouldFuse(producerStoreOf(src), firstLoad))
      return src;
  }
  return nullptr;
}

} // namespace

unsigned fuseInput(MO::GraphOp graphOp, const InputFusionConfig &config) {
  PatternRewriter rewriter(graphOp.getContext());

  // Kernels live in the graph body and in any nested control-flow regions (an
  // `mo.if`'s branches); a producer/consumer edge is always within one block.
  llvm::SmallSetVector<Block *, 8> blocks;
  graphOp.getBody().walk([&](Operation *op) {
    if (isa<IterForeachOp, IterOpaqueOp>(op))
      blocks.insert(op->getBlock());
  });

  // Process consumers bottom-up (reverse topological) so each consumer's
  // producer chain drains in one visit and each producer is reached only after
  // the consumers that pull it in.
  MO::ParametricTopologyAnalysis topo(graphOp);
  DenseSet<Operation *> erased;
  unsigned numFused = 0;
  for (Block *block : blocks) {
    SmallVector<Operation *> ops;
    for (Operation &op : *block)
      ops.push_back(&op);
    topo.computeTopologicalSorting(ops);

    for (Operation *op : llvm::reverse(ops)) {
      if (erased.contains(op) || !isa<IterForeachOp, IterOpaqueOp>(op))
        continue;
      while (IterForeachOp src = findEligibleProducer(op, config)) {
        if (isa<IterForeachOp>(op))
          spliceIntoForeach(cast<IterForeachOp>(op), src, rewriter);
        else
          spliceIntoClosure(cast<IterOpaqueOp>(op), src, rewriter);
        if (src.getResult(0).use_empty()) {
          erased.insert(src);
          src->erase();
        }
        // Uniquing inputs re-exposes a shared producer as single-use, so this
        // drain fuses it once instead of duplicating or dropping it.
        dedupeKernelInputs(op, rewriter);
        ++numFused;
      }
    }
  }
  return numFused;
}

namespace {

// Unconstrained input fusion (fuse every eligible edge). For testing.
struct FuseInputPass : public MAP::impl::FuseInputBase<FuseInputPass> {
  using Base::Base;

  void runOnOperation() override {
    InputFusionConfig config;
    config.shouldFuse = [](StoreOp, LoadOp) { return true; };
    config.canDuplicate = [](IterForeachOp) { return true; };
    fuseInput(getOperation(), config);
  }
};

} // namespace

} // namespace M::MAP
