//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Support/OutputFusion.h"

#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/Support/DataflowUtils.h"

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEOUTPUT
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

bool consumerReadsUntransformed(LoadOp load) {
  auto foreachOp = load->getParentOfType<IterForeachOp>();
  if (!foreachOp)
    return false;
  return load.getIndex() == foreachOp.getBody().front().getArguments().back();
}

bool consumerReadsThroughReshape(LoadOp load) {
  auto foreachOp = load->getParentOfType<IterForeachOp>();
  if (!foreachOp)
    return false;
  Value loopIdx = foreachOp.getBody().front().getArguments().back();
  auto reshape = load.getIndex().getDefiningOp<ReshapeIndexOp>();
  return reshape && reshape.getIndex() == loopIdx;
}

bool consumerIsPureReshape(LoadOp load) {
  if (!consumerReadsThroughReshape(load))
    return false;
  auto foreachOp = load->getParentOfType<IterForeachOp>();
  auto store = getUniqueStore(foreachOp.getBody().front());
  auto storedLoad = store.getValue().getDefiningOp<LoadOp>();
  return storedLoad && storedLoad.getTensor() == load.getTensor();
}

namespace {

// One output-fusible edge: the consumer `src` reads `dst`'s `resultIdx` result
// (guarded by `epilogueOp`), either untransformed (`reshape` null) or through
// the single `reshape` of the loop index.
struct OutputFusionEdge {
  IterOpaqueOp dst;
  unsigned resultIdx;
  IterEpilogueOp epilogueOp;
  IterForeachOp src;
  ReshapeIndexOp reshape;
};

// The `map.iter.epilogue` for `resultIdx`, or null (e.g. a chain result).
IterEpilogueOp getEpilogueForResult(IterOpaqueOp op, unsigned resultIdx) {
  Value outArg = op.getOutputBlockArgs()[resultIdx];
  for (auto epilogueOp : op.getBody().front().getOps<IterEpilogueOp>())
    if (llvm::is_contained(epilogueOp.getCaptures(), outArg))
      return epilogueOp;
  return nullptr;
}

// Whether every use of `fusionArg` reads at the loop index.
bool readsAtLoopIndex(BlockArgument fusionArg, Value loopIdx) {
  for (Operation *user : fusionArg.getUsers()) {
    auto load = dyn_cast<LoadOp>(user);
    if (!load || load.getIndex() != loopIdx)
      return false;
  }
  return true;
}

// The single `map.index.reshape` of the loop index that every use of
// `fusionArg` loads through, or null if a use reads directly, through another
// transform, or through a different reshape.
ReshapeIndexOp reshapeOnFusedInput(Block &srcBody, BlockArgument fusionArg) {
  BlockArgument loopIdx = srcBody.getArguments().back();
  ReshapeIndexOp found = nullptr;
  for (Operation *user : fusionArg.getUsers()) {
    auto load = dyn_cast<LoadOp>(user);
    if (!load)
      return nullptr;
    auto reshape = load.getIndex().getDefiningOp<ReshapeIndexOp>();
    if (!reshape || reshape.getIndex() != loopIdx)
      return nullptr;
    if (found && found != reshape)
      return nullptr;
    found = reshape;
  }
  return found;
}

// The block arg of the single `src` input that reads `dstResult`.
BlockArgument fusedInputArg(IterForeachOp src, Value dstResult) {
  auto it = llvm::find(src.getInputs(), dstResult);
  return src.getBody().front().getArgument(it - src.getInputs().begin());
}

// The first load of `fusionArg` (all its uses are loads by the checks above).
LoadOp firstLoadOf(BlockArgument fusionArg) {
  for (Operation *user : fusionArg.getUsers())
    if (auto load = dyn_cast<LoadOp>(user))
      return load;
  return nullptr;
}

// The first output-fusible edge on `dst` that `config` accepts. Read-only.
std::optional<OutputFusionEdge>
findFusibleEdge(IterOpaqueOp dst, const OutputFusionConfig &config) {
  for (unsigned resultIdx = 0; resultIdx < dst.getNumNonChainResults();
       ++resultIdx) {
    Value res = dst->getResult(resultIdx);
    if (!res.hasOneUse())
      continue;
    IterEpilogueOp epilogueOp = getEpilogueForResult(dst, resultIdx);
    if (!epilogueOp)
      continue;
    Operation *user = res.getUses().begin()->getOwner();
    // Fusion stays within a block: never across a control-flow boundary.
    if (user->getBlock() != dst->getBlock())
      continue;
    auto src = dyn_cast<IterForeachOp>(user);
    if (!src || src.getNumResults() != 1)
      continue;
    if (llvm::count(src.getInputs(), res) != 1)
      continue;

    // A chain-returning consumer is `mo.mutable.store`: its buffer INPUT is the
    // store target (see `lowerMutableStoreToForeach`), so the epilogue's store
    // is pointed at that buffer instead of `dst`'s own output.
    if (isa<MO::ChainType>(src.getResult(0).getType())) {
      // A fused kernel must not write a buffer it also reads.
      if (llvm::any_of(src->getOperands(), [&](Value srcOperand) {
            return isa<MO::BufferType>(srcOperand.getType()) &&
                   llvm::is_contained(dst->getOperands(), srcOperand);
          }))
        continue;
      // `dst` holds no chain, so a used output chain would lose its ordering.
      // TODO(GEX-3964): give `dst` a chain in/out the way
      // `FuseMutableLoads::fuseLoadBufferAndChains` does, and fuse either way.
      if (!src.getResult(0).use_empty())
        continue;
    }

    Block &srcBody = src.getBody().front();
    BlockArgument fusionArg = fusedInputArg(src, res);
    BlockArgument loopIdx = srcBody.getArguments().back();

    // Only a single reshape can be inverted onto the store (it preserves the
    // row-major offset).
    ReshapeIndexOp reshape = nullptr;
    if (!readsAtLoopIndex(fusionArg, loopIdx)) {
      reshape = reshapeOnFusedInput(srcBody, fusionArg);
      if (!reshape)
        continue;
    }

    // Reject if another operand transitively depends on `dst`: fusing would
    // create a cycle.
    if (llvm::any_of(src->getOperands(), [&](Value other) {
          return other != res && dataflowGraphDependsOn(other, dst);
        }))
      continue;

    auto epilogueStore = getUniqueStore(epilogueOp.getBody().front());
    if (config.shouldFuse(epilogueStore, firstLoadOf(fusionArg)))
      return OutputFusionEdge{dst, resultIdx, epilogueOp, src, reshape};
  }
  return std::nullopt;
}

// Splices `src`'s per-element compute into `epilogueOp`. For a reshape read,
// the reshape is relocated onto the store index with in/out swapped, since the
// store direction is the inverse of the load.
void fuseOutputOnce(const OutputFusionEdge &edge, PatternRewriter &rewriter) {
  IterOpaqueOp dst = edge.dst;
  IterEpilogueOp epilogueOp = edge.epilogueOp;
  IterForeachOp src = edge.src;
  ReshapeIndexOp reshape = edge.reshape;
  MLIRContext *ctx = dst.getContext();
  Value dstResult = dst->getResult(edge.resultIdx);
  Block &srcBody = src.getBody().front();
  Block &epilogueBody = epilogueOp.getBody().front();

  // May already reflect earlier fusions (e.g. a reshaped store index).
  auto epilogueStore = getUniqueStore(epilogueBody);
  Value val = epilogueStore.getValue();
  Value storeIdx = epilogueStore.getIndex();
  Value cap = epilogueStore.getTensor();
  Location storeLoc = epilogueStore.getLoc();

  Value storeTransform = storeIdx;
  if (reshape) {
    ViewSignatureAttr origSig = reshape.getSpec().getSignature();
    auto swappedSpec = ReshapeSpecAttr::get(
        ctx, ViewSignatureAttr::get(ctx, origSig.getOut(), origSig.getIn()));
    rewriter.setInsertionPoint(epilogueStore);
    storeTransform = ReshapeIndexOp::create(
        rewriter, storeLoc, IndexType::get(ctx, origSig.getOut().getRank()),
        swappedSpec, storeIdx);
  }

  auto fusionArgIt = llvm::find(src.getInputs(), dstResult);
  unsigned fusionArgIdx = fusionArgIt - src.getInputs().begin();
  BlockArgument fusionArg = srcBody.getArgument(fusionArgIdx);

  // `val` is the scalar `dst` produced for this element; it replaces every
  // load of the fused operand.
  for (Operation *user : llvm::to_vector(fusionArg.getUsers())) {
    auto load = cast<LoadOp>(user);
    load.getResult().replaceAllUsesWith(val);
    load.erase();
  }
  if (reshape)
    reshape.erase();

  // Every other input becomes a fresh epilogue capture, threaded through a new
  // shadowed input of `dst` (the epilogue is isolated from above).
  for (auto [i, input] : llvm::enumerate(src.getInputs())) {
    if (i == fusionArgIdx)
      continue;
    unsigned dstInputIdx = dst.getInputs().size();
    dst.insertInputsAt(dstInputIdx, input);
    Value dstInputArg = dst.getInputBlockArgs()[dstInputIdx];
    epilogueOp.getCapturesMutable().append(dstInputArg);
    BlockArgument capArg =
        epilogueBody.addArgument(input.getType(), input.getLoc());
    srcBody.getArgument(i).replaceAllUsesWith(capArg);
  }

  // The loop index is the body's last argument for both pointwise and
  // `mo.mutable.store` consumers; rewire it to the store index.
  auto srcStore = getUniqueStore(srcBody);
  Value finalVal = srcStore.getValue();
  // A `mo.mutable.store` consumer's buffer input is already an epilogue
  // capture; the store is rebuilt on it.
  Value storeTargetBuffer;
  if (isa<MO::ChainType>(src.getResult(0).getType()))
    storeTargetBuffer = srcStore.getTensor();
  srcStore.erase();
  srcBody.getArguments().back().replaceAllUsesWith(storeTransform);
  srcBody.eraseArguments(0, srcBody.getNumArguments());

  // Retype the store-target path when the consumer changes the result's type.
  // For a reshape read the epilogue's RESULT keeps the producer's shape, which
  // the kernel (e.g. matmul) binds; the reshaped index reaches the destination.
  if (storeTargetBuffer) {
    cap = storeTargetBuffer;
    dst.setHasWriteEffect(true);
  }

  Type fusedTy = src.getResult(0).getType();
  if (!storeTargetBuffer && fusedTy != dstResult.getType()) {
    dstResult.setType(fusedTy);
    dst.getOutputBlockArgs()[edge.resultIdx].setType(fusedTy);
    cap.setType(fusedTy);
    if (!reshape)
      epilogueOp.getResult().setType(fusedTy);
  }

  epilogueStore.erase();
  rewriter.inlineBlockBefore(&srcBody, &epilogueBody, epilogueBody.end());
  rewriter.setInsertionPointToEnd(&epilogueBody);
  StoreOp::create(rewriter, storeLoc, cap, storeTransform, finalVal);

  // A `mo.mutable.store` consumer yields only an unused chain.
  if (!storeTargetBuffer)
    src->replaceAllUsesWith(ValueRange{dstResult});
}

} // namespace

unsigned fuseOutput(MO::GraphOp graphOp, const OutputFusionConfig &config) {
  PatternRewriter rewriter(graphOp.getContext());

  // Kernels live in the graph body and in any nested control-flow regions (an
  // `mo.if`'s branches); a producer/consumer edge is always within one block.
  llvm::SmallSetVector<Block *, 8> blocks;
  graphOp.getBody().walk(
      [&](IterOpaqueOp op) { blocks.insert(op->getBlock()); });

  // Producers before consumers. Fusion erases only the consumer and mutates
  // the producer in place.
  MO::ParametricTopologyAnalysis topo(graphOp);
  DenseSet<Operation *> erased;
  unsigned numFused = 0;
  for (Block *block : blocks) {
    SmallVector<Operation *> ops;
    for (Operation &op : *block)
      ops.push_back(&op);
    topo.computeTopologicalSorting(ops);

    for (Operation *op : ops) {
      if (erased.contains(op))
        continue;
      auto dst = dyn_cast<IterOpaqueOp>(op);
      if (!dst)
        continue;
      while (std::optional<OutputFusionEdge> edge =
                 findFusibleEdge(dst, config)) {
        fuseOutputOnce(*edge, rewriter);
        dst.setFusionInfoAttr(MOGG::createFusionInfo<MOGG::FusionEpilogueAttr>(
            dst.getContext(), {dst.getOperation(), edge->src.getOperation()},
            {static_cast<uint32_t>(edge->resultIdx)}));
        dst->setLoc(mlir::FusedLoc::get(dst.getContext(),
                                        {edge->src->getLoc(), dst->getLoc()}));
        erased.insert(edge->src);
        edge->src->erase();
        dedupeKernelInputs(dst, rewriter);
        ++numFused;
      }
    }
  }
  return numFused;
}

namespace {

// Unconstrained output fusion (fuse every eligible edge). For testing.
struct FuseOutputPass : public MAP::impl::FuseOutputBase<FuseOutputPass> {
  using Base::Base;

  void runOnOperation() override {
    OutputFusionConfig config;
    config.shouldFuse = [](StoreOp, LoadOp) { return true; };
    fuseOutput(getOperation(), config);
  }
};

} // namespace

} // namespace M::MAP
