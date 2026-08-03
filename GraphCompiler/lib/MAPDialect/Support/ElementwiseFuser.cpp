//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Support/ElementwiseFuser.h"

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MOGGDialect/Analysis/SequenceFusionAnalysis.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"

using namespace M;

namespace M::MAP {

namespace {

// The max number of elements in a tensor operand above which re-reading it is
// considered expensive. Mirrors MOGG's FuseElementwiseKernels.cpp; see there
// for the full duplication-vs-materialization rationale (JSC-30 / MSTDL-2759).
constexpr int64_t kLargeTensorElementThreshold = 4096;

static bool isLargeTensorOperand(Type type) {
  auto tensorType = dyn_cast<MO::TensorType>(type);
  if (!tensorType)
    return false;

  std::optional<int64_t> numElements = tensorType.getNumElements();
  if (!numElements.has_value())
    return true;

  return *numElements > kLargeTensorElementThreshold;
}

// A `map.iter.foreach` is cheap to duplicate into more than one consumer iff it
// has at most one large tensor input: duplication re-reads all of its inputs
// once per extra consumer, which is only a net win when there is at most one
// expensive re-read.
static bool isCheapToDuplicate(IterForeachOp op) {
  int numLargeOperands = 0;
  for (Value input : op.getInputs())
    if (isLargeTensorOperand(input.getType()))
      if (++numLargeOperands > 1)
        return false;
  return true;
}

// Every `map.iter.foreach` produced today is a pure pointwise kernel (one
// result, body = load(s) -> elementwise -> store); there is no view/epilogue
// variant yet, so eligibility reduces to "is this a map.iter.foreach".
static bool isAnElementwiseForeach(Operation *op) {
  auto foreachOp = dyn_cast<IterForeachOp>(op);
  return foreachOp && foreachOp.getNumResults() == 1;
}

// Mirrors MOGG's checkMutableKernelFusible (FuseElementwiseKernels.cpp):
// guards against fusing across a chain-carrying kernel (e.g. mo.mutable.store)
// when doing so could change effective ordering relative to another consumer
// of the same chain.
static bool checkMutableKernelFusible(Value srcInChain, Value srcOutChain,
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

// Fusing means splicing `src`'s per-element body into `dst`'s shared index
// space, so both must iterate the exact same shape. (`dynamic_dimensions` is
// always empty at this point in the pipeline -- it is only populated later,
// inside a kernel-stub body after `mogg-to-mgp` -- so there is nothing else to
// reconcile there.) Chain operands are already excluded from `dst`'s `inputs`
// (they live in a separate `chain` field), matching MOGG's explicit chain-skip
// in getFusionIndices; checkMutableKernelFusible covers the remaining safety.
static bool foreachKernelsAreCompatible(Operation *src, Operation *dst) {
  auto srcOp = cast<IterForeachOp>(src);
  auto dstOp = cast<IterForeachOp>(dst);
  if (srcOp.getShape() != dstOp.getShape())
    return false;

  return checkMutableKernelFusible(
      srcOp.getInputChain(), srcOp.getOutputChain(), dstOp.getInputChain(),
      dstOp.getOutputChain());
}

} // namespace

ElementwiseFuser::ElementwiseFuser(MO::GraphOp graphOp) : graphOp(graphOp) {
  MO::ParametricTopologyAnalysis topologyAnalysis(graphOp);

  MOGG::EligibilityFunction isEligible = isAnElementwiseForeach;
  MOGG::CompatibilityFunction areCompatible = foreachKernelsAreCompatible;
  MOGG::CanDuplicate canDuplicate = [](Operation *op) {
    auto foreachOp = cast<IterForeachOp>(op);
    return !MO::isOnHost(foreachOp) && isCheapToDuplicate(foreachOp);
  };

  /// TODO: MOCO-1469 Remove this limit (mirrors mogg-fuse-elementwise).
  static constexpr uint64_t kMaximumFusionSetSize = 64;
  subgraphs = MOGG::sourceDestinationSequenceFusionAnalysis(
      graphOp.getBody(), topologyAnalysis, isEligible, areCompatible,
      canDuplicate, kMaximumFusionSetSize);

  for (auto &subgraph : subgraphs)
    topologyAnalysis.computeTopologicalSorting(subgraph);
}

void ElementwiseFuser::fuse() {
  for (ArrayRef<Operation *> subgraph : subgraphs)
    fuseSubgraph(subgraph);

  // Erase fused-away kernels once they have no remaining users. A DenseSet
  // guards against a double-erase when duplication placed the same op in
  // more than one subgraph.
  DenseSet<Operation *> erased;
  for (ArrayRef<Operation *> subgraph : subgraphs) {
    for (Operation *op : llvm::reverse(subgraph.drop_back())) {
      if (!erased.contains(op) && op->getUsers().empty()) {
        erased.insert(op);
        op->erase();
      }
    }
  }
}

void ElementwiseFuser::fuseSubgraph(ArrayRef<Operation *> subgraph) {
  auto dst = cast<IterForeachOp>(subgraph.back());
  PatternRewriter rewriter(dst.getContext());
  Block &dstBody = dst.getBody().front();

  // Process producers nearest to `dst` first (reverse topological order): by
  // the time a farther producer is reached, the nearer one has already
  // spliced its own inputs directly into `dst`'s operand list, so every `src`
  // below is always found as one of `dst`'s CURRENT inputs.
  for (Operation *op : llvm::reverse(subgraph.drop_back())) {
    auto src = cast<IterForeachOp>(op);

    // Every position in dst's current inputs fed by src's result. There can
    // be more than one (e.g. `mo.mul(src, src)`); each gets an independent
    // copy of src's compute spliced in, mirroring MOGG's elementwise fuser.
    SmallVector<unsigned> fusionIndices;
    for (auto [i, input] : llvm::enumerate(dst.getInputs()))
      if (input == src.getResult(0))
        fusionIndices.push_back(i);
    assert(!fusionIndices.empty() &&
           "src must be a current input of dst inside its own subgraph");

    for (unsigned argIdx : llvm::reverse(fusionIndices)) {
      unsigned numSrcInputs = src.getInputs().size();

      // Pull src's own inputs in as new dst inputs, positioned directly
      // before the tensor argument they are about to replace.
      dst.insertInputsAt(argIdx, src.getInputs());
      ValueRange newDstArgs =
          dstBody.getArguments().slice(argIdx, numSrcInputs);
      BlockArgument oldTensorArg = dstBody.getArgument(argIdx + numSrcInputs);

      // Clone src's body so the original is left untouched: it may be
      // spliced in at more than one fusion index above, or duplicated across
      // more than one subgraph (see `canDuplicate`).
      rewriter.cloneRegionBefore(src.getBody(), *dstBody.getParent(),
                                 dstBody.getParent()->begin());
      Block &srcClone = dstBody.getParent()->front();

      // The scalar src would have written to its output tensor is exactly
      // the value dst needs in place of a load of src's tensor result.
      auto storeOp = cast<StoreOp>(srcClone.back());
      Value fusedVal = storeOp.getValue();
      storeOp.erase();
      // The output argument's only use was the store above; drop it now
      // that it is unused, so `inlineBlockBefore` below need not replace it.
      srcClone.eraseArgument(numSrcInputs);

      // Redirect every load of the old tensor argument to the fused value --
      // erase below might corrupt the use-def list, so snapshot uses first.
      SmallVector<Operation *> loads(oldTensorArg.getUsers().begin(),
                                     oldTensorArg.getUsers().end());
      for (Operation *user : loads) {
        auto load = cast<LoadOp>(user);
        load.getResult().replaceAllUsesWith(fusedVal);
        load.erase();
      }
      // Inline src's clone (its load(s) + elementwise call) at the front of
      // dst's body, wiring its own inputs to the new dst arguments above and
      // reusing dst's shared index argument -- every fused op iterates the
      // exact same index space. This also erases `srcClone`, restoring dst's
      // body region to holding a single block again.
      SmallVector<Value> fusedArgs(newDstArgs.begin(), newDstArgs.end());
      fusedArgs.push_back(dstBody.getArguments().back());
      rewriter.inlineBlockBefore(&srcClone, &dstBody, dstBody.begin(),
                                 fusedArgs);

      // Only now erase the old (fully unused) tensor argument: `eraseInputsAt`
      // erases via dst's body region, which must hold a single block (it
      // would otherwise target the wrong block while `srcClone` sat at the
      // region's front, right after `cloneRegionBefore` above).
      dst.eraseInputsAt(argIdx + numSrcInputs);
    }

    dst.setFusionInfoAttr(MOGG::createFusionInfo<MOGG::FusionElementwiseAttr>(
        src->getContext(), {src, dst}));
    // Merge source locations so debugging/error-reporting tooling downstream
    // still sees both original ops.
    dst->setLoc(
        mlir::FusedLoc::get(src->getContext(), {src->getLoc(), dst->getLoc()}));
    // Mirrors MOGG's fuseKernelsIntoLastKernel: dst now also executes src's
    // body, so dst's declared effects must cover whatever src had.
    dst.setHasReadEffect(src.getHasReadEffect() || dst.getHasReadEffect());
    dst.setHasWriteEffect(src.getHasWriteEffect() || dst.getHasWriteEffect());
  }
}

} // namespace M::MAP
