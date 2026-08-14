//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include <functional>
#include <optional>

using namespace M;

#define DEBUG_TYPE "map-fuse-mutable-loads"

namespace M::MAP {

#define GEN_PASS_DEF_FUSEMUTABLELOADS
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

// Both kernel ops load fusion applies to -- `map.iter.foreach` and
// `map.iter.opaque` -- carry a chain, an `$inputs` list shadowed as body block
// arguments, and a `fusionInfo` attribute, but each behind its own accessors.
// These bridge the two so the fusion logic below is written once against
// `Operation *`.

bool isFusibleKernel(Operation *op) {
  return isa_and_present<MAP::IterForeachOp, MAP::IterOpaqueOp>(op);
}

Value getKernelInputChain(Operation *kernelOp) {
  if (auto foreachOp = dyn_cast<MAP::IterForeachOp>(kernelOp))
    return foreachOp.getInputChain();
  return cast<MAP::IterOpaqueOp>(kernelOp).getInputChain();
}

Value getKernelOutputChain(Operation *kernelOp) {
  if (auto foreachOp = dyn_cast<MAP::IterForeachOp>(kernelOp))
    return foreachOp.getOutputChain();
  return cast<MAP::IterOpaqueOp>(kernelOp).getOutputChain();
}

void setKernelChain(Operation *kernelOp, Value chain) {
  if (auto foreachOp = dyn_cast<MAP::IterForeachOp>(kernelOp))
    return foreachOp.setChain(chain);
  return cast<MAP::IterOpaqueOp>(kernelOp).setChain(chain);
}

void insertKernelInputAt(Operation *kernelOp, unsigned idx, Value val) {
  if (auto foreachOp = dyn_cast<MAP::IterForeachOp>(kernelOp))
    return foreachOp.insertInputsAt(idx, val);
  return cast<MAP::IterOpaqueOp>(kernelOp).insertInputsAt(idx, val);
}

void eraseKernelInputAt(Operation *kernelOp, unsigned idx) {
  if (auto foreachOp = dyn_cast<MAP::IterForeachOp>(kernelOp))
    return foreachOp.eraseInputsAt(idx);
  return cast<MAP::IterOpaqueOp>(kernelOp).eraseInputsAt(idx);
}

Block &getKernelBody(Operation *kernelOp) {
  if (auto foreachOp = dyn_cast<MAP::IterForeachOp>(kernelOp))
    return foreachOp.getBody().front();
  return cast<MAP::IterOpaqueOp>(kernelOp).getBody().front();
}

void setKernelFusionInfo(Operation *kernelOp, MOGG::FusionInfoAttr fusionInfo) {
  if (auto foreachOp = dyn_cast<MAP::IterForeachOp>(kernelOp))
    return foreachOp.setFusionInfoAttr(fusionInfo);
  return cast<MAP::IterOpaqueOp>(kernelOp).setFusionInfoAttr(fusionInfo);
}

bool kernelHasWriteEffect(Operation *kernelOp) {
  if (auto foreachOp = dyn_cast<MAP::IterForeachOp>(kernelOp))
    return foreachOp.getHasWriteEffect();
  if (auto opaqueOp = dyn_cast<MAP::IterOpaqueOp>(kernelOp))
    return opaqueOp.getHasWriteEffect();
  return false;
}

/// Determines if there's a valid kernel op to perform load fusion with, and
/// returns it (null if there is none).
///
/// The main condition is that the kernel must be the sole consumer of the
/// tensor output of the load op.
Operation *maybeGetCompatibleKernel(MO::MutableLoadOp loadOp) {

  auto tensor = loadOp.getOutTensor();

  if (!tensor.hasOneUse()) {
    LLVM_DEBUG(loadOp.emitRemark()
               << "Mutable load fusion is disabled for this load op as its "
                  "output tensor has more than one use.");

    return nullptr;
  }

  if (loadOp->getBlock() != (*tensor.getUsers().begin())->getBlock()) {
    LLVM_DEBUG(
        loadOp.emitRemark()
        << "Mutable load fusion is disabled for this load op as its "
           "output tensor is produced in a different block than its user.");
    return nullptr;
  }

  Operation *kernelOp = *tensor.getUsers().begin();
  if (!isFusibleKernel(kernelOp))
    return nullptr;

  // Fusing makes the kernel read the buffer as it iterates, in place of the
  // snapshot the load took beforehand. That only preserves the load's semantics
  // if the kernel visits each element at most once and writes only the element
  // it is reading -- which is exactly what map.iter.foreach is, by
  // construction. An in-place map.iter.opaque wraps a hand-written kernel free
  // to write any element while reading any other, so it has to keep reading the
  // snapshot.
  if (isa<MAP::IterOpaqueOp>(kernelOp) &&
      llvm::is_contained(kernelOp->getOperands(), loadOp.getInBuffer())) {
    LLVM_DEBUG(loadOp.emitRemark()
               << "Mutable load fusion is disabled for this load op as the "
                  "opaque kernel consuming it also writes its buffer.");
    return nullptr;
  }

  return kernelOp;
}

IterForeachOp fuseLoadBufferAndChains(PatternRewriter &rewriter,
                                      MO::MutableLoadOp loadOp,
                                      IterForeachOp kernelOp) {
  MLIRContext *ctx = rewriter.getContext();

  auto chainType = MO::ChainType::get(ctx);

  auto &tensorUse = *loadOp.getOutTensor().getUses().begin();
  auto argIdx = tensorUse.getOperandNumber();

  // Build the new operand list: the fused tensor input is replaced in place
  // by the load's buffer at the same index.
  SmallVector<Value> newInputs(kernelOp.getInputs());
  newInputs[argIdx] = loadOp.getInBuffer();

  SmallVector<Type> newResultTypes(TypeRange(kernelOp.getResults()));
  newResultTypes.push_back(chainType);

  rewriter.setInsertionPointAfter(kernelOp);

  auto newKernelOp =
      IterForeachOp::create(rewriter, kernelOp->getLoc(),
                            /*resultTypes=*/newResultTypes,
                            /*shape=*/kernelOp.getShape(),
                            /*inputs=*/newInputs,
                            /*chain=*/loadOp.getInChain(),
                            /*fusionInfo=*/
                            MOGG::createFusionInfo<MOGG::FusionMutableLoadAttr>(
                                kernelOp->getContext(), {loadOp, kernelOp}),
                            /*hasReadEffect=*/true,
                            /*hasWriteEffect=*/kernelOp.getHasWriteEffect());

  // Merge source locations so the fused kernel keeps both ops' provenance.
  newKernelOp->setLoc(mlir::FusedLoc::get(
      kernelOp->getContext(), {loadOp->getLoc(), kernelOp->getLoc()}));

  // Copy over any discardable attributes from the original kernel
  newKernelOp->setDiscardableAttrs(kernelOp->getDiscardableAttrDictionary());

  // A map.iter.foreach body is just a block whose arguments mirror the op's own
  // operand/result types, so there is nothing to clone it by structure. Build
  // that block explicitly (with the buffer type substituted at argIdx), then
  // splice the old kernel's body into it via cloneRegionBefore +
  // inlineBlockBefore (the same idiom ElementwiseFuser::fuse() uses), remapping
  // the old tensor block-arg onto the new buffer-typed one.
  Block &oldBody = kernelOp.getBody().front();
  Block *newBody = &newKernelOp.getBody().emplaceBlock();
  for (auto [i, arg] : llvm::enumerate(oldBody.getArguments())) {
    Type argType = i == argIdx ? loadOp.getInBuffer().getType() : arg.getType();
    newBody->addArgument(argType, arg.getLoc());
  }

  rewriter.cloneRegionBefore(kernelOp.getBody(), newKernelOp.getBody(),
                             newKernelOp.getBody().begin());
  Block &oldBodyClone = newKernelOp.getBody().front();

  SmallVector<Value> fusedArgs(newBody->getArguments().begin(),
                               newBody->getArguments().end());
  rewriter.inlineBlockBefore(&oldBodyClone, newBody, newBody->begin(),
                             fusedArgs);

  auto outChainIdx = newKernelOp.getNumResults() - 1;
  Value outChain = newKernelOp->getResult(outChainIdx);

  loadOp.getOutChain().replaceAllUsesWith(outChain);

  for (auto &&[oldRes, newRes] :
       llvm::zip(kernelOp->getResults(), newKernelOp->getResults())) {
    oldRes.replaceAllUsesWith(newRes);
  }

  kernelOp->erase();
  loadOp->erase();

  return newKernelOp;
}

// Replaces `loadOutChain` with `loadInChain` in all chain.create ops on the
// path from a mutable.load op to a kernel op.
// Attempts to "detach" the load's outgoing chain from the kernel's incoming
// chain by rewriting an exclusive path of `mo.chain.create` operations so they
// consume the load's *input* chain instead.
// Abandons the rewrite if the path contains anything other than single-use
// chain.create ops.
//
// Enables fusion when the kernel depends on the load's out-chain through an
// exclusive path of single-use chain.create operations.
//
// Problem: Direct fusion would create a cycle (kernel depends on load's
//          out-chain, but load would be fused into kernel).
// Solution: Rewrite chain.creates on the path to consume load's in-chain
//           instead, "detaching" the load's out-chain from the dependency.
//
// Example:
// BEFORE (fusion blocked):
//   %t, %outCh = mo.mutable.load(buf, %inCh)
//   %merged = mo.chain.create(%other, %outCh)  // kernel depends on %outCh
//   %result = mo.custom(..., %t, %merged)      // can't fuse: creates cycle
//
// AFTER DETACHMENT (fusion enabled):
//   %t, %outCh = mo.mutable.load(buf, %inCh)   // %outCh becomes dead
//   %merged = mo.chain.create(%other, %inCh)   // %outCh -> %inCh
//   %result = map.iter.foreach(buf, ..., %merged)  // fused!
//     fusionInfo = MutableLoad(...)
//
// Requires: %outCh and %merged have single use (exclusivity constraint).
static bool detachExclusiveChainPath(Value loadOutChain, Value loadInChain,
                                     Value kernelInChain) {
  if (kernelInChain == loadOutChain) {
    // Direct dependency: nothing to rewrite safely here.
    return false;
  }

  // Collect every chain.create that sits on the unique path from the load's
  // out-chain to the kernel's in-chain.
  SmallVector<MO::ChainCreateOp> toRewrite;
  llvm::SmallPtrSet<Operation *, 16> onPath;

  // Returns:
  //   * std::nullopt - this branch never reaches the load's out-chain.
  //   * false        - the branch touches the load but fails exclusivity checks
  //                    (non-chain.create op or extra users).
  //   * true         - the branch is exclusively composed of single-use
  //                    chain.create ops and is safe to rewrite.
  std::function<std::optional<bool>(Value)> analyzeChainValue =
      [&](Value v) -> std::optional<bool> {
    if (v == loadOutChain)
      return true;

    auto chainCreate = v.getDefiningOp<MO::ChainCreateOp>();
    if (!chainCreate)
      return std::nullopt;

    bool subtreeReachesLoad = false;
    for (Value operand : chainCreate.getInputs()) {
      if (operand == loadOutChain) {
        subtreeReachesLoad = true;
        continue;
      }

      if (std::optional<bool> pathStatus = analyzeChainValue(operand)) {
        if (!*pathStatus) {
          // Bail since dependency exists but is not exclusive.
          return false;
        }
        // If we're here, pathStatus is true (on path and exclusive).
        subtreeReachesLoad = true;
      }
    }

    if (!subtreeReachesLoad) {
      // This subtree is not on the load -> kernel path.
      return std::nullopt;
    }

    // Check exclusivity for chain.creates that are on the path.
    if (!chainCreate.getResult().hasOneUse()) {
      // Bail since another user would observe rewiring the chain.
      return false;
    }

    // Rewrite this chain.create since it reaches the load op and is exclusive.
    if (onPath.insert(chainCreate.getOperation()).second)
      toRewrite.push_back(chainCreate);

    return true;
  };

  // Abort if any branch does not satisfy the exclusivity requirements.
  if (analyzeChainValue(kernelInChain) != true)
    return false;

  // Rewrite each chain.create that was verified to be on the exclusive path to
  // use the load's input chain instead of its output chain.
  for (MO::ChainCreateOp chainCreate : toRewrite)
    chainCreate->replaceUsesOfWith(loadOutChain, loadInChain);

  // Sanity-check that the rewrite fully severed the dependency.
  ASSERT_STREAM(!MOGG::valueDependsTransitivelyOn(kernelInChain, loadOutChain),
                "Expected exclusive chain path from mutable.load to kernel op "
                "to be severed.");

  return true;
}

// An opaque kernel reads a fusible input through a `map.iter.prologue` that
// captures the corresponding body block argument. The capture's own inner block
// argument mirrors the captured type, so it has to follow the block argument's
// tensor -> buffer retype; the `map.load` it feeds already reads either. Inputs
// the kernel is not fused into instead reach `map.call.opaque` directly, whose
// operands are untyped and need no retype.
void retypeFusedPrologueCaptures(Value bufferArg) {
  for (OpOperand &use : bufferArg.getUses()) {
    auto prologueOp = dyn_cast<MAP::IterPrologueOp>(use.getOwner());
    if (!prologueOp)
      continue;

    // Prologue body block args are [index, captures...].
    prologueOp.getBody()
        .getArgument(use.getOperandNumber() + 1)
        .setType(bufferArg.getType());
  }
}

void fuseLoadBuffer(PatternRewriter &rewriter, MO::MutableLoadOp loadOp,
                    Operation *kernelOp, bool kernelHasLoadsOutputChain) {
  auto buffer = loadOp.getInBuffer();
  auto tensor = loadOp.getOutTensor();

  ASSERT_STREAM(*tensor.getUsers().begin() == kernelOp,
                "Expected value semantic tensor from load op to only be "
                "consumed by current kernel op");

  OpOperand &operand = *tensor.getUses().begin();
  auto argIdx = operand.getOperandNumber();

  Block &kernelBody = getKernelBody(kernelOp);

  TypedValue<MO::ChainType> loadOutChain = loadOp.getOutChain();
  if (Value kernelInChain = getKernelInputChain(kernelOp)) {
    if (kernelInChain != loadOutChain &&
        MOGG::valueDependsTransitivelyOn(kernelInChain, loadOutChain) &&
        !detachExclusiveChainPath(loadOutChain, loadOp.getInChain(),
                                  kernelInChain)) {
      LLVM_DEBUG(loadOp.emitRemark() << "Skipping mutable load fusion to avoid "
                                        "introducing a chain cycle.");
      return;
    }
  }

  insertKernelInputAt(kernelOp, argIdx, buffer);
  Value newBufferArg = kernelBody.getArgument(argIdx);

  auto oldBlockArg = kernelBody.getArgument(argIdx + 1);
  oldBlockArg.replaceAllUsesWith(newBufferArg);
  eraseKernelInputAt(kernelOp, argIdx + 1);

  retypeFusedPrologueCaptures(newBufferArg);

  if (!kernelHasLoadsOutputChain) {
    loadOutChain.replaceAllUsesWith(getKernelOutputChain(kernelOp));

    rewriter.setInsertionPoint(kernelOp);

    auto newChain = rewriter.createOrFold<MO::ChainCreateOp>(
        loadOp.getLoc(),
        ValueRange{getKernelInputChain(kernelOp), loadOp.getInChain()});
    setKernelChain(kernelOp, newChain);
  } else {
    // Simply rewrite the loads input chain when the kernel has the loads output
    // chain as input.
    setKernelChain(kernelOp, loadOp.getInChain());
  }

  setKernelFusionInfo(kernelOp,
                      MOGG::createFusionInfo<MOGG::FusionMutableLoadAttr>(
                          kernelOp->getContext(), {loadOp, kernelOp}));
  // Merge source locations so the fused kernel keeps both ops' provenance.
  kernelOp->setLoc(mlir::FusedLoc::get(kernelOp->getContext(),
                                       {loadOp->getLoc(), kernelOp->getLoc()}));

  loadOp.erase();
}

/// Turns a mo.mutable.load op with n uses into n mo.mutable.load ops each with
/// a single use of their value semantic tensor output.
struct ParallelizeLoadsWithMultipleUsers
    : MO::DeviceAwareRewritePattern<OpRewritePattern, MO::MutableLoadOp> {
  ParallelizeLoadsWithMultipleUsers(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::MutableLoadOp loadOp,
                             MO::DeviceAwareRewriter &rewriter) const override {
    auto tensor = loadOp.getOutTensor();
    auto loadBuffer = loadOp.getInBuffer();

    if (tensor.use_empty() || tensor.hasOneUse()) {
      return rewriter.notifyMatchFailure(
          loadOp.getLoc(),
          "Output value semantic tensor has <= 1 uses, skipping "
          "parallelization pattern.");
    }

    // Splitting the load lets each consumer fuse the buffer and read it
    // directly, so a consumer that writes that same buffer would race the reads
    // the other consumers now do inline. Keep the single shared load in that
    // case, which also keeps the buffer out of every consumer.
    for (OpOperand &use : tensor.getUses()) {
      Operation *consumer = use.getOwner();
      if (!kernelHasWriteEffect(consumer))
        continue;

      if (llvm::is_contained(consumer->getOperands(), loadBuffer)) {
        return rewriter.notifyMatchFailure(
            loadOp.getLoc(),
            "Parallelization blocked: a consumer kernel has write effect and "
            "shares the same buffer as the load operation.");
      }
    }

    auto *loadBlock = loadOp->getBlock();

    // Check if any uses are in a different block - if so, early exit
    for (OpOperand &use : tensor.getUses()) {
      if (use.getOwner()->getBlock() != loadBlock) {
        // TODO(GEX-1203): Add support for cross-block load parallelization
        return rewriter.notifyMatchFailure(
            loadOp.getLoc(),
            "Found use of output tensor in different block than load op, "
            "skipping parallelization to maintain cross-block semantics");
      }
    }

    auto chainType = MO::ChainType::get(loadOp.getContext());
    auto outTensor = loadOp.getOutTensor();

    llvm::SmallVector<Value> chains;
    llvm::SmallVector<OpOperand *> opOperands;
    for (OpOperand &use : outTensor.getUses())
      opOperands.push_back(&use);

    for (OpOperand *use : opOperands) {
      auto newLoadOp = rewriter.create<MO::MutableLoadOp>(
          loadOp.getLoc(), outTensor.getType(), chainType, loadOp.getInBuffer(),
          loadOp.getInChain());

      rewriter.modifyOpInPlace(use->getOwner(), [&newLoadOp, &use]() {
        use->set(newLoadOp.getOutTensor());
      });

      chains.push_back(newLoadOp.getOutChain());
    }

    // Want the generated mo.chain.create to show up in a semi-reasonable spot.
    for (Operation *user : loadOp.getOutChain().getUsers()) {
      rewriter.setInsertionPoint(user);
      break;
    }

    Value newChain =
        rewriter.create<MO::ChainCreateOp>(loadOp.getLoc(), ValueRange{chains});

    rewriter.replaceAllUsesWith(loadOp.getOutChain(), newChain);

    loadOp.erase();

    return success();
  }
};

bool kernelAlreadyHasDirectBufferInput(Operation *kernelOp,
                                       MO::MutableLoadOp loadOp) {
  // Check if kernel is sole consumer of load's chain output
  if (!loadOp.getOutChain().hasOneUse())
    return false;

  auto *user = *loadOp.getOutChain().getUsers().begin();
  if (user != kernelOp)
    return false;

  // Get the operand index of the tensor output in the kernel
  OpOperand *tensorOperand = nullptr;
  for (OpOperand &opOperand : kernelOp->getOpOperands()) {
    if (opOperand.get() == loadOp.getOutTensor()) {
      tensorOperand = &opOperand;
      break;
    }
  }

  if (!tensorOperand)
    return false;

  // Check if the kernel also has a direct input of the buffer at the
  // appropriate position
  unsigned tensorIdx = tensorOperand->getOperandNumber();
  for (unsigned i = 0; i < kernelOp->getNumOperands(); ++i) {
    if (kernelOp->getOperand(i) == loadOp.getInBuffer() && i != tensorIdx)
      return true;
  }

  return false;
}

bool kernelHasInOutChains(Operation *kernelOp) {
  return getKernelInputChain(kernelOp) && getKernelOutputChain(kernelOp);
}

struct FuseMutableLoads : public impl::FuseMutableLoadsBase<FuseMutableLoads> {
  using FuseMutableLoadsBase::FuseMutableLoadsBase;
  void runOnOperation() override {
    MO::GraphOp moGraph = getOperation();
    MLIRContext *ctx = moGraph->getContext();
    PatternRewriter rewriter{ctx};

    SmallVector<Operation *> opsToProcess;
    moGraph.walk(
        [&](MO::MutableLoadOp loadOp) { opsToProcess.push_back(loadOp); });

    RewritePatternSet patterns{ctx};
    patterns.add<ParallelizeLoadsWithMultipleUsers>(ctx);

    if (failed(
            mlir::applyOpPatternsGreedily(opsToProcess, std::move(patterns))))
      return signalPassFailure();

    LLVM_DEBUG(llvm::dbgs() << "Graph after load parallelization:\n";
               moGraph.print(llvm::dbgs()); llvm::dbgs() << '\n');

    SmallVector<MO::MutableLoadOp> loadOps;
    moGraph.walk([&](MO::MutableLoadOp loadOp) { loadOps.push_back(loadOp); });

    for (auto loadOp : loadOps) {
      Operation *kernelOp = maybeGetCompatibleKernel(loadOp);
      if (!kernelOp)
        continue;

      if (kernelAlreadyHasDirectBufferInput(kernelOp, loadOp)) {
        // TODO: Unify all these cases
        fuseLoadBuffer(rewriter, loadOp, kernelOp,
                       /*kernelHasLoadsOutputChain=*/true);
      } else if (!kernelHasInOutChains(kernelOp)) {
        // Transform logic is a bit more involved if load fusion hasn't already
        // happened as we need to have the kernel both produce and consume a
        // chain now.
        //
        // TODO(GEX-1031): Break the transforms this operation does into two
        // steps
        // -- rewriting the kernel body to accept and produce a chain and then
        // performing the re-write from mo.tensor -> mo.buffer for the
        // appropriate operand index.
        auto foreachOp = dyn_cast<MAP::IterForeachOp>(kernelOp);
        if (!foreachOp) {
          // TODO(GEX-3962): Give a chainless opaque kernel a chain in/out pair
          // too. Unlike a foreach, that also means adding an output block
          // argument for the new chain result and threading it through the
          // kernel's map.call.opaque.
          LLVM_DEBUG(loadOp.emitRemark()
                     << "Skipping mutable load fusion into an opaque kernel "
                        "that has no chain to sequence the load against.");
          continue;
        }
        fuseLoadBufferAndChains(rewriter, loadOp, foreachOp);
      } else {
        // Transform is much simpler in the case that load fusion has already
        // occurred as we just need to re-write the kernel body to use a buffer
        // type (and possibly turn off input fusion for the buffer we're fusing
        // in).
        fuseLoadBuffer(rewriter, loadOp, kernelOp,
                       /*kernelHasLoadsOutputChain=*/false);
      }
    }
  }
};

} // namespace M::MAP
