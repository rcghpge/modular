//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include <functional>
#include <optional>

using namespace M;

#define DEBUG_TYPE "fuse-mutable-loads"

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEMUTABLELOADS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

/// Determines if there's a valid KernelOp to perform load fusion with.
///
/// The sole condition is that the kernel must be the sole consumer of the
/// tensor output of the load op.
///
/// Also the kernel op must not have any other buffer inputs
std::optional<MOGG::KernelOp>
maybeGetCompatibleKernel(MO::MutableLoadOp loadOp) {

  auto tensor = loadOp.getOutTensor();

  if (!tensor.hasOneUse()) {
    LLVM_DEBUG(loadOp.emitRemark()
               << "Mutable load fusion is disabled for this load op as its "
                  "output tensor has more than one use.");

    return std::nullopt;
  }

  if (loadOp->getBlock() != (*tensor.getUsers().begin())->getBlock()) {
    LLVM_DEBUG(
        loadOp.emitRemark()
        << "Mutable load fusion is disabled for this load op as its "
           "output tensor is produced in a different block than its user.");
    return std::nullopt;
  }

  MOGG::KernelOp kernelOp =
      dyn_cast<MOGG::KernelOp>(*tensor.getUsers().begin());

  if (!kernelOp)
    return std::nullopt;

  // At present the only kernels marked as elementwise are the standard ones
  // you would expect (pure compute ops like mul,add,sqrt, etc.) and
  // mo.mutable.store.
  //
  // If we have a kernel that has a buffer operand that is the same as the load
  // op's input then in general it's unsafe to perform load fusion so block it
  // here.
  if (llvm::any_of(kernelOp.getOperands(), [&](Value operand) {
        return operand == loadOp.getInBuffer();
      })) {
    // If the kernel is elementwise we can still load fuse even if the buffer is
    // shared.
    if (!isa<MOGG::CallForeachOp>(kernelOp.getTerminator())) {
      LLVM_DEBUG(loadOp.emitRemark()
                 << "Attempting load fusion with an in-place kernel that is "
                    "not elementwise.");
      return std::nullopt;
    }
  }

  return kernelOp;
}

KernelOp fuseLoadBufferAndChains(OpBuilder &builder, MO::MutableLoadOp loadOp,
                                 KernelOp kernelOp) {
  MLIRContext *ctx = builder.getContext();

  auto loc = kernelOp.getLoc();

  auto chainType = MO::ChainType::get(ctx);

  auto &tensorUse = *loadOp.getOutTensor().getUses().begin();
  auto replaceIdx = tensorUse.getOperandNumber();

  // create the new operand list
  SmallVector<Value> newOperandList(kernelOp.getOperands());
  newOperandList[replaceIdx] = loadOp.getInBuffer();
  // Add the extra chain operand.
  newOperandList.push_back(loadOp.getInChain());

  // create new result list
  SmallVector<Type> newResults(TypeRange(kernelOp.getResults()));
  newResults.push_back(chainType);

  builder.setInsertionPointAfter(kernelOp);

  auto newKernelOp = MOGG::KernelOp::create(
      builder, kernelOp->getLoc(),
      /*results=*/newResults,
      /*operands=*/newOperandList,
      /*fusionInfo=*/
      MOGG::createFusionInfo<MOGG::FusionMutableLoadAttr>(
          kernelOp->getContext(), {loadOp, kernelOp}),
      /*isAsync=*/kernelOp.getIsAsyncAttr(),
      /*device*/ kernelOp.getExecutionDevice(),
      /*outputParamDecls=*/kernelOp.getOutputParamDeclsAttr(),
      /*allocIndices=*/kernelOp.getAllocIndicesAttr(),
      /*hasReadEffect=*/true,
      /*hasWriteEffect=*/kernelOp.getHasWriteEffect());

  // Copy over any discardable attributes from the original kernel
  for (auto attr : kernelOp->getDiscardableAttrs())
    newKernelOp->setDiscardableAttr(attr.getName(), attr.getValue());

  Block &oldBody = *kernelOp.getBody();
  Block &newBody = *newKernelOp.getBody();

  builder.setInsertionPointToStart(&newBody);

  IRMapping remapper;

  auto outputPlaceholder = kernelOp.getOutputPlaceholder();
  ASSERT_STREAM(outputPlaceholder != nullptr,
                "Expected kernel op to have an output placeholder op.");
  SmallVector<Type> newPlaceholderTypes;
  for (Value val : outputPlaceholder.getResults()) {
    newPlaceholderTypes.push_back(val.getType());
  }

  auto newOutputPlaceHolderOp =
      OutputPlaceholdersOp::create(builder, loc, newPlaceholderTypes);

  for (auto &&[oldRes, newres] : llvm::zip(outputPlaceholder.getResults(),
                                           newOutputPlaceHolderOp.getResults()))
    remapper.map(oldRes, newres);

  for (auto &&[oldArg, newArg] :
       llvm::zip(oldBody.getArguments(), newBody.getArguments())) {
    remapper.map(oldArg, newArg);
  }

  for (Operation &op : kernelOp.getOps()) {
    if (!isa<MOGG::CreateReadyChainOp, OutputPlaceholdersOp>(op))
      builder.insert(op.clone(remapper));
  }

  auto outChainIdx = newKernelOp.getNumResults() - 1;
  auto outChain = newKernelOp->getResult(outChainIdx);

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
//   %merged = mo.chain.create(%other, %inCh)   // %outCh → %inCh
//   %result = mogg.kernel(buf, ..., %merged)  // fused!
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
  ASSERT_STREAM(!valueDependsTransitivelyOn(kernelInChain, loadOutChain),
                "Expected exclusive chain path from mutable.load to kernel op "
                "to be severed.");

  return true;
}

void fuseLoadBuffer(OpBuilder &builder, MO::MutableLoadOp loadOp,
                    KernelOp kernelOp, bool kernelHasLoadsOutputChain) {
  auto buffer = loadOp.getInBuffer();
  auto tensor = loadOp.getOutTensor();

  ASSERT_STREAM(*tensor.getUsers().begin() == kernelOp,
                "Expected value semantic tensor from load op to only be "
                "consumed by current kernel op");

  OpOperand &operand = *tensor.getUses().begin();
  auto argIdx = operand.getOperandNumber();

  Block &kernelBody = *kernelOp.getBody();

  TypedValue<MO::ChainType> loadOutChain = loadOp.getOutChain();
  if (auto maybeKernelInChain = kernelOp.maybeGetInChain()) {
    if (*maybeKernelInChain != loadOutChain &&
        valueDependsTransitivelyOn(*maybeKernelInChain, loadOutChain) &&
        !detachExclusiveChainPath(loadOutChain, loadOp.getInChain(),
                                  *maybeKernelInChain)) {
      LLVM_DEBUG(loadOp.emitRemark() << "Skipping mutable load fusion to avoid "
                                        "introducing a chain cycle.");
      return;
    }
  }

  kernelOp.insertInputsAt(argIdx, buffer);
  Value newBufferArg = kernelBody.getArgument(argIdx);

  auto oldBlockArg = kernelBody.getArgument(argIdx + 1);
  oldBlockArg.replaceAllUsesWith(newBufferArg);
  kernelOp.eraseInputsAt(argIdx + 1);

  if (!kernelHasLoadsOutputChain) {
    loadOutChain.replaceAllUsesWith(kernelOp.maybeGetOutChain().value());

    builder.setInsertionPoint(kernelOp);

    auto newChain = builder.createOrFold<MO::ChainCreateOp>(
        loadOp.getLoc(),
        ValueRange{kernelOp.maybeGetInChain().value(), loadOp.getInChain()});
    kernelOp.setChain(newChain);
  } else {
    // Simply rewrite the loads input chain when the kernel has the loads output
    // chain as input.
    kernelOp.setChain(loadOp.getInChain());
  }

  kernelOp.setFusionInfoAttr(
      MOGG::createFusionInfo<MOGG::FusionMutableLoadAttr>(
          kernelOp->getContext(), {loadOp, kernelOp}));

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

    // Check if any of the consumer kernels have a write effect and share the
    // same buffer
    for (OpOperand &use : tensor.getUses()) {
      if (auto kernelOp = dyn_cast<MOGG::KernelOp>(use.getOwner())) {
        if (kernelOp.getHasWriteEffect()) {
          // Check if this kernel uses the same buffer as the load op
          for (auto operand : kernelOp.getOperands()) {
            if (operand == loadBuffer) {
              return rewriter.notifyMatchFailure(
                  loadOp.getLoc(),
                  "Parallelization blocked: a consumer kernel has write effect "
                  "and "
                  "shares the same buffer as the load operation.");
            }
          }
        }
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

bool kernelAlreadyHasDirectBufferInput(KernelOp kernelOp,
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
  for (unsigned i = 0; i < kernelOp.getNumOperands(); ++i) {
    if (kernelOp.getOperand(i) == loadOp.getInBuffer() && i != tensorIdx)
      return true;
  }

  return false;
}

bool hasBufferOperand(KernelOp kernelOp) {
  return llvm::any_of(kernelOp.getOperandTypes(),
                      [](Type type) { return isa<MO::BufferType>(type); });
}

bool kernelHasInOutChains(KernelOp kernelOp) {
  return kernelOp.maybeGetInChain().has_value() &&
         kernelOp.maybeGetOutChain().has_value();
}

struct FuseMutableLoads : public impl::FuseMutableLoadsBase<FuseMutableLoads> {
  using FuseMutableLoadsBase::FuseMutableLoadsBase;
  void runOnOperation() override {
    MO::GraphOp moGraph = getOperation();
    MLIRContext *ctx = moGraph->getContext();
    OpBuilder builder{ctx};

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
      auto maybeKernelOp = maybeGetCompatibleKernel(loadOp);
      if (!maybeKernelOp)
        continue;
      auto kernelOp = maybeKernelOp.value();

      if (kernelAlreadyHasDirectBufferInput(kernelOp, loadOp)) {
        // TODO: Unify all these cases
        fuseLoadBuffer(builder, loadOp, kernelOp,
                       /*kernelHasLoadsOutputChain=*/true);
      } else if (!kernelHasInOutChains(kernelOp)) {
        // Skip load fusion for kernels without output placeholders (view
        // operations like slice). These need tensor semantics and can't work
        // with buffer types directly.
        if (!kernelOp.getOutputPlaceholder()) {
          LLVM_DEBUG(loadOp.emitRemark()
                     << "Skipping mutable load fusion for kernel without "
                        "output placeholder (view operation).");
          continue;
        }
        // Transform logic is a bit more involved if load fusion hasn't already
        // happened as we need to have the kernel both produce and consume a
        // chain now.
        //
        // TODO(GEX-1031): Break the transforms this operation does into two
        // steps
        // -- rewriting the kernel body to accept and produce a chain and then
        // performing the re-write from mo.tensor -> mo.buffer for the
        // appropriate operand index.
        fuseLoadBufferAndChains(builder, loadOp, kernelOp);
      } else {
        // Transform is much simpler in the case that load fusion has already
        // occurred as we just need to re-write the kernel body to use a buffer
        // type (and possibly turn off input fusion for the buffer we're fusing
        // in).
        fuseLoadBuffer(builder, loadOp, kernelOp,
                       /*kernelHasLoadsOutputChain=*/false);
      }
    }
  }
};

} // namespace M::MOGG
