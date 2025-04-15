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
#include "llvm/Support/Debug.h"
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

  MOGG::KernelOp kernelOp = dyn_cast<KernelOp>(*tensor.getUsers().begin());

  if (!kernelOp)
    return std::nullopt;

  return kernelOp;
}

KernelOp fuseLoadBufferAndChains(OpBuilder &builder, MO::MutableLoadOp loadOp,
                                 KernelOp kernelOp) {
  MLIRContext *ctx = builder.getContext();

  auto loc = kernelOp.getLoc();

  auto chainType = MO::ChainType::get(ctx);
  auto emptyStrAttr = builder.getStringAttr("");

  auto &tensorUse = *loadOp.getOutTensor().getUses().begin();
  auto replaceIdx = tensorUse.getOperandNumber();

  // create the new operand list
  SmallVector<Value> newOperandList(kernelOp.getOperands());
  newOperandList[replaceIdx] = loadOp.getInBuffer();
  newOperandList.push_back(loadOp.getInChain());

  // create new result list
  SmallVector<Type> newResults(TypeRange(kernelOp.getResults()));
  newResults.push_back(chainType);

  // add empty entry for the chain output if the output fusion interface is
  // present
  ArrayAttr newOutputFusionInterface;
  auto oldOutputFusionInterface = kernelOp.getOutputFusionInterface();

  newOutputFusionInterface = nullptr;
  if (oldOutputFusionInterface) {
    SmallVector<Attribute> rawFusionInterface(
        oldOutputFusionInterface->getValue());
    rawFusionInterface.push_back(emptyStrAttr);

    newOutputFusionInterface = builder.getArrayAttr(rawFusionInterface);
  }

  // disable input fusion for the buffer input
  auto oldFusionInterface = kernelOp.getInputFusionInterface();

  ArrayAttr newInputFusionInterface = nullptr;
  if (oldFusionInterface) {
    SmallVector<Attribute> rawFusionInterface(oldFusionInterface->getValue());
    rawFusionInterface[replaceIdx] = emptyStrAttr;
    rawFusionInterface.push_back(emptyStrAttr);

    newInputFusionInterface = builder.getArrayAttr(rawFusionInterface);
  }

  builder.setInsertionPointAfter(kernelOp);

  auto newKernelOp = builder.create<MOGG::KernelOp>(
      kernelOp.getLoc(),
      /*results=*/newResults,
      /*operands=*/newOperandList,
      /*outputParamDecls=*/kernelOp.getOutputParamDeclsAttr(),
      /*inputFusionInterface=*/
      newInputFusionInterface,
      /*outputFusionInterface=*/
      newOutputFusionInterface,
      /*fusionBodyLambda=*/kernelOp.getFusionBodyLambdaAttr(),
      /*isViewOp=*/kernelOp.getIsViewOpAttr(),
      /*summary=*/kernelOp.getSummaryAttr(),
      /*isAsync=*/kernelOp.getIsAsyncAttr(),
      /*computationallyTrivial=*/kernelOp.getComputationallyTrivialAttr(),
      /*allocIndices=*/kernelOp.getAllocIndicesAttr(),
      /*hasReadEffect*/ true, // from load
      /*hasWriteEffects*/ kernelOp.getHasWriteEffect(),
      /*device*/ kernelOp.getDeviceRef());

  // respect the old deviceRef if it was present
  if (auto deviceRef = kernelOp.getDeviceRef())
    newKernelOp.setDeviceRef(deviceRef);

  // Copy over any discardable attributes from the original kernel
  for (auto attr : kernelOp->getDiscardableAttrs())
    newKernelOp->setDiscardableAttr(attr.getName(), attr.getValue());

  Block &oldBody = kernelOp.getBody().front();

  mlir::Block &newBody = newKernelOp.getBody().emplaceBlock();
  newKernelOp.setInsertionPointToTop(builder);

  IRMapping remapper;

  auto outputPlaceholder = kernelOp.getOutputPlaceholder();
  ASSERT_STREAM(outputPlaceholder != nullptr,
                "Expected kernel op to have an output placeholder op.");
  SmallVector<Type> newPlaceholderTypes;
  for (Value val : outputPlaceholder.getResults()) {
    newPlaceholderTypes.push_back(val.getType());
  }
  // Create ready chain at top of kernel body
  auto chainPlaceholder =
      builder.create<MOGG::CreateReadyChainOp>(loc, chainType);

  auto newOutputPlaceHolderOp =
      builder.create<OutputPlaceholdersOp>(loc, newPlaceholderTypes);

  for (auto &&[oldRes, newres] : llvm::zip(outputPlaceholder.getResults(),
                                           newOutputPlaceHolderOp.getResults()))
    remapper.map(oldRes, newres);

  for (Value val : newOperandList)
    newBody.addArgument(val.getType(), val.getLoc());

  for (auto &&[oldArg, newArg] :
       llvm::zip(oldBody.getArguments(), newBody.getArguments()))
    remapper.map(oldArg, newArg);

  newKernelOp.setInsertionPointToTop(builder);
  for (Operation &op : kernelOp.getOps()) {
    if (!isa<MOGG::CreateReadyChainOp, OutputPlaceholdersOp>(op))
      builder.insert(op.clone(remapper));
  }

  auto oldOut = newKernelOp.getOutputOp();
  builder.setInsertionPointAfter(oldOut);

  SmallVector<Value> newOutputs = oldOut.getOperands();
  newOutputs.push_back(chainPlaceholder);

  builder.create<MOGG::OutputOp>(loc, newOutputs);

  oldOut->erase();

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

void fuseLoadBuffer(OpBuilder &builder, MO::MutableLoadOp loadOp,
                    KernelOp kernelOp) {
  auto buffer = loadOp.getInBuffer();
  auto tensor = loadOp.getOutTensor();

  ASSERT_STREAM(*tensor.getUsers().begin() == kernelOp,
                "Expected value semantic tensor from load op to only be "
                "consumed by current kernel op");

  OpOperand &operand = *tensor.getUses().begin();
  auto argIdx = operand.getOperandNumber();

  auto newBufferArg = kernelOp.addOperandAtIndex(buffer, argIdx);

  Block &kernelBody = kernelOp.getBody().front();

  auto oldBlockArg = kernelBody.getArgument(argIdx + 1);
  oldBlockArg.replaceAllUsesWith(newBufferArg);
  kernelOp.removeOperandAtIndex(argIdx + 1);

  loadOp.getOutChain().replaceAllUsesWith(kernelOp.maybeGetOutChain().value());

  builder.setInsertionPoint(kernelOp);

  auto newChain = builder.createOrFold<MO::ChainCreateOp>(
      loadOp.getLoc(),
      ValueRange{kernelOp.maybeGetInChain().value(), loadOp.getInChain()});

  kernelOp.setOperand(kernelOp.getNumOperands() - 1, newChain);

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

    if (tensor.use_empty() || tensor.hasOneUse()) {
      return rewriter.notifyMatchFailure(
          loadOp.getLoc(),
          "Output value semantic tensor has <= 1 uses, skipping "
          "parallelization pattern.");
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
          loadOp.getLoc(), chainType, outTensor.getType(), loadOp.getInChain(),
          loadOp.getInBuffer());

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

bool hasBufferOperand(KernelOp kernelOp) {
  return llvm::any_of(kernelOp.getOperandTypes(),
                      [](Type type) { return isa<MO::BufferType>(type); });
}

bool kernelWasLoadFused(KernelOp kernelOp) {
  return kernelOp.getHasReadEffect() && hasBufferOperand(kernelOp);
}

struct FuseMutableLoads : public impl::FuseMutableLoadsBase<FuseMutableLoads> {
  using FuseMutableLoadsBase::FuseMutableLoadsBase;
  void runOnOperation() override {
    MO::GraphOp moGraph = getOperation();
    MLIRContext *ctx = moGraph->getContext();
    OpBuilder builder{ctx};

    RewritePatternSet patterns{ctx};
    patterns.add<ParallelizeLoadsWithMultipleUsers>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
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

      // Transform logic is a bit more involved if load fusion hasn't already
      // happened as we need to have the kernel both produce and consume a chain
      // now.
      //
      // TODO(GEX-1031): Break the transforms this operation does into two steps
      // -- rewriting the kernel body to accept and produce a chain and then
      // performing the re-write from mo.tensor -> mo.buffer for the appropriate
      // opereand index.
      if (!kernelWasLoadFused(kernelOp)) {
        fuseLoadBufferAndChains(builder, loadOp, kernelOp);
      } else {
        // Transform is much simpler in the case that load fusion has already
        // occured as we just need to re-write the kernel body to use a buffer
        // type (and possibly turn off input fusion for the buffer we're fusing
        // in).
        fuseLoadBuffer(builder, loadOp, kernelOp);
      }
    }
  }
};

} // namespace M::MOGG
