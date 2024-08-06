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
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/IRMapping.h>
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

  MOGG::KernelOp kernelOp = dyn_cast<KernelOp>(*tensor.getUsers().begin());

  if (!kernelOp)
    return std::nullopt;

  return kernelOp;
}

KernelOp fuseMutableLoadWithKernel(OpBuilder &builder, MO::MutableLoadOp loadOp,
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
      /*isKGEN=*/kernelOp.getIsKGENAttr(),
      /*summary=*/kernelOp.getSummaryAttr(),
      /*isAsync=*/kernelOp.getIsAsyncAttr(),
      /*allocIndices=*/kernelOp.getAllocIndicesAttr());

  // respect the old deviceRef if it was present
  if (auto deviceRef = kernelOp.getDeviceRef())
    newKernelOp.setDeviceRef(deviceRef);

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
  newPlaceholderTypes.push_back(chainType);
  auto newOutputPlaceHolderOp =
      builder.create<OutputPlaceholdersOp>(loc, newPlaceholderTypes);
  auto pchain = newOutputPlaceHolderOp.getResult(
      newOutputPlaceHolderOp.getNumResults() - 1);

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
    if (!isa<OutputPlaceholdersOp>(op))
      builder.insert(op.clone(remapper));
  }

  auto oldOut = newKernelOp.getOutputOp();
  builder.setInsertionPointAfter(oldOut);

  SmallVector<Value> newOutputs = oldOut.getOperands();
  newOutputs.push_back(pchain);

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

struct FuseMutableLoads : public impl::FuseMutableLoadsBase<FuseMutableLoads> {
  using FuseMutableLoadsBase::FuseMutableLoadsBase;
  void runOnOperation() override {
    MO::GraphOp moGraph = getOperation();
    MLIRContext *ctx = moGraph->getContext();
    OpBuilder builder{ctx};

    SmallVector<MO::MutableLoadOp> loadOps;
    moGraph.walk([&](MO::MutableLoadOp loadOp) { loadOps.push_back(loadOp); });

    for (auto loadOp : loadOps) {
      auto maybeKernelOp = maybeGetCompatibleKernel(loadOp);
      if (!maybeKernelOp)
        continue;

      fuseMutableLoadWithKernel(builder, loadOp, *maybeKernelOp);
    }
  }
};

} // namespace M::MOGG
