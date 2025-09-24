//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEPROLOGUESEPILOGUES
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// TODO(#22349): Remove checks when matmul/conv/concat kernels respect the
// fusion contract.
static bool containsMatmulOrConvOrConcat(MOGG::ExpKernelOp kernel) {
  auto callExecuteOp =
      dyn_cast<MOGG::CallExecuteOp>(kernel.getTerminator().getOperation());
  if (!callExecuteOp)
    return false;

  auto kernelName = callExecuteOp.getKernelNameAttr().strref();

  return (kernelName.contains(MO::ConcatOp::getOperationName()) ||
          kernelName.contains(MO::MatmulOp::getOperationName()) ||
          kernelName.contains(MO::BatchMatmulOp::getOperationName()) ||
          kernelName.contains(MO::ConvOp::getOperationName()) ||
          kernelName.contains(MO::ConvTransposeOp::getOperationName()));
}

static bool containsDistributedOp(MOGG::ExpKernelOp kernel) {
  auto callExecuteOp =
      dyn_cast<MOGG::CallExecuteOp>(kernel.getTerminator().getOperation());
  if (!callExecuteOp)
    return false;

  auto kernelName = callExecuteOp.getKernelNameAttr().strref();

  return (kernelName.contains(MO::AllGatherOp::getOperationName()) ||
          kernelName.contains(MO::AllReduceSumOp::getOperationName()) ||
          kernelName.contains(MO::MatmulAllReduceOp::getOperationName()));
}

static bool containsMutableStore(MOGG::ExpKernelOp kernel) {
  for (auto bindOp : kernel.getOps<MOGG::BindLambdaOp>()) {
    auto kind = bindOp.getBindKind();
    if (kind != MOGG::BindLambdaKind::ElemwiseLambda &&
        kind != MOGG::BindLambdaKind::OutputLambda)
      continue;

    for (auto elementwiseOp :
         bindOp.getBody()->getOps<MOGG::CallElementwiseOp>()) {
      if (elementwiseOp.getKernelNameAttr().strref() ==
          MO::MutableStoreOp::getOperationName())
        return true;
    }
  }

  return false;
}

// Returns true if the two kernels share any operand of type mo.buffer.
static bool haveSharedBufferOperand(MOGG::ExpKernelOp lhs,
                                    MOGG::ExpKernelOp rhs) {
  for (Value lhsOperand : lhs.getInputs()) {
    if (!isa<MO::BufferType>(lhsOperand.getType()))
      continue;
    for (Value rhsOperand : rhs.getInputs()) {
      if (!isa<MO::BufferType>(rhsOperand.getType()))
        continue;
      if (lhsOperand == rhsOperand)
        return true;
    }
  }
  return false;
}

static bool checkMutableFusible(std::optional<Value> proInChain,
                                std::optional<Value> proOutChain,
                                std::optional<Value> epiInChain,
                                std::optional<Value> epiOutChain) {

  if (proInChain.has_value() && proOutChain.has_value() &&
      epiInChain.has_value() && epiOutChain.has_value()) {

    // Ensure the input chain to epilogue kernel is output chain from producer
    // kernel
    if (*epiInChain != *proOutChain)
      return false;

    // Ensure that user kernel is the sole consumer of the output chain from
    // source kernel.
    // Strictly speaking source kernel having more than one consumer of
    // its output chain would be UB when those consumers perform writes to
    // mo.buffers -- skip fusion here to make the presence of that situation
    // clearer.
    if (!proOutChain->hasOneUse())
      return false;
  }

  return true;
}

struct FuseProloguesEpilogues
    : public impl::FuseProloguesEpiloguesBase<FuseProloguesEpilogues> {
  using FuseProloguesEpiloguesBase::FuseProloguesEpiloguesBase;

  FuseProloguesEpilogues() = default;

  template <typename... ExcludedOpTy, typename OperationIter>
  static void moveBeforeExcept(Operation *dst,
                               iterator_range<OperationIter> ops) {
    for (auto &op : llvm::make_early_inc_range(ops)) {
      if (!isa<ExcludedOpTy...>(&op))
        op.moveBefore(dst);
    }
  }

  void fusePrologues(MO::GraphOp moGraph) {
    IRRewriter rewriter(moGraph->getContext());
    moGraph.walk([&](MOGG::ExpKernelOp dst) {
      if (dst.getElemwiseBodyLambda())
        return;
      // Do signed subtraction to avoid integer underflow.
      int64_t numInputs = dst.getExecBody().getNumArguments();
      for (int64_t fusionIdx = numInputs - 1; fusionIdx >= 0; --fusionIdx) {
        auto inputLambda = dst.getBoundInputLambda(fusionIdx);
        if (!inputLambda)
          continue;

        auto providerOp = dst->getOperand(fusionIdx).getDefiningOp();
        if (!providerOp || !providerOp->hasOneUse())
          continue;

        if (providerOp->getBlock() != dst->getBlock())
          continue;

        if (providerOp->getNumResults() != 1)
          continue;

        auto src = dyn_cast<MOGG::ExpKernelOp>(providerOp);
        if (!src)
          continue;

        if (dst.getExecutionDevice() != src.getExecutionDevice())
          continue;

        if (!src.getElemwiseBodyLambda())
          continue;

        ++numberOfFusedPrologues;

        // Add the indices in the operand list of the mogg output op.
        auto elementwiseLambda = src.getElemwiseBodyLambda();
        Block *elementwiseLambdaBody = elementwiseLambda.getBody();
        auto elementwiseLambdaTerminator =
            elementwiseLambdaBody->getTerminator();
        elementwiseLambdaTerminator->insertOperands(
            1, elementwiseLambdaBody->getArguments().back());

        // Bring in all the operands (except chain) of src into dst and do SSA
        // replacement.
        // TODO: we probably need to merge chain here.
        dst.insertInputsAt(dst.getInputs().size(), src.getInputs());
        auto newArgs =
            dst.getExecBody().getArguments().take_back(src.getInputs().size());
        auto oldArgs = src.getExecBody().getArguments();
        for (auto [newArg, oldArg] : llvm::zip_equal(newArgs, oldArgs))
          oldArg.replaceAllUsesWith(newArg);

        // Finally, move in the elementwise lambda and turn it into an input
        // lambda.
        elementwiseLambda->moveAfter(inputLambda);
        elementwiseLambda.setBindKind(M::MOGG::BindLambdaKind::InputLambda);
        inputLambda->replaceAllUsesWith(elementwiseLambda);
        inputLambda->erase();

        // Remove the associated input tensor from the kernel signature.
        dst.eraseInputsAt(fusionIdx);

        // Copy views that were previously fused into the elementwise kernel, if
        // any.
        moveBeforeExcept<MOGG::OutputPlaceholdersOp, MOGG::BindLambdaOp,
                         MOGG::KernelTerminator>(elementwiseLambda,
                                                 src.getOps());

        dst.setSummaryAttr(getSummaryString({src, dst}, FusionType::HARD));
        // Delete the src because it is no longer needed.
        src->erase();
      }
    });
  }

  MO::TensorType getTensorType(Type type) {
    if (auto asTensorType = dyn_cast<MO::TensorType>(type))
      return asTensorType;

    if (auto asBufferType = dyn_cast<MO::BufferType>(type))
      return asBufferType.toTensorType();

    return nullptr;
  }

  bool dataTypesMatch(Type t1, Type t2) {
    auto tensor1 = getTensorType(t1);
    auto tensor2 = getTensorType(t2);
    if (!tensor1 || !tensor2)
      return false;

    return tensor1.getElementType() == tensor2.getElementType();
  }

  void fuseEpilogues(MO::GraphOp moGraph) {
    IRRewriter rewriter(moGraph->getContext());
    DenseSet<Operation *> toErase;
    moGraph.walk([&](MOGG::ExpKernelOp dst) {
      // skip fused one
      if (toErase.find(dst) != toErase.end())
        return;

      if (dst.getElemwiseBodyLambda())
        return;

      bool isADistributedOp = containsDistributedOp(dst);

      for (int64_t fusionIdx = dst.getNumResults() - 1; fusionIdx >= 0;
           --fusionIdx) {
        if (isa<MO::ChainType>(dst.getResult(fusionIdx).getType()))
          continue;

        auto outputLambda = dst.getBoundOutputLambda(fusionIdx);
        if (!outputLambda)
          continue;

        Value dstOutput = dst.getResult(fusionIdx);
        if (!dstOutput.hasOneUse())
          continue;

        OpOperand &use = *dstOutput.getUses().begin();
        Operation *consumingOp = use.getOwner();
        if (consumingOp->getNumResults() != 1)
          continue;

        if (consumingOp->getBlock() != dst->getBlock())
          continue;

        auto src = dyn_cast<MOGG::ExpKernelOp>(consumingOp);
        if (!src)
          continue;

        if (!src.getElemwiseBodyLambda())
          continue;

        // We can not fuse if the output's view is updated in the elementwise
        // kernel.
        if (!src.isOperandFusible(use.getOperandNumber(),
                                  /*lookThroughView=*/false))
          continue;

        // TODO(GEX-2213): Come up with more general mechanisms for when fusion
        // with kernels handling mutable values is permissible.
        // If the src contains a mutable store and the two kernels share any
        // mo.buffer operand, bail out of fusion.
        if (containsMutableStore(src) && haveSharedBufferOperand(src, dst))
          continue;

        // Check chain compatibility if both kernels have input and output
        // chains
        auto proInChain = dst.maybeGetInChain();
        auto proOutChain = dst.maybeGetOutChain();
        auto epiInChain = src.maybeGetInChain();
        auto epiOutChain = src.maybeGetOutChain();
        if (!checkMutableFusible(proInChain, proOutChain, epiInChain,
                                 epiOutChain))
          continue;

        // If this is a distributed op, the op itself will be executed on the
        // CPU (it would make no sense to put on a GPU if it uses many GPUs, for
        // instance). For that reason, we only compare the output's device with
        // the consuming op's device. Otherwise, just compare both op's devices
        // as usual.
        if (!isADistributedOp &&
            dst.getExecutionDevice() != src.getExecutionDevice()) {
          continue;
        }

        if (isADistributedOp &&
            cast<MO::TensorType>(dstOutput.getType()).getDeviceRef() !=
                src.getExecutionDevice()) {
          continue;
        }

        ++numberOfFusedEpilogues;

        // Move in all src's kernel operands and arguments into dst's except for
        // the one that is being fused.
        //
        // TODO: we probably need to merge chain here.
        Value srcFusionArgument;
        for (const auto [argument, operand] : llvm::reverse(llvm::zip_first(
                 src.getExecBody().getArguments(), src.getInputs()))) {
          if (operand != dstOutput) {
            dst.insertInputsAt(0, operand);
            Value newArgument = dst.getExecBody().getArgument(0);
            argument.replaceAllUsesWith(newArgument);
          } else {
            srcFusionArgument = argument;
          }
        }

        auto fusedResultTp = src->getResult(0).getType();
        // TODO: we also need to rotate chain to the end.
        dstOutput.setType(fusedResultTp);
        MOGG::OutputPlaceholdersOp outHolderOp = dst.getOutputPlaceholder();
        if (containsMatmulOrConvOrConcat(dst) &&
            !dataTypesMatch(outputLambda.getOutput().getType(),
                            src.getElemwiseBodyLambda().getType())) {
          rewriter.setInsertionPointAfter(outHolderOp);
          auto scratchPlaceholder =
              rewriter
                  .create<MOGG::ScratchPlaceholdersOp>(outputLambda.getLoc(),
                                                       outputLambda.getType())
                  .getResult(0);
          rewriter.setInsertionPointAfter(outputLambda);
          auto concretizeOp = rewriter.create<MOGG::ConcretizeTensorOp>(
              outputLambda.getLoc(), scratchPlaceholder, outputLambda);
          outputLambda->replaceUsesWithIf(
              concretizeOp, [&](OpOperand &operand) {
                return operand.getOwner() != concretizeOp;
              });
        }
        if (isa<MO::ChainType>(fusedResultTp)) {
          auto foreachCall = cast<MOGG::CallForeachOp>(src.getTerminator());
          auto bufferOut = foreachCall.getOutput();
          outHolderOp->getResult(fusionIdx).replaceAllUsesWith(bufferOut);
        } else {
          outHolderOp->getResult(fusionIdx).setType(fusedResultTp);
        }

        auto outputLambdaBody = outputLambda.getBody(0);
        auto outputLambdaTerminator = outputLambdaBody->getTerminator();
        Value outVal = outputLambdaTerminator->getOperand(0);
        Value indices = outputLambdaTerminator->getOperand(1);

        BindLambdaOp elementwiseLambda = src.getElemwiseBodyLambda();
        Block *elemFuseBody = elementwiseLambda.getBody();
        elemFuseBody->getArguments().back().replaceAllUsesWith(indices);
        elemFuseBody->getTerminator()->insertOperands(1, indices);

        // Replace all the elementwise's arguments with an explicit load except
        // for the fusion argument.
        rewriter.setInsertionPointToStart(elemFuseBody);
        for (auto [operand, argument] :
             llvm::zip_first(elementwiseLambda.getOperands(),
                             elemFuseBody->getArguments())) {
          if (operand == srcFusionArgument) {
            argument.replaceAllUsesWith(outVal);
          } else {
            Value loadedValue = rewriter.create<MOGG::LoadTensorOp>(
                argument.getLoc(), argument.getType(), operand, indices);
            argument.replaceAllUsesWith(loadedValue);
          }
        }

        // Move in the elementwise lambda's content into the output lambda.
        rewriter.setInsertionPoint(outputLambdaTerminator);
        elemFuseBody->eraseArguments(0, elemFuseBody->getNumArguments());
        rewriter.inlineBlockBefore(elemFuseBody, outputLambdaBody,
                                   rewriter.getInsertionPoint());
        outputLambdaTerminator->erase();

        moveBeforeExcept<MOGG::OutputPlaceholdersOp, MOGG::BindLambdaOp,
                         MOGG::KernelTerminator>(outputLambda, src.getOps());

        src->replaceAllUsesWith(ValueRange{dstOutput});
        dst.setSummaryAttr(getSummaryString({dst, src}, FusionType::HARD));

        dst.setHasReadEffect(dst.getHasReadEffect() || src.getHasReadEffect());
        dst.setHasWriteEffect(dst.getHasWriteEffect() ||
                              src.getHasWriteEffect());
        // TODO: this should be a merge of chain instead to be more general.
        dst.setChain(src.getChain());

        toErase.insert(src);
      }
    });

    for (auto op : toErase)
      op->erase();
  }

  void runOnOperation() override {
    MO::GraphOp moGraph = getOperation();
    fuseEpilogues(moGraph);
    fusePrologues(moGraph);
  }
};

} // namespace

} // namespace M::MOGG
