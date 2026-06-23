//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/FusionUtils.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEEPILOGUES
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// TODO(#22349): Remove checks when matmul/conv/concat kernels respect the
// fusion contract.
static bool containsMatmulOrConvOrConcat(MOGG::KernelOp kernel) {
  auto callExecuteOp =
      dyn_cast<MOGG::CallExecuteOp>(kernel.getTerminator().getOperation());
  if (!callExecuteOp)
    return false;

  auto opName = callExecuteOp.getMoOpName();

  return (opName.contains(MO::ConcatOp::getOperationName()) ||
          opName.contains(MO::MatmulOp::getOperationName()) ||
          opName.contains(MO::BatchMatmulOp::getOperationName()) ||
          opName.contains(MO::ConvOp::getOperationName()) ||
          opName.contains(MO::ConvTransposeOp::getOperationName()));
}

static bool containsDistributedOp(MOGG::KernelOp kernel) {
  auto callExecuteOp =
      dyn_cast<MOGG::CallExecuteOp>(kernel.getTerminator().getOperation());
  if (!callExecuteOp)
    return false;

  auto opName = callExecuteOp.getMoOpName();

  return (opName.contains(MO::AllGatherOp::getOperationName()) ||
          opName.contains(MO::AllReduceSumOp::getOperationName()) ||
          opName.contains(MO::Composite::DistributedMatmulReduceScatterSumOp::
                              getOperationName()) ||
          opName.contains(MO::ReduceScatterSumOp::getOperationName()) ||
          opName.contains(MO::DistributedBroadcastOp::getOperationName()) ||
          opName.contains(MO::EPDispatchOp::getOperationName()) ||
          opName.contains(MO::EPDispatchFP8Op::getOperationName()) ||
          opName.contains(MO::EPDispatchBlockScaledNVOp::getOperationName()) ||
          opName.contains(MO::EPDispatchMXFP4Op::getOperationName()) ||
          opName.contains(MO::EPCombineOp::getOperationName()));
}

static bool containsMutableStore(MOGG::KernelOp kernel) {
  for (auto bindOp : kernel.getOps<MOGG::BindLambdaOp>()) {
    auto kind = bindOp.getBindKind();
    if (kind != MOGG::BindLambdaKind::ElemwiseLambda &&
        kind != MOGG::BindLambdaKind::OutputLambda)
      continue;

    for (auto elementwiseOp :
         bindOp.getBody()->getOps<MOGG::CallElementwiseOp>()) {
      if (elementwiseOp.getMoOpName() == MO::MutableStoreOp::getOperationName())
        return true;
    }
  }

  return false;
}

// Returns true if the two kernels share any operand of type mo.buffer.
static bool haveSharedBufferOperand(MOGG::KernelOp lhs, MOGG::KernelOp rhs) {
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

struct FuseEpilogues : public impl::FuseEpiloguesBase<FuseEpilogues> {
  using FuseEpiloguesBase::FuseEpiloguesBase;

  FuseEpilogues() = default;

  template <typename... ExcludedOpTy, typename OperationIter>
  static void moveBeforeExcept(Operation *dst,
                               iterator_range<OperationIter> ops) {
    for (auto &op : llvm::make_early_inc_range(ops)) {
      if (!isa<ExcludedOpTy...>(&op))
        op.moveBefore(dst);
    }
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

  // Returns the number of epilogues fused.
  size_t fuseEpilogues(MO::GraphOp moGraph) {
    IRRewriter rewriter(moGraph->getContext());
    DenseSet<Operation *> toErase;
    moGraph.walk([&](MOGG::KernelOp dst) {
      // skip fused one
      if (toErase.find(dst) != toErase.end())
        return;

      if (dst.getElemwiseBodyLambda())
        return;

      if (dst.getNumResults() == 0)
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

        auto src = dyn_cast<MOGG::KernelOp>(consumingOp);
        if (!src)
          continue;

        if (!src.getElemwiseBodyLambda())
          continue;

        // We can not fuse if the output's view is updated in the elementwise
        // kernel.
        if (!src.isOperandFusible(use.getOperandNumber(),
                                  /*lookThroughView=*/false)) {
          continue;
        }

        // TODO(GEX-2213): Come up with more general mechanisms for when
        // fusion with kernels handling mutable values is permissible.
        // If the src contains a mutable store and the two kernels share any
        // mo.buffer operand, bail out of fusion.
        if (containsMutableStore(src) && haveSharedBufferOperand(src, dst))
          continue;

        // Ensure we don't fuse src if another operand is dependant on dst.
        // It would lead to an invalid IR with a cycle.
        // Eg:
        // ```
        // %0 = foo(%arg0)
        // %1 = bar(%0)
        // %2 = mo.add(%1, %0)
        // mo.output %2
        // ```
        // If we try to fuse foo with mo.add, we would get:
        // ```
        // %0 = foo_mo_add_fused(%arg0, %1) // Cycle %0 <-> %1
        // %1 = bar(%0)
        // mo.output %0
        // ```
        // We already check that `src` output has a single use, which avoid
        // cycles in the simplest cases. But for ops with multiple outputs /
        // chains (like allreduce), we could get a cycle because of the chain.
        // This check ensures there is no dependency between dst -> src.
        if (llvm::any_of(src.getOperands(), [&](Value otherOperand) {
              return otherOperand != dstOutput &&
                     valueDependsTransitivelyOn(otherOperand, dst);
            })) {
          continue;
        }

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
        // CPU (it would make no sense to put on a GPU if it uses many GPUs,
        // for instance). For that reason, we only compare the output's device
        // with the consuming op's device. Otherwise, just compare both op's
        // devices as usual.
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

        // Move in all src's kernel operands and arguments into dst's except
        // for the one that is being fused.
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
              MOGG::ScratchPlaceholdersOp::create(
                  rewriter, outputLambda.getLoc(), outputLambda.getType())
                  .getResult(0);
          rewriter.setInsertionPointAfter(outputLambda);

          auto concretizeOp = MOGG::TensorFusedOp::create(
              rewriter, outputLambda.getLoc(), outputLambda.getType(),
              outputLambda, scratchPlaceholder);
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

        // Replace all the elementwise's arguments with an explicit load
        // except for the fusion argument.
        rewriter.setInsertionPointToStart(elemFuseBody);
        for (auto [operand, argument] :
             llvm::zip_first(elementwiseLambda.getOperands(),
                             elemFuseBody->getArguments())) {
          if (operand == srcFusionArgument) {
            argument.replaceAllUsesWith(outVal);
          } else {
            Value loadedValue = MOGG::LoadTensorOp::create(
                rewriter, argument.getLoc(), argument.getType(), operand,
                indices);
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
        dst.setFusionInfoAttr(createFusionInfo<FusionEpilogueAttr>(
            dst->getContext(), {dst, src}, {static_cast<uint32_t>(fusionIdx)}));
        // Merge source locations so that Python stack traces from fused-in ops
        // survive into AttachPythonStackTraces.
        dst->setLoc(mlir::FusedLoc::get(dst->getContext(),
                                        {dst->getLoc(), src->getLoc()}));

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

    return toErase.size();
  }

  void runOnOperation() override {
    MO::GraphOp moGraph = getOperation();
    MojoRegistry registry(moGraph->getParentOfType<mlir::ModuleOp>());
    // Run epilogue fusion interleaved with affine view fusion until
    // convergence. This handles multi-hop cases where an affine view separates
    // a producer from a potential epilogue:
    //   matmul -> add -> affine_view -> add
    // After one epilogue pass: matmul_add -> affine_view -> add
    // After affine view pass:  matmul_add_affine -> add
    // After another epilogue:  matmul_add_affine_add
    size_t fused;
    do {
      fused = fuseEpilogues(moGraph);
      fused += fuseReshapesIntoEpiloguesStep(moGraph, registry);
    } while (fused > 0);
  }
};

} // namespace

} // namespace M::MOGG
