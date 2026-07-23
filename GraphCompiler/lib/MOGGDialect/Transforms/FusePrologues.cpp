//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEPROLOGUES
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct FusePrologues : public impl::FuseProloguesBase<FusePrologues> {
  using FuseProloguesBase::FuseProloguesBase;

  FusePrologues() = default;

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
    moGraph.walk([&](MOGG::KernelOp dst) {
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

        auto src = dyn_cast<MOGG::KernelOp>(providerOp);
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

        // Copy views that were previously fused into the elementwise kernel,
        // if any.
        moveBeforeExcept<MOGG::OutputPlaceholdersOp, MOGG::BindLambdaOp,
                         MOGG::KernelTerminator>(elementwiseLambda,
                                                 src.getOps());

        dst.setFusionInfoAttr(createFusionInfo<FusionPrologueAttr>(
            dst->getContext(), {src, dst}, {static_cast<uint32_t>(fusionIdx)}));
        // Merge source locations so that Python stack traces from fused-in ops
        // survive into AddStackTracesToAssertsPass.
        dst->setLoc(mlir::FusedLoc::get(dst->getContext(),
                                        {src->getLoc(), dst->getLoc()}));
        // Delete the src because it is no longer needed.
        src->erase();
      }
    });
  }

  void runOnOperation() override {
    MO::GraphOp moGraph = getOperation();
    fusePrologues(moGraph);
  }
};

} // namespace

} // namespace M::MOGG
