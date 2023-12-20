//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "llvm/ADT/StringSet.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_SIMPLIFYKERNELOPERANDS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct SimplifyKernelOperands
    : public impl::SimplifyKernelOperandsBase<SimplifyKernelOperands> {
  using SimplifyKernelOperandsBase::SimplifyKernelOperandsBase;

  void runOnOperation() override {
    MO::GraphOp graphOp = getOperation();
    OpBuilder builder{graphOp.getContext()};

    graphOp.walk([](MOGG::KernelOp kernelOp) {
      if (!kernelOp.getIsKGEN().value_or(false))
        return;

      auto kernelArguments = kernelOp.getBody().front().getArguments();

      struct CacheEntry {
        Value argument;
        KGEN::ParamDeclareRegionOp inputLambda;
      };

      DenseMap<Value, CacheEntry> operandValueToCacheEntry;
      SmallVector<uint64_t> indicesToErase;

      for (uint64_t currentIndex = 0, e = kernelOp->getNumOperands();
           currentIndex < e; ++currentIndex) {
        Value currentOperandValue = kernelOp->getOperand(currentIndex);
        Value currentArgument = kernelArguments[currentIndex];
        KGEN::ParamDeclareRegionOp currentInputLambda = kernelOp.findLambda(
            kernelOp.getInputFusionInterfaceAt(currentIndex));

        auto insertionPair = operandValueToCacheEntry.try_emplace(
            currentOperandValue,
            CacheEntry{currentArgument, currentInputLambda});

        if (insertionPair.second)
          continue;

        auto &[previousArgument, previousInputLambda] =
            insertionPair.first->getSecond();

        // Cannot de-duplicate if the inputs are not accessed similarly.
        if (bool(currentInputLambda) ^ bool(previousInputLambda))
          continue;

        indicesToErase.push_back(currentIndex);
        currentArgument.replaceAllUsesWith(previousArgument);

        auto currentInputLambdaName =
            currentInputLambda.getParamDecl().getName();
        auto previousInputLambdaName =
            previousInputLambda.getParamDecl().getName();
        auto previousInputLambdaType =
            previousInputLambda.getParamDecl().getType();

        // Replacing lambda calls. This is necessary because the replaced
        // operand is accessed via a lambda and not directly.
        mlir::AttrTypeReplacer walker;
        walker.addReplacement([&](KGEN::ParamDeclRefAttr attr)
                                  -> std::optional<KGEN::ParamDeclRefAttr> {
          if (attr.getName() == currentInputLambdaName)
            return KGEN::ParamDeclRefAttr::get(previousInputLambdaName,
                                               previousInputLambdaType);
          return std::nullopt;
        });
        walker.recursivelyReplaceElementsIn(kernelOp, /*replaceAttrs=*/true,
                                            /*replaceLocs=*/false,
                                            /*replaceTypes=*/true);
        currentInputLambda.erase();
      }

      // Iterating in reverse order to preserve the indices.
      for (uint64_t index : llvm::reverse(indicesToErase))
        kernelOp.removeOperandAtIndex(index);
    });
  }
};

} // namespace

} // namespace M::MOGG
