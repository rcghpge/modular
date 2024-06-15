//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "ArraySupport/Tensor.h"
#include "GenericML/GraphCompiler/Analysis/MODialect/ParametricTopologyAnalysis.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/SequenceFusionAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/AssertStream.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/IRMapping.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEELEMENTWISE
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

static std::optional<SmallVector<int>> getFusionIndices(KernelOp src,
                                                        KernelOp dst) {
  if (src.getOutputFusionInterfaceAt(0).empty())
    return std::nullopt;

  SmallVector<int> indices;
  for (auto [i, operand] : llvm::enumerate(dst.getOperands())) {
    if (operand.getDefiningOp() != src)
      continue;

    if (dst.getInputFusionInterfaceAt(i).empty())
      return std::nullopt;

    indices.push_back(i);
  }

  return indices;
}

static inline bool isAnElementwiseKernel(Operation *op) {
  auto kernelOp = dyn_cast<KernelOp>(op);
  if (!kernelOp)
    return false;

  if (kernelOp.getIsViewOp().value_or(false))
    return false;

  if (!kernelOp.getFusionBodyLambda())
    return false;

  if (!kernelOp.getOutputParamDecls().empty())
    return false;

  // We disable fusion for in-place allocated kernel.
  if (kernelOp.isInPlaceAlloc())
    return false;

  return true;
}

static inline bool kernelsAreCompatible(Operation *src, Operation *dst) {
  auto srcKernel = cast<KernelOp>(src);
  auto dstKernel = cast<KernelOp>(dst);

  if (srcKernel.getIsKGEN() != dstKernel.getIsKGEN())
    return false;

  if (!getFusionIndices(srcKernel, dstKernel).has_value())
    return false;

  return true;
}

struct FuseElementwisePass
    : public impl::FuseElementwiseBase<FuseElementwisePass> {
  using FuseElementwiseBase::FuseElementwiseBase;

  void runOnOperation() override {
    MO::GraphOp graphOp = getOperation();
    MO::ParametricTopologyAnalysis topologyAnalysis(graphOp);
    auto subgraphs = sourceDestinationSequenceFusionAnalysis(
        graphOp.getBody(), topologyAnalysis, isAnElementwiseKernel,
        kernelsAreCompatible, false);

    for (auto &subgraph : subgraphs)
      topologyAnalysis.computeTopologicalSorting(subgraph);

    for (auto &subgraph : subgraphs) {
      auto dstKernelOp = cast<KernelOp>(subgraph.back());
      bool isKGEN = dstKernelOp.getIsKGEN().value_or(false);
      for (auto op : llvm::reverse(subgraph)) {
        if (op == dstKernelOp)
          continue;

        auto srcKernelOp = cast<KernelOp>(op);
        if (isKGEN)
          fuseSrcKernelIntoDstKernelKGEN(srcKernelOp, dstKernelOp);
        else
          fuseSrcKernelIntoDstKernelMOGG(srcKernelOp, dstKernelOp);

        dstKernelOp.setSummaryAttr(
            getSummaryString({srcKernelOp, dstKernelOp}));
        srcKernelOp.erase();
      }

      if (isKGEN) {
        dstKernelOp.walk([](Block *block) { mlir::sortTopologically(block); });
      }
    }
  }

  // The fusion itself is applied by the following steps.
  // 1. Add all operands of the src kernel to the dst kernel, and add new args
  //    to the body.
  // 2. For each input lambda on the src kernel clone it into the dst kernel
  //    and remap to refer to the added arguments.
  // 3. Take the elementwise lambda from the src kernel and replace the lambda
  //    invoke of the argument we are replacing (remember we are fusing the
  //    src body into a use of the src kernel). Remap each input invocation
  //    to point to the cloned input lambdas.
  // 4. Remove the old argument and associated lambda.
  // 5. Update the attributes to reflect the new inputs.
  void fuseSrcKernelIntoDstKernelMOGG(MOGG::KernelOp srcKernel,
                                      KernelOp dstKernel) {
    auto fusionIndices = *getFusionIndices(srcKernel, dstKernel);
    // This assumes that srcKernel has a single output. When that changes, we
    // will have to create a mapping.
    Value previouslyFusedValue;
    // Iterating in reverse to get the right index (each iteration will erase an
    // operand and a fusion interface). The first iteration, we actually fuse.
    // The subsequent iterations will use previouslyFusedValue.
    for (int fusionIndex : llvm::reverse(fusionIndices)) {
      // The input lambda we are going to replace with the fused body.
      ArrayAttr dstHooks = dstKernel.getInputFusionInterfaceAttr();
      StringAttr dstKernelInputLambdaName =
          cast<StringAttr>(dstHooks[fusionIndex]);

      // Get the output placeholder operation in the block we are inserting
      // into. We want to this to be at the top as that simplifies other
      // transforms so we use this as the start of the block.
      MOGG::OutputPlaceholdersOp dstOuts = dstKernel.getOutputPlaceholder();

      // Symbol tables for both the incoming kernel and the kernel we are fusing
      // into.
      mlir::SymbolTable dstSymTable{dstKernel};
      mlir::SymbolTable srcSymTable{srcKernel};

      // Add the new lambdas to the top of the dst kernel after the output
      // placeholders.
      mlir::OpBuilder inserter{dstOuts->getContext()};
      dstKernel.setInsertionPointToTop(inserter);

      // Get the main elementwise block in the kernel we are fusing into.
      StringRef dstMainBody = *dstKernel.getFusionBodyLambda();
      auto dstElemwiseLambda = cast<LambdaOp>(dstSymTable.lookup(dstMainBody));

      // If we've already fused then early exit by just reusing the value we
      // have already replaced. This occurs when the srcKernel feeds into many
      // arguments of the dstKernel.
      if (previouslyFusedValue) {
        replaceLoadWithExistingUse(dstKernel, dstKernelInputLambdaName,
                                   dstElemwiseLambda, inserter,
                                   previouslyFusedValue);
        dstKernel.removeOperandAtIndex(fusionIndex);
        continue;
      }

      llvm::StringMap<StringRef> oldNameToNew;

      // Clone each of the input lambdas for the inputs we don't already have.
      ArrayAttr srcInputLambdas = srcKernel.getInputFusionInterfaceAttr();
      ArrayAttr dstInputLambdas = dstKernel.getInputFusionInterfaceAttr();

      llvm::SmallDenseMap<Value, size_t> dstOperands;
      for (size_t i = 0, e = dstKernel->getNumOperands(); i < e; ++i) {
        Value operand = dstKernel->getOperand(i);
        dstOperands[operand] = i;
      }

      IRMapping remapper;
      Block &oldBlock = srcKernel.getBody().front();
      Block &dstBlock = dstKernel.getBody().front();
      // Add the operands to the graph level op and remap the values from the
      // incoming kernel.
      for (size_t i = 0, e = srcKernel->getNumOperands(); i < e; ++i) {
        // The values on the kernel being fused in.
        Value srcOperand = srcKernel->getOperand(i);
        Value srcBlockArg = oldBlock.getArgument(i);

        // Get the lambda which loads this operand in the src kernel. If the
        // kernel doesn't expose any input attributes or only exposes null
        // attributes then we should add the null attribute.
        LambdaOp srcLambda = nullptr;
        if (srcInputLambdas) {
          StringRef name = cast<StringAttr>(srcInputLambdas[i]).getValue();
          if (!name.empty())
            srcLambda = cast<LambdaOp>(srcSymTable.lookup(name));
        }

        auto itr = dstOperands.find(srcOperand);
        if (itr != dstOperands.end()) {
          // The input of the srcKernel is already an input of the dstKernel,
          // re-use the already existing lambda or completely nullify it.
          size_t dstInIndex = itr->second;
          remapper.map(srcBlockArg, dstBlock.getArgument(dstInIndex));

          StringRef dstLambdaName =
              cast<StringAttr>(dstInputLambdas[dstInIndex]).getValue();

          if (!srcLambda || dstLambdaName.empty())
            dstKernel.setInputFusionInterface(dstInIndex, "");
          else
            oldNameToNew[srcLambda.getSymName()] = dstLambdaName;
        } else {
          // Otherwise, just add the new operand and the lambda that comes with
          // it.
          Value dstBlockArg = dstKernel.addOperand(srcOperand);
          remapper.map(srcBlockArg, dstBlockArg);
          if (srcLambda) {
            // As we have added a new input we need to clone the lambda which
            // loads that input.
            auto newLambda = cast<LambdaOp>(srcLambda->clone(remapper));
            // Add the lambda to the kernel.
            dstSymTable.insert(newLambda, inserter.getInsertionPoint());
            // Add to the input interfaces so they can then be fused into later.
            dstKernel.setInputFusionInterface(dstKernel.getNumOperands() - 1,
                                              newLambda.getSymName());
            // Remap so any uses of the old lambda can be remapped onto the new
            // lambda when we fuse the main elementwise body.
            oldNameToNew[srcLambda.getSymName()] = newLambda.getSymName();
          } else {
            dstKernel.setInputFusionInterface(dstKernel.getNumOperands() - 1,
                                              "");
          }
        }
      }

      StringRef mainBody = *srcKernel.getFusionBodyLambda();
      auto srcElementwiseLambda = cast<LambdaOp>(srcSymTable.lookup(mainBody));

      // Clone the block.
      auto newOp = cast<LambdaOp>(srcElementwiseLambda->clone(remapper));

      // Track the calls to the output lambda[s] so they can be removed.
      llvm::SmallDenseSet<StringRef> outputFns;
      ArrayAttr outputHooks = dstKernel.getOutputFusionInterfaceAttr();
      for (StringRef name : outputHooks.getAsValueRange<StringAttr>())
        outputFns.insert(name);

      auto returnOp =
          cast<MOGG::OutputOp>(newOp.getBody().front().getTerminator());

      // When we inline the new body we will have lambda invokes representing
      // loads from the input lambdas. These symbols will be references to
      // lambdas in the other kernel so we need to remap onto the ones in this
      // kernel.
      for (auto invokeOp :
           llvm::make_early_inc_range(newOp.getOps<MOGG::LambdaInvokeOp>())) {
        StringRef oldName = invokeOp.getLambda().getLeafReference().strref();

        // Remove all references to the old stores.
        if (outputFns.contains(oldName)) {
          // Add the values we previously stored as the output.
          for (Value arg : invokeOp.getOperands()) {
            // Add everything but indices.
            if (!isa<MOGG::IndicesType>(arg.getType()))
              returnOp->insertOperands(returnOp.getNumOperands(), arg);
          }

          invokeOp.erase();
        } else {
          auto itr = oldNameToNew.find(oldName);
          if (itr != oldNameToNew.end()) {
            invokeOp.setLambdaAttr(
                FlatSymbolRefAttr::get(invokeOp.getContext(), itr->second));
          }
        }
      }

      // Add the new lambda to the table.
      dstSymTable.insert(newOp, inserter.getInsertionPoint());
      inserter.setInsertionPointAfter(newOp);

      // Replace all uses of the old input lambda inside the main body with the
      // inlined body of the replacement lambda. The only reason we don't
      // replace it with a call is because it returns void so we would need to
      // create a new one for a legal call.
      auto uses = *dstSymTable.getSymbolUses(dstKernelInputLambdaName,
                                             dstElemwiseLambda);
      for (auto use : llvm::make_early_inc_range(uses)) {
        auto toReplace = cast<MOGG::LambdaInvokeOp>(use.getUser());
        // Inline the first invoke and then reuse those values.
        if (!previouslyFusedValue) {
          auto newValues = inlineInvoke(toReplace, newOp);
          ASSERT_STREAM(newValues.size() == 1)
              << "Currently assuming one return from elementwise ops when "
                 "fusing but got "
              << newValues.size();
          previouslyFusedValue = newValues.front();
        } else {
          toReplace->replaceAllUsesWith(ValueRange{previouslyFusedValue});
        }

        // Remove the lambda if we didn't already remove it.
        auto op = SymbolTable::lookupSymbolIn(dstKernel, toReplace.getLambda());
        if (op)
          op->erase();

        // Remove the call to the old input lambda.
        toReplace.erase();
      }

      // All uses have been inlined so get rid of the new lambda.
      newOp.erase();

      // Remove the argument we have effectively inlined.
      dstKernel.removeOperandAtIndex(fusionIndex);
    }
  }

  // Replace all loads from `dstKernelInputLambdaName` in `dstElemwiseLambda`
  // with the existing value `newVal`.
  void replaceLoadWithExistingUse(MOGG::KernelOp dst,
                                  StringAttr dstKernelInputLambdaName,
                                  MOGG::LambdaOp dstElemwiseLambda,
                                  mlir::OpBuilder &builder, Value newVal) {
    mlir::SymbolTable dstSymTable{dst};

    // The input lambda we should remove.
    auto lambdaToRemove =
        cast<LambdaOp>(dstSymTable.lookup(dstKernelInputLambdaName));

    // Replace all uses inside the body of this lambda with the value we
    // already fused in.
    auto uses = *dstSymTable.getSymbolUses(lambdaToRemove, dstElemwiseLambda);
    for (auto use : llvm::make_early_inc_range(uses)) {
      auto toReplace = cast<MOGG::LambdaInvokeOp>(use.getUser());
      toReplace.replaceAllUsesWith(ValueRange{newVal});
      toReplace.erase();
    }

    lambdaToRemove.erase();
  }

  // The KGEN equivalent of fusion consists of replacing calls of the old input
  // lambda with calls to the elementwise lambda in the source kernel. This is
  // possible because they have the same signature.
  void fuseSrcKernelIntoDstKernelKGEN(KernelOp srcKernel, KernelOp dstKernel) {
    stageKernelsForFusion({srcKernel, dstKernel});

    auto fusionIndices = *getFusionIndices(srcKernel, dstKernel);

    IRMapping mapper;
    OpBuilder builder{dstKernel.getContext()};

    MOGG::OutputPlaceholdersOp dstOutput = dstKernel.getOutputPlaceholder();
    auto dstConverterOps = dstKernel.getOps<ToKGENOp>();
    if (dstConverterOps.empty()) {
      dstKernel.setInsertionPointToTop(builder);
    } else {
      SmallVector<ToKGENOp> dstConverterOpsArray(dstConverterOps.begin(),
                                                 dstConverterOps.end());
      builder.setInsertionPointAfter(dstConverterOpsArray.back());
    }

    KGEN::ParamDeclareRegionOp srcOutputLambda = srcKernel.findLambda(
        cast<StringAttr>(srcKernel.getOutputFusionInterfaceAttr()[0]));
    KGEN::ParamDeclareRegionOp srcElementwiseLambda =
        srcKernel.findLambda(srcKernel.getFusionBodyLambdaAttr());
    assert(srcElementwiseLambda);

    // Add each of the new operands to the kernel body we are fusing into and
    // the corresponding input lambdas.
    llvm::StringMap<int> srcInputLambdaPositions;
    uint64_t srcOperandIndex = 0;
    for (auto [srcOperand, srcArgument] : llvm::zip(
             srcKernel.getOperands(), srcKernel.getBody().getArguments())) {
      Value dstArgument = dstKernel.addOperand(srcOperand);
      mapper.map(srcArgument, dstArgument);
      srcInputLambdaPositions[srcKernel.getInputFusionInterfaceAt(
          srcOperandIndex++)] = dstKernel.getNumOperands() - 1;
    }

    StringAttr renamedSrcElementwiseLambda;
    mlir::SymbolTable dstSymTable{dstKernel};
    for (Operation &srcOp : srcKernel.getBody().getOps()) {
      if (isa<KGEN::CallOp, MOGG::OutputOp>(srcOp))
        continue;

      if (auto srcOutput = dyn_cast<MOGG::OutputPlaceholdersOp>(srcOp)) {
        // The duplicate ToKGEN will be CSE-ed.
        mapper.map(srcOutput.getResult(0), dstOutput.getResult(0));
        continue;
      }

      auto lambda = dyn_cast<KGEN::ParamDeclareRegionOp>(srcOp);
      if (lambda == srcOutputLambda)
        continue;

      auto newOp = builder.clone(srcOp, mapper);

      // Symbols have to be added through the table to avoid duplication.
      if (lambda) {
        auto newName = dstSymTable.insert(newOp, builder.getInsertionPoint());
        auto oldName = lambda.getParamDecl().getName();
        if (srcInputLambdaPositions.contains(oldName)) {
          dstKernel.setInputFusionInterface(srcInputLambdaPositions[oldName],
                                            newName);
        }

        if (lambda == srcElementwiseLambda)
          renamedSrcElementwiseLambda = newName;
      }
    }

    assert(renamedSrcElementwiseLambda);

    auto &dstKernelBody = dstKernel.getBody().front();
    for (int fusionIndex : llvm::reverse(fusionIndices)) {
      auto dstKernelArgument = dstKernelBody.getArgument(fusionIndex);

      StringAttr dstKernelInputLambdaName =
          dstKernel.getInputFusionInterfaceAt(fusionIndex);
      KGEN::ParamDeclareRegionOp dstInputLambda =
          dstKernel.findLambda(dstKernelInputLambdaName);

      // Now replace all references to the input lambda with the new lambda.
      mlir::AttrTypeReplacer walker;
      walker.addReplacement([&](KGEN::ParamDeclRefAttr attr)
                                -> std::optional<KGEN::ParamDeclRefAttr> {
        if (attr.getName() == dstKernelInputLambdaName)
          return KGEN::ParamDeclRefAttr::get(
              renamedSrcElementwiseLambda,
              srcElementwiseLambda.getParamDecl().getType());
        return std::nullopt;
      });
      walker.recursivelyReplaceElementsIn(dstKernel, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);

      dstInputLambda.erase();

      // By composition we've already guaranteed the pointer is not going to be
      // accessed. What this means is that although this tensor is used by some
      // function calls, it's never actually read/written from/to. This is
      // because functions (including lambdas) have access to more than their
      // operands. They can capture values outside of their scope and that's how
      // lambdas actually access the right tensor. This one will never be
      // accessed but it is necessary to not have to temper with the KGEN code
      // (changing the signature of whatever was taking in that argument).
      dstKernelArgument.replaceAllUsesWith(
          dstKernelBody.getArgument(dstKernelBody.getNumArguments() - 1));
      dstKernel.removeOperandAtIndex(fusionIndex);
    }
  }
};

} // namespace

} // namespace M::MOGG
