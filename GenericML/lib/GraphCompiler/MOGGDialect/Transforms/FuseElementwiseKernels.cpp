//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/Support/Tensor.h"
#include "KGEN/POPDialect/POPTypes.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/ADT/SetOperations.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEELEMENTWISE
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

static SmallVector<int> getFusionIndices(KernelOp src, KernelOp dst) {
  SmallVector<int> indices;
  auto srcOutputInterfaces =
      src.getOutputFusionInterfaceAttr().getAsValueRange<StringAttr>();
  auto dstInputInterfaces =
      dst.getInputFusionInterfaceAttr().getAsValueRange<StringAttr>();
  for (auto [i, operand] : llvm::enumerate(dst.getOperands())) {
    auto srcInterface = *(srcOutputInterfaces.begin() + 0);
    auto dstInterface = *(dstInputInterfaces.begin() + i);
    if (operand.getDefiningOp() == src && !srcInterface.empty() &&
        !dstInterface.empty())
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

  return true;
}

static bool isASrcElementwiseKernel(Operation *op) {
  if (!isAnElementwiseKernel(op))
    return false;

  auto kernelOp = cast<KernelOp>(op);
  if (!kernelOp.getOutputFusionInterface())
    return false;

  return true;
}

static bool isADstElementwiseKernel(Operation *op) {
  if (!isAnElementwiseKernel(op))
    return false;

  auto kernelOp = cast<KernelOp>(op);
  if (!kernelOp.getInputFusionInterface())
    return false;

  return true;
}

static inline bool kernelsAreCompatible(KernelOp src, KernelOp dst) {
  if (src->getBlock() != dst->getBlock() ||
      src.getIsKGEN() != dst.getIsKGEN()) {
    return false;
  }

  return true;
}

struct FuseElementwisePass
    : public impl::FuseElementwiseBase<FuseElementwisePass> {
  using FuseElementwiseBase::FuseElementwiseBase;

  void runOnOperation() override {
    // Elementwise fusion consists of fusing a src kernel into a dst kernel. The
    // src kernel is used by the dst kernel. Certain patterns like diamond
    // patterns can cause inefficient fusion (duplicate values in the body of
    // the elementwise). To fix this, src kernels are processed in reverse
    // topological order. Note that this is not strictly necessary. When kernels
    // are ultimately lowered to KGEN, CSE takes care of this.

    // First find all blocks that contain src elementwise kernels.
    MO::GraphOp graphOp = getOperation();
    llvm::SmallPtrSet<Block *, 2> blocksThatContainSrcElementwiseKernels;
    graphOp.walk([&](KernelOp kernelOp) {
      if (isASrcElementwiseKernel(kernelOp))
        blocksThatContainSrcElementwiseKernels.insert(kernelOp->getBlock());
    });

    // Then, for each block, sort its contents in topological order and gather
    // all src elementwise kernels.
    SmallVector<KernelOp> srcElementwiseKernels;
    for (Block *block : blocksThatContainSrcElementwiseKernels) {
      auto kernelOpsInBlock = block->getOps<KernelOp>();
      SmallVector<Operation *> kernelOpsInBlockAsArray(kernelOpsInBlock.begin(),
                                                       kernelOpsInBlock.end());
      // Not actually sorting because we must not temper with parametric shape
      // logic.
      mlir::computeTopologicalSorting(kernelOpsInBlockAsArray);
      for (Operation *op : kernelOpsInBlockAsArray) {
        if (isASrcElementwiseKernel(op))
          srcElementwiseKernels.push_back(cast<KernelOp>(op));
      }
    }

    // Finally, iterate through the src elementwise kernels in the reverse
    // topological sort and fuse with dst elementwise kernels.
    DenseSet<KernelOp> dstElementwiseKernels;
    for (KernelOp srcKernel : llvm::reverse(srcElementwiseKernels)) {
      auto users = srcKernel->getUsers();
      DenseSet<Operation *> uniqueUsers(users.begin(), users.end());

      const auto canFuseWith = [&](Operation *dstOp) {
        if (!isADstElementwiseKernel(dstOp))
          return false;

        auto dstKernel = cast<KernelOp>(dstOp);
        if (!kernelsAreCompatible(srcKernel, dstKernel))
          return false;

        return true;
      };

      // Do not attempt to fuse if the operation will not be DCE-ed later.
      if (!llvm::all_of(uniqueUsers, canFuseWith))
        continue;

      bool isKGEN = srcKernel.getIsKGEN().value_or(false);
      for (Operation *dstOp : uniqueUsers) {
        auto dstKernel = cast<KernelOp>(dstOp);
        if (isKGEN)
          fuseSrcKernelIntoDstKernelKGEN(srcKernel, dstKernel);
        else
          fuseSrcKernelIntoDstKernelMOGG(srcKernel, dstKernel);

        // Update the dst's summary to capture the producer's summary.
        dstKernel.setSummaryAttr(getSummaryString({srcKernel, dstKernel}));
        dstElementwiseKernels.insert(dstKernel);
      }
    }

    // The fusion may have re-ordered the contents of the dst kernels. Sort them
    // topologically.
    for (KernelOp dstKernel : dstElementwiseKernels)
      dstKernel.walk([](Block *block) { mlir::sortTopologically(block); });
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
    auto fusionIndices = getFusionIndices(srcKernel, dstKernel);
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
          assert(
              newValues.size() == 1 &&
              "Currently assuming one return from elementwise ops when fusing");
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

  static KGEN::ParamDeclareRegionOp findLambda(KernelOp kernelOp,
                                               StringAttr name) {
    // FIXME: Use mlir::SymbolTable dstSymTable{kernelOp};
    for (auto op : kernelOp.getBody().getOps<KGEN::ParamDeclareRegionOp>()) {
      if (op.getParamDecl().getName() == name)
        return op;
    }

    return nullptr;
  }

  // The KGEN equivalent of fusion consists of replacing calls of the old input
  // lambda with calls to the elementwise lambda in the source kernel. This is
  // possible because they have the same signature.
  void fuseSrcKernelIntoDstKernelKGEN(KernelOp srcKernel, KernelOp dstKernel) {
    auto fusionIndices = getFusionIndices(srcKernel, dstKernel);
    if (fusionIndices.empty())
      return;

    IRMapping mapper;
    OpBuilder builder{dstKernel.getContext()};

    ArrayAttr dstKernelHooks = dstKernel.getInputFusionInterfaceAttr();
    MOGG::OutputPlaceholdersOp dstOutput = dstKernel.getOutputPlaceholder();
    auto dstConverterOps = dstKernel.getOps<ToKGENOp>();
    SmallVector<ToKGENOp> dstConverterOpsArray(dstConverterOps.begin(),
                                               dstConverterOps.end());
    ToKGENOp entryPoint = dstConverterOpsArray.back();

    /// ASSUMPTION: Single output kernel.
    KGEN::ParamDeclareRegionOp srcOutputLambda = findLambda(
        srcKernel,
        cast<StringAttr>(srcKernel.getOutputFusionInterfaceAttr()[0]));
    KGEN::ParamDeclareRegionOp srcElementwiseLambda =
        findLambda(srcKernel, srcKernel.getFusionBodyLambdaAttr());

    // Add each of the new operands to the kernel body we are fusing into and
    // the corresponding input lambdas.
    /// FIXME: (#18013) Simplify duplicate operands.
    builder.setInsertionPointAfter(entryPoint);
    mlir::SymbolTable dstSymTable{dstKernel};
    llvm::StringMap<int> srcInputLambdaPositions;
    for (auto [srcOperand, srcArgument, srcInputLambdaName] : llvm::zip(
             srcKernel.getOperands(), srcKernel.getBody().getArguments(),
             srcKernel.getInputFusionInterface()->getAsRange<StringAttr>())) {
      Value dstArgument = dstKernel.addOperand(srcOperand);
      mapper.map(srcArgument, dstArgument);
      srcInputLambdaPositions[srcInputLambdaName] =
          dstKernel.getNumOperands() - 1;
    }

    for (Operation &srcOp : srcKernel.getBody().getOps()) {
      if (auto moggCallOp = dyn_cast<MOGG::CallOp>(srcOp)) {
        if (moggCallOp.getMojoKernelTarget() == "mark_output_chain_ready")
          continue;
      }

      if (isa<KGEN::CallOp, MOGG::OutputOp>(srcOp))
        continue;

      if (auto srcOutput = dyn_cast<MOGG::OutputPlaceholdersOp>(srcOp)) {
        /// ASSUMPTION: Single output kernel.
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
      }
    }

    auto &dstKernelBody = dstKernel.getBody().front();
    for (int fusionIndex : llvm::reverse(fusionIndices)) {
      builder.setInsertionPointAfter(entryPoint);

      // The input lambda we are going to replace with the fused body.
      auto dstKernelArgument = dstKernelBody.getArgument(fusionIndex);

      StringAttr dstKernelInputLambdaName =
          cast<StringAttr>(dstKernelHooks[fusionIndex]);
      KGEN::ParamDeclareRegionOp dstInputLambda =
          findLambda(dstKernel, dstKernelInputLambdaName);

      // Now replace all references to the input lambda with the new lambda.
      mlir::AttrTypeReplacer walker;
      walker.addReplacement([&](KGEN::ParamDeclRefAttr attr)
                                -> std::optional<KGEN::ParamDeclRefAttr> {
        if (attr.getName() == dstKernelInputLambdaName)
          return KGEN::ParamDeclRefAttr::get(
              srcKernel.getFusionBodyLambdaAttr(),
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
      // ambdas actually access the right tensor. This one will never be
      // accessed but it is necessary to not have to temper with the KGEN code.
      builder.setInsertionPoint(entryPoint);
      dstKernelArgument.replaceAllUsesWith(
          dstKernelBody.getArgument(dstKernelBody.getNumArguments() - 1));
      dstKernel.removeOperandAtIndex(fusionIndex);
    }
  }
};

} // namespace

} // namespace M::MOGG
