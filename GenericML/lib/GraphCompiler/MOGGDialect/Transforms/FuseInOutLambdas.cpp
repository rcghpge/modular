//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Analysis/MODialect/ParametricTopologyAnalysis.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoTargetFunctions.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/AssertStream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEINOUTLAMBDAS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// TODO(#22349): Remove checks when matmul/conv/concat kernels respect the
// fusion contract.
static bool containsMatmulOrConvOrConcat(MOGG::KernelOp kernel) {
  StringRef summary = kernel.getSummaryAttr().strref();
  return (summary.contains(MO::ConcatOp::getOperationName()) ||
          summary.contains(MO::MatmulOp::getOperationName()) ||
          summary.contains(MO::BatchMatmulOp::getOperationName()) ||
          summary.contains(MO::ConvOp::getOperationName()) ||
          summary.contains(MO::ConvTransposeOp::getOperationName()));
}

static bool containsMutableStore(MOGG::KernelOp kernel) {
  StringRef summary = kernel.getSummaryAttr().strref();
  return summary.contains(MO::MutableStoreOp::getOperationName());
}

struct FuseInOutLambdas : public impl::FuseInOutLambdasBase<FuseInOutLambdas> {
  using FuseInOutLambdasBase::FuseInOutLambdasBase;

  FuseInOutLambdas() = default;
  FuseInOutLambdas(const FuseInOutLambdas &pass) {}

  Statistic numberOfFusedPrologues{this, "numberOfFusedPrologues",
                                   "Number of fused input elementwise kernels"};
  Statistic numberOfFusedEpilogues{
      this, "numberOfFusedEpilogues",
      "Number of fused output elementwise kernels"};

  void fuseExpKernelInLambda(MO::GraphOp moGraph) {
    IRRewriter rewriter(moGraph->getContext());
    SmallVector<Operation *> toErase;
    moGraph.walk([&](MOGG::ExpKernelOp dst) {
      if (dst.getElemwiseBodyLambda())
        return;

      uint64_t numOperands = dst->getNumOperands();
      for (uint64_t fusionIdx = 0; fusionIdx < numOperands; ++fusionIdx) {
        if (!dst.getBoundInputLambda(fusionIdx))
          continue;

        auto providerOp = dst->getOperand(fusionIdx).getDefiningOp();
        if (!providerOp || !providerOp->hasOneUse())
          continue;

        if (providerOp->getNumResults() != 1)
          continue;

        auto src = dyn_cast<MOGG::ExpKernelOp>(providerOp);
        if (!src)
          continue;

        if (!src.getElemwiseBodyLambda())
          continue;

        ++numberOfFusedPrologues;

        auto inputLambda = dst.getBoundInputLambda(fusionIdx);
        auto inputLambdaBody = inputLambda.getBody();
        Value inVal = inputLambdaBody->getArgument(0);
        Value indices = inputLambdaBody->getArgument(1);

        auto elementwiseLambda = src.getElemwiseBodyLambda();
        Block *elementwiseLambdaBody = elementwiseLambda.getBody();
        inVal.replaceAllUsesWith(
            elementwiseLambdaBody->getTerminator()->getOperand(0));
        inputLambdaBody->eraseArgument(0);
        inputLambda->eraseOperand(0);
        dst->eraseOperand(fusionIdx);
        dst.getExecBody().eraseArgument(fusionIdx);
        elementwiseLambdaBody->getTerminator()->erase();

        rewriter.setInsertionPointToStart(elementwiseLambdaBody);
        for (uint64_t idx = 0; idx < elementwiseLambda->getNumOperands();
             ++idx) {
          Value outsideValue = elementwiseLambda.getOperand(idx);
          Value insideValue = elementwiseLambdaBody->getArgument(idx);
          Value loadedValue = rewriter.create<MOGG::LoadTensorOp>(
              outsideValue.getLoc(), insideValue.getType(), outsideValue,
              indices);
          insideValue.replaceAllUsesWith(loadedValue);
        }
        elementwiseLambdaBody
            ->getArgument(elementwiseLambda->getNumOperands() - 1)
            .replaceAllUsesWith(indices);

        dst->insertOperands(dst->getNumOperands(), src->getOperands());
        rewriter.setInsertionPoint(inputLambdaBody->getTerminator());
        for (auto [idx, operand] : llvm::enumerate(src->getOperands())) {
          auto oldBlockArgument = src.getExecBody().getArgument(idx);
          auto newBlockArgument = dst.getExecBody().insertArgument(
              dst.getExecBody().getNumArguments(), oldBlockArgument.getType(),
              operand.getLoc());
          oldBlockArgument.replaceAllUsesWith(newBlockArgument);
        }

        rewriter.setInsertionPointToStart(inputLambdaBody);
        elementwiseLambdaBody->eraseArguments(
            0, elementwiseLambdaBody->getNumArguments());
        rewriter.inlineBlockBefore(elementwiseLambdaBody, inputLambdaBody,
                                   rewriter.getInsertionPoint());
        src->erase();
      }
    });
  }

  void fuseExpKernelOutLambda(MO::GraphOp moGraph) {
    auto getFusibleUser = [](Value v) -> MOGG::ExpKernelOp {
      if (!v.hasOneUse())
        return nullptr;

      auto kernelUser = dyn_cast<MOGG::ExpKernelOp>(*v.getUsers().begin());
      if (!kernelUser)
        return nullptr;

      // elementwise kernel user.
      return kernelUser.getElemwiseBodyLambda() ? kernelUser : nullptr;
    };

    IRRewriter rewriter(moGraph->getContext());
    SmallVector<Operation *> toErase;
    moGraph.walk([&](MOGG::ExpKernelOp dst) {
      // Skips elementwise kernel.
      if (dst.getElemwiseBodyLambda())
        return;

      SmallVector<std::pair<OpResult, Block *>> outputLambdas;
      for (OpResult result : dst->getResults()) {
        auto outLambda = dst.getBoundOutputLambda(result.getResultNumber());

        if (!outLambda)
          continue;

        Block *lambdaBlock = outLambda.getBody();
        outputLambdas.push_back({result, lambdaBlock});
      }

      for (auto [output, outputLambda] : outputLambdas) {
        MOGG::ExpKernelOp src = getFusibleUser(output);
        if (!src)
          continue;

        ++numberOfFusedEpilogues;

        // Update the type after fusion.
        auto fusedResultTp = cast<MO::TensorType>(src->getResult(0).getType());
        output.setType(fusedResultTp);
        auto outOps = dst.getExecBody().getOps<MOGG::OutputPlaceholdersOp>();
        ASSERT_STREAM(llvm::range_size(outOps) == 1,
                      "Expect one mogg.output_placeholder per mogg.kernel");
        MOGG::OutputPlaceholdersOp outHolderOp = *outOps.begin();
        outHolderOp->getResult(output.getResultNumber()).setType(fusedResultTp);

        // Merge all the operands into the target kernel.
        auto fusedOperands = src->getOpOperands();
        SmallVector<Value> fusedArgs(fusedOperands.size(), nullptr);
        for (auto [idx, operand] : llvm::enumerate(fusedOperands)) {
          dst->insertOperands(idx, operand.get());
          fusedArgs[idx] = dst.getExecBody().insertArgument(
              idx, src.getExecBody().getArgument(idx).getType(),
              operand.get().getLoc());
        }

        // Inlining elementwise lambda into output lambda
        rewriter.setInsertionPoint(outputLambda->getTerminator());
        Value outVal = outputLambda->getTerminator()->getOperands().front();
        Value indices = outputLambda->getTerminator()->getOperands().back();

        Block *elemFuseBody = src.getElemwiseBodyLambda().getBody();
        unsigned fusedIdx = std::distance(
            src->operand_begin(), llvm::find(src->getOperands(), output));
        Value yieldVal = elemFuseBody->getTerminator()->getOperand(0);
        outVal.replaceAllUsesWith(yieldVal);
        elemFuseBody->getTerminator()->erase();

        for (auto [idx, val] : llvm::enumerate(fusedArgs)) {
          // Relink SSA;
          if (fusedIdx == idx) {
            elemFuseBody->getArgument(idx).replaceAllUsesWith(outVal);
          } else {
            // We are fusing a N-ary elementwise kernel, for operands other than
            // the fused one, we produce the value by capture the tensor and
            // load.
            Value loaded = rewriter.create<MOGG::LoadTensorOp>(
                val.getLoc(),
                cast<MO::TensorType>(val.getType()).getElementType(), val,
                indices);
            elemFuseBody->getArgument(idx).replaceAllUsesWith(loaded);
          }
        }
        // All the uses of the original argument are taken care of above, remove
        // all the arguments and inline.
        elemFuseBody->eraseArguments(0, elemFuseBody->getNumArguments());
        rewriter.inlineBlockBefore(elemFuseBody, outputLambda,
                                   rewriter.getInsertionPoint());
        cast<MOGG::BindLambdaOp>(outputLambda->getParentOp())
            .getResult()
            .setType(fusedResultTp);

        // Remove fused operation.
        dst->eraseOperand(fusedIdx);
        dst.getExecBody().eraseArgument(fusedIdx);

        // TODO: allocate accumulate buffer for mix-precision matmul.
        // TODO: we should stop on elementwise kernel and look the producer
        // to be fused into so that the elementwise kernel can be safely
        // erased during walking without invalidate tree iterator.
        src->replaceAllUsesWith(dst->getResults());
        toErase.push_back(src);
      }
    });

    for (auto op : toErase)
      op->erase();
  }

  bool canDoInputFusion(MOGG::KernelOp kernel, ArrayAttr fusionInterfaces,
                        uint64_t inputIdx) {
    // For in-place allocated kernel, we disable for now.
    if (kernel.isInPlaceAlloc())
      return false;

    // Should be unreachable sanity check. Ensure this is not an elementwise
    // kernel (as if it where after fusion we would trigger more fusions).
    auto isElemwise = kernel.getFusionBodyLambda();
    if (isElemwise)
      return false;

    // Input fusion interface must be non null.
    auto inputFusion = SmallVector<StringAttr>(
        fusionInterfaces.getAsRange<StringAttr>())[inputIdx];
    if (inputFusion.empty())
      return false;

    // Parent must have at most one unique user operation which is the kernel we
    // are fusing into.
    auto parentValue = kernel.getOperand(inputIdx);
    for (auto user : parentValue.getUsers()) {
      if (user != kernel)
        return false;
    }

    // Parent op must be a kernel op.
    auto parentKernel =
        dyn_cast_or_null<MOGG::KernelOp>(parentValue.getDefiningOp());
    if (!parentKernel)
      return false;

    // Only fuse within the same block and not across, i.e, control flow.
    if (kernel->getBlock() != parentKernel->getBlock())
      return false;

    // Check that the kernel has a single output.
    if (parentKernel->getNumResults() > 1)
      return false;

    // Check the kernel is a fusable elementwise.
    if (!parentKernel.getFusionBodyLambda())
      return false;

    // Kernel must expose output fusion contract.
    auto outputInterfaceAttr = parentKernel.getOutputFusionInterface();
    if (!outputInterfaceAttr)
      return false;

    // Fusion interface must be non-null.
    auto outputInterface =
        *outputInterfaceAttr->getAsRange<StringAttr>().begin();
    if (outputInterface.empty())
      return false;

    // Must have the same device
    if (kernel.getDeviceRef() != parentKernel.getDeviceRef())
      return false;

    return true;
  }

  void fuseIntoInputLambdaKGEN(MOGG::KernelOp kernelToFuseInto,
                               ArrayAttr fusionInterfaces, uint64_t inputIdx,
                               SmallVectorImpl<KernelOp> &opsToErase,
                               MojoLibraryAnalysis &analysis) {
    ++numberOfFusedPrologues;
    MLIRContext *ctx = kernelToFuseInto.getContext();
    OpBuilder builder{ctx};

    auto inputKernel = cast<MOGG::KernelOp>(
        kernelToFuseInto.getOperand(inputIdx).getDefiningOp());

    Block &dstKernelBody = kernelToFuseInto.getBody().front();
    Block &inputKernelBody = inputKernel.getBody().front();

    auto fusionArgument = dstKernelBody.getArgument(inputIdx);

    // Step 1: Add extra operands to kernelTofuseInto.
    // All arguments of inputKernel are added.
    // The original fused argument will be removed at the end.

    IRMapping oldToNew;
    oldToNew.map(inputKernel.getOutputPlaceholder()->getResult(0),
                 fusionArgument);

    for (auto [operand, blockArg] :
         llvm::zip(inputKernel.getOperands(), inputKernelBody.getArguments())) {
      Value newBlockArg = kernelToFuseInto.addOperand(operand);
      oldToNew.map(blockArg, newBlockArg);
    }

    // Step 2: We clone all non-trivial operators in the user kernel into the
    // src kernel. Here, we ignore all the ParamDeclareRegionOp, which we will
    // take care of later in a more specific way.

    SmallVector<KGEN::ParamDeclareOp> newParamDeclOps;

    kernelToFuseInto.setInsertionPointToTop(builder);

    // Creating at most one empty tensors op. This will be used as replacement
    // for the fused argument which has some uses doing nothing.
    auto fakeTensors = kernelToFuseInto.getCreateEmptyTensor();
    if (!fakeTensors) {
      fakeTensors = builder.create<MOGG::CreateEmptyTensorsOp>(
          kernelToFuseInto.getLoc(), fusionArgument.getType());
    } else {
      auto fakeTensorTypes = SmallVector<Type>(fakeTensors->getResultTypes());
      fakeTensorTypes.push_back(fusionArgument.getType());
      auto newFakeTensors = builder.create<MOGG::CreateEmptyTensorsOp>(
          kernelToFuseInto.getLoc(), fakeTensorTypes);
      fakeTensors->replaceAllUsesWith(newFakeTensors.getResults().drop_back());
      fakeTensors.erase();
      fakeTensors = newFakeTensors;
      kernelToFuseInto.setInsertionPointToTop(builder);
    }
    Value newFakeFusedArgument = fakeTensors->getResults().back();

    // Do not copy foreach, or any ops which are used to handle the fact it
    // raises
    DenseSet<Operation *> skipCopy;
    for (auto &op : inputKernel.getOps()) {
      auto callOp = dyn_cast<KGEN::CallOp>(op);
      if (!callOp)
        continue;
      StringRef funcName =
          cast<mlir::FlatSymbolRefAttr>(callOp.getCalleeSymbol()).getValue();

      if (!funcName.starts_with(Utils::DPSTensorForEach) &&
          !funcName.starts_with(Utils::DPSTensorCopy))
        continue;

      auto res = MO::getRootedComponents(&op);
      skipCopy.insert(res.begin(), res.end());
    }

    for (auto &op : inputKernel.getOps()) {
      if (skipCopy.contains(&op))
        continue;

      if (auto toKgen = dyn_cast<MOGG::ToKGENOp>(op)) {
        // We clone ToKGENOp except for the one related to the output.
        if (toKgen.getInput().getDefiningOp<MOGG::OutputPlaceholdersOp>())
          continue;

        auto newToKgen = cast<MOGG::ToKGENOp>(builder.clone(op, oldToNew));
        oldToNew.map(toKgen.getResults(), newToKgen.getResults());
      } else if (isa<KGEN::ParamDeclareOp>(op)) {
        auto newOp = cast<KGEN::ParamDeclareOp>(builder.clone(op));
        newParamDeclOps.push_back(newOp);
      } else if (isa<MOGG::OutputOp, KGEN::ParamDeclareRegionOp,
                     MOGG::OutputPlaceholdersOp>(op)) {
        // ParamDeclareRegionOp will be cloned later.
        // OutputPlaceholdersOp relates to the temporary fused output that
        // doesn't need to be cloned.
      } else {
        oldToNew.map(op.getResults(),
                     builder.clone(op, oldToNew)->getResults());
      }
    }

    // Step 3) Clone all of the lambda regions of inputKernel. (except the
    // elementwise / output one).
    // Insert them right before the destination input
    // lambda.

    mlir::SymbolTable inputSymTable{inputKernel};
    mlir::SymbolTable dstSymTable{kernelToFuseInto};

    StringAttr dstInputLambdaName =
        kernelToFuseInto.getInputFusionInterfaceAt(inputIdx);
    assert(!dstInputLambdaName.empty());
    auto dstInputLambdaOp = cast<KGEN::ParamDeclareRegionOp>(
        dstSymTable.lookup(dstInputLambdaName));

    builder.setInsertionPoint(dstInputLambdaOp);

    llvm::DenseSet<StringRef> lambdaSkipSet;
    // Skip the output fusion interface.
    lambdaSkipSet.insert(inputKernel.getOutputFusionInterfaceAt(0));
    // SKip the body interface
    lambdaSkipSet.insert(inputKernel.getFusionBodyLambdaAttr());

    // Clone all the regions.
    for (auto declRegion : inputKernel.getOps<KGEN::ParamDeclareRegionOp>()) {
      if (!lambdaSkipSet.contains(declRegion.getParamDeclAttr().getName())) {
        builder.clone(*declRegion.getOperation(), oldToNew);
      }
    }

    // Step 4. Clone the elementwise kernel call inside fuseIntoKernel call

    // It is guaranteed that the user kernel is an elementwise op.
    StringAttr elementwiseLambdaName = inputKernel.getFusionBodyLambdaAttr();
    auto elementwiseOp = inputKernel.findLambda(elementwiseLambdaName);

    // Get the original load inside the dst input lambda.
    // We want to replace it with the elementwise function call.
    KGEN::CallOp dstInputLoadOp =
        *dstInputLambdaOp.getOps<KGEN::CallOp>().begin();
    assert(dstInputLoadOp);

    // Remap arguments for the elementwise call.
    // Map the old elementwise indices argument to the new input indices
    // argument.
    oldToNew.map(elementwiseOp.getFunctionBody().getArgument(0),
                 dstInputLambdaOp.getFunctionBody().getArgument(0));

    // Clone the elementwise kernel body.
    SmallVector<Operation *> clonedElementwiseBodyOps;
    builder.setInsertionPoint(dstInputLoadOp);
    for (Operation &elemwOp : elementwiseOp.getBody()->without_terminator()) {
      clonedElementwiseBodyOps.push_back(builder.clone(elemwOp, oldToNew));
    }

    // Replace the uses of the load with the elementwise output instead.
    Operation *elemwiseBodyReturnOp = elementwiseOp.getBody()->getTerminator();
    assert(dstInputLoadOp->getNumResults() == 1 &&
           elemwiseBodyReturnOp->getNumOperands() == 1);
    dstInputLoadOp.getResult(0).replaceAllUsesWith(
        oldToNew.lookup(elemwiseBodyReturnOp->getOperand(0)));
    // Get rid of the load.
    dstInputLoadOp->erase();

    // Step 5. Remap the parameters for the elementwise body.
    DenseMap<KGEN::ParamDeclRefAttr, TypedAttr> paramRemaps;
    auto oldElementwiseParams = elementwiseOp.getInputParams();
    auto newElementwiseParams = dstInputLambdaOp.getInputParams();

    for (auto [oldParam, newParam] :
         llvm::zip(oldElementwiseParams, newElementwiseParams)) {
      auto oldDeclRef =
          KGEN::ParamDeclRefAttr::get(oldParam.getName(), oldParam.getType());
      auto newDeclRef =
          KGEN::ParamDeclRefAttr::get(newParam.getName(), newParam.getType());
      paramRemaps[oldDeclRef] = newDeclRef;
    }

    mlir::AttrTypeReplacer walker;
    walker.addReplacement(
        [&](KGEN::ParamDeclRefAttr attr) -> std::optional<TypedAttr> {
          auto itr = paramRemaps.find(attr);
          if (itr != paramRemaps.end())
            return itr->second;
          return std::nullopt;
        });

    for (Operation *newElemwOp : clonedElementwiseBodyOps) {
      walker.recursivelyReplaceElementsIn(newElemwOp,
                                          /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);
    }

    // Step 6: Get rid of the fused argument (elementwise op output)
    // It still has some uses, which shouldn't do anything since it should be
    // read by the lambda. Eg `x._fused_load()` actually call the input_lambda
    // without ever using `x`.
    // Let's replace it with an empty tensor instead.
    fusionArgument.replaceAllUsesWith(newFakeFusedArgument);

    // Remove the fused argument from the final kernel.
    kernelToFuseInto.removeOperandAtIndex(inputIdx);

    // Step 7: Cleanup the kernel.

    // Ensure only one instance of these ops exist after fusing
    mergeAllocationOfType<MOGG::StackAllocateTensorsOp>(kernelToFuseInto);
    mergeAllocationOfType<MOGG::CreateEmptyTensorsOp>(kernelToFuseInto);

    // Update the lambdas_have_fusion parameter for the extensibility kernels
    setKGENKernelParamValueIfPresent(
        kernelToFuseInto, analysis,
        KGEN::MOGGPreElab::kMOGGLambdasHaveFusionLabel,
        BoolAttr::get(&getContext(), true));

    kernelToFuseInto.setSummaryAttr(
        getSummaryString({inputKernel, kernelToFuseInto}, FusionType::HARD));

    // Get rid of the original input kernel.
    opsToErase.push_back(inputKernel);

    // Get rid of the unused declarations for some of the removed lambdas.
    clearUnusedParamDecls(kernelToFuseInto, newParamDeclOps);
  }

  /// Returns true if we can fuse into the output lambda. This sanity checks a
  /// number of constraints. Mainly it checks:
  /// 1. We are an op which can be fused into
  /// 2. We are NOT an elementwise op
  /// 3. Result we can fuse into is only used by an elementwise op.
  ///
  /// This ensures the fusion is legal. Many of the checks can be moved into the
  /// canonicalizer / verifier.
  llvm::BitVector canDoOutputFusion(MOGG::KernelOp kernel,
                                    ArrayAttr fusionInterface) {
    // Unreachable: Ambigious what we should actually fuse. Previous passes
    // should have maintained 1:1 relationship of output to output interfaces.
    if (fusionInterface.size() != kernel.getNumResults())
      return llvm::BitVector();

    // For in-place allocated kernel, we disable for now.
    if (kernel.isInPlaceAlloc())
      return llvm::BitVector();

    // Should be unreachable sanity check. Ensure this is not an elementwise
    // kernel (as if it where after fusion we would trigger more fusions).
    auto isElemwise = kernel.getFusionBodyLambda();
    if (isElemwise.has_value())
      return llvm::BitVector();

    const bool producerIsMatmulConvOrConcat =
        containsMatmulOrConvOrConcat(kernel);

    llvm::BitVector fusableOutput(kernel.getNumNonChainResults(), true);
    for (auto [idx, result] : llvm::enumerate(kernel.getNonChainResults())) {
      // Output can only have one user and that user must be a fusable
      // elementwise kernel.
      if (result.getUsers().empty()) {
        // The last all_reduce only used one of its results?
        fusableOutput.reset(idx);
        continue;
      }

      if (llvm::range_size(result.getUsers()) > 1) {
        fusableOutput.reset(idx);
        continue;
      }

      // Check it's a kernel.
      MOGG::KernelOp user = dyn_cast<MOGG::KernelOp>(*result.user_begin());
      if (!user) {
        fusableOutput.reset(idx);
        continue;
      }

      // Only fuse within the same block and not across, i.e, control flow.
      if (kernel->getBlock() != user->getBlock()) {
        fusableOutput.reset(idx);
        continue;
      }

      // Check that it has a single output.
      if (user->getNumResults() != 1) {
        fusableOutput.reset(idx);
        continue;
      }

      // Check the kernel is a fusable elementwise.
      if (!user.getFusionBodyLambda()) {
        fusableOutput.reset(idx);
        continue;
      }

      // TODO(GEX-2213): Come up with more general mechanisms for when fusion
      // with kernels handling mutable values is permissible.
      if (containsMutableStore(user) && !producerIsMatmulConvOrConcat) {
        fusableOutput.reset(idx);
        continue;
      }

      // Kernel must expose input fusion contract.
      std::optional<ArrayAttr> inputInterfaces = user.getInputFusionInterface();
      if (!inputInterfaces.has_value()) {
        fusableOutput.reset(idx);
        continue;
      }

      // It must expose a valid input lambda attribute for every operand which
      // uses this.
      for (auto [operand, attr] : llvm::zip(
               user.getOperands(), inputInterfaces->getAsRange<StringAttr>())) {
        if (operand == result && attr.strref().empty()) {
          fusableOutput.reset(idx);
          continue;
        }
      }

      // Check the output lambda fusion hook.
      if (cast<StringAttr>(fusionInterface[idx]).strref().empty()) {
        fusableOutput.reset(idx);
        continue;
      }

      // For in-place kernels, verify they have the required fusion body output
      // lambda attribute
      const bool userIsInPlace = user.maybeGetInChain() &&
                                 user.maybeGetOutChain() &&
                                 user.getNumResults() == 1;
      if (userIsInPlace) {
        auto fusionBodyOutputLambdaAttr = user->getDiscardableAttr(
            KGEN::MOGGPreElab::kMOGGFusionBodyOutputLambda);
        if (!fusionBodyOutputLambdaAttr ||
            cast<StringAttr>(fusionBodyOutputLambdaAttr).empty()) {
          fusableOutput.reset(idx);
          continue;
        }
      }

      // Must have the same device.
      if (kernel.getResults().size() == 1) {
        if (kernel.getDeviceRef() != user.getDeviceRef()) {
          fusableOutput.reset(idx);
          continue;
        }
      } else if (cast<MO::TensorType>(result.getType()).getDeviceRef() !=
                 user.getDeviceRef()) {
        // AllReduce is a distributed ops, compare the device ref on a
        // particular tensor result individually.
        fusableOutput.reset(idx);
        continue;
      }

      // Check chain compatibility if both kernels have input and output chains
      auto srcInChain = kernel.maybeGetInChain();
      auto srcOutChain = kernel.maybeGetOutChain();
      auto dstInChain = user.maybeGetInChain();
      auto dstOutChain = user.maybeGetOutChain();

      if (srcInChain.has_value() && srcOutChain.has_value() &&
          dstInChain.has_value() && dstOutChain.has_value()) {
        // Get actual chain values
        Value srcOutChain = kernel.getOutChain();
        Value dstInChain = user.getInChain();

        // Ensure the input chain to user kernel is output chain from source
        // kernel
        if (dstInChain != srcOutChain) {
          fusableOutput.reset(idx);
          continue;
        }

        // Ensure that user kernel is the sole consumer of the output chain from
        // source kernel.
        // Strictly speaking source kernel having more than one consumer of
        // its output chain would be UB when those consumers perform writes to
        // mo.buffers -- skip fusion here to make the presence of that situation
        // clearer.
        if (!srcOutChain.hasOneUse()) {
          fusableOutput.reset(idx);
          continue;
        }
      }
    }

    return fusableOutput;
  }

  // Fuse the user of `kernelToFuseInto` into the src kernel with KGEN.
  KernelOp fuseIntoOutputLambdaKGEN(KernelOp kernelToFuseInto,
                                    StringAttr outputLambdaName,
                                    unsigned outputIdx,
                                    SmallVector<KernelOp> &opsToErase,
                                    MojoLibraryAnalysis &analysis) {

    // In brief, we have several steps to implement this fusion:

    // Step 1: Update the operands and block arguments of src kernel to absorb
    // the user kernel.

    // Step 2: Clone operators in the user kernel into the src kernel.

    // Step 3: Clone all input lamba functions into the body and output lambda
    // of the src kernel.

    // Step 4: Clone all the elementwise lambdas into the output lambda of the
    // src kernel.

    // Step 5: Remap all the parameters of the operators we cloned if needed.

    // Step 6: Clean up and handle the mixed precision case.

    ++numberOfFusedEpilogues;

    MLIRContext *ctx = kernelToFuseInto.getContext();

    OpBuilder builder{ctx};

    // Maps kernel's old operand to new operand.
    IRMapping oldToNew;

    // Guarded to only fuse 1 output kernels.
    Value originalResult = kernelToFuseInto.getResult(outputIdx);

    // We've already guarded that we only have one user.
    MOGG::KernelOp userToFuseIn =
        cast<MOGG::KernelOp>(*originalResult.user_begin());

    Value newExpectedResult = userToFuseIn->getResult(0);

    OutputPlaceholdersOp originalOuts = kernelToFuseInto.getOutputPlaceholder();
    OutputPlaceholdersOp userOuts = userToFuseIn.getOutputPlaceholder();

    // TODO: in the can do output fusion checks enforce these conditions
    const bool userIsInPlace = userToFuseIn.maybeGetInChain() &&
                               userToFuseIn.maybeGetOutChain() &&
                               userToFuseIn.getNumResults() == 1;

    auto maybeSrcInChain = kernelToFuseInto.maybeGetInChain();
    auto maybeSrcOutChain = kernelToFuseInto.maybeGetOutChain();

    if (userIsInPlace) {
      // Just in case passes swap around and canonicalization doesn't run before
      // this.
      if (userOuts && userOuts.getNumResults() == 0) {
        userOuts.erase();
        userOuts = nullptr;
      }

      ASSERT_STREAM(
          userOuts == nullptr,
          "user kernel is in-place allocated but has output placeholder");
    }

    const bool isMatmulOrConvOrConcat =
        containsMatmulOrConvOrConcat(kernelToFuseInto);

    bool requiresMixedPrecisionScratchPlaceholder = false;
    // No need for a placeholder in the in-place case as in 'mixed precision'
    // cases there the output buffer should have the correct dtype.
    if (!userIsInPlace) {
      requiresMixedPrecisionScratchPlaceholder =
          originalOuts.getResult(outputIdx).getType() !=
          userOuts.getResult(0).getType();
    }

    // Step 1: We unique the user kernel operands and fuse them into the src
    // kernel. The way we fuse these by operands is simply append them in the
    // src kernel's block argument list.

    DenseSet<Value> seenOperands;
    Block &userBlock = userToFuseIn.getBody().front();

    for (auto operand : userToFuseIn.getOperands())
      seenOperands.insert(operand);

    unsigned fusionOperandIdx = 0;
    for (auto [operand, blockArg] :
         llvm::zip(userToFuseIn.getOperands(), userBlock.getArguments())) {
      if (operand == originalResult) {
        fusionOperandIdx = blockArg.getArgNumber();
        continue;
      }

      if (seenOperands.contains(operand)) {
        Value newBlockArg = kernelToFuseInto.addOperand(operand);
        oldToNew.map(blockArg, newBlockArg);
      }
    }

    // update output type if needed
    originalResult.setType(newExpectedResult.getType());

    // Step 2: We clone all non-trivial operators in the user kernel into the
    // src kernel. Here, we ignore all the ParamDeclareRegionOp, which we will
    // take care of later in a more specific way.

    // Major mapper we use to clone all kinds of the operators in the user
    // kernel.
    OutputPlaceholdersOp newOuts;
    // Track the fusion MOGG ToKGENOp
    MOGG::ToKGENOp fusionMoggToKgen;

    // Create or extend an empty tensor op for the fusion block argument
    auto fusionBlockArg = userBlock.getArgument(fusionOperandIdx);

    // Set insertion point at the appropriate location - after
    // OutputPlaceholdersOp if one exists, otherwise at the top
    auto outputPlaceholdersOps =
        kernelToFuseInto.getOps<MOGG::OutputPlaceholdersOp>();
    if (!outputPlaceholdersOps.empty())
      builder.setInsertionPointAfter(*outputPlaceholdersOps.begin());
    else
      kernelToFuseInto.setInsertionPointToTop(builder);

    // Unconditionally create a new empty tensors op
    auto emptyTensorsOp = builder.create<MOGG::CreateEmptyTensorsOp>(
        kernelToFuseInto.getLoc(), fusionBlockArg.getType());
    Value emptyTensorForFusion = emptyTensorsOp.getResult(0);

    SmallVector<KGEN::ParamDeclareOp> newParamDeclOps;

    // Find all the ParamDeclRef that corresponds to staticTensorSpec attrs.
    DenseSet<StringAttr> tensorSpecAttrNames;
    for (auto toKgenOp : userToFuseIn.getOps<MOGG::ToKGENOp>()) {
      StringAttr tensorSpecName = toKgenOp.getStaticTensorSpecNameAttr();
      if (tensorSpecName)
        tensorSpecAttrNames.insert(tensorSpecName);
    }

    // Build the mapping from the lambda to tensor spec param decl that refers
    // to it. When we move a lambda inside another lambda, we also need to
    // move the corresponding tensor spec param decl. Otherwise, it will
    // complain that the lambda isn't defined (different scope).
    DenseMap<StringAttr, SmallVector<KGEN::ParamDeclareOp>>
        lambdaToParamDeclareOpMapping;
    auto findParamDeclMapping = [&](KGEN::ParamDeclareOp paramDeclOp) {
      mlir::AttrTypeWalker walker;
      walker.addWalk([&](KGEN::ParamDeclRefAttr refAttr) {
        if (isa<KGEN::FuncTypeGeneratorType>(refAttr.getType()))
          lambdaToParamDeclareOpMapping[refAttr.getName()].push_back(
              paramDeclOp);
      });
      walker.walk(paramDeclOp.getValue());
    };

    DenseSet<Operation *> skipCopy;
    for (auto &op : userToFuseIn.getOps()) {
      auto callOp = dyn_cast<KGEN::CallOp>(op);
      if (!callOp)
        continue;
      StringRef funcName =
          cast<mlir::FlatSymbolRefAttr>(callOp.getCalleeSymbol()).getValue();

      if (!funcName.starts_with(Utils::DPSTensorForEach) &&
          !funcName.starts_with(Utils::DPSTensorCopy))
        continue;

      auto res = MO::getRootedComponents(&op);
      skipCopy.insert(res.begin(), res.end());
    }

    for (auto &op : userToFuseIn.getOps()) {
      if (skipCopy.contains(&op))
        continue;
      if (auto toKgen = dyn_cast<MOGG::ToKGENOp>(op)) {
        if (auto blockArg = dyn_cast<mlir::BlockArgument>(toKgen.getInput())) {
          if (blockArg.getArgNumber() == fusionOperandIdx)
            oldToNew.map(blockArg, emptyTensorForFusion);
        }

        auto newToKgen = cast<MOGG::ToKGENOp>(builder.clone(op, oldToNew));

        // Store the newly created ToKGENOp that corresponds to the fusion block
        // arg
        if (auto blockArg = dyn_cast<mlir::BlockArgument>(toKgen.getInput())) {
          if (blockArg.getArgNumber() == fusionOperandIdx)
            fusionMoggToKgen = newToKgen;
        }

        oldToNew.map(toKgen.getResults(), newToKgen.getResults());
      } else if (isa<KGEN::POP::StackAllocationOp>(op)) {
        // We clone StackAllocationOp and track its mapping. Both
        // StackAllocationOp and ToKGENOp can produce outputs directly, feeding
        // the KGEN calls.
        Operation *newAlloc = builder.clone(op, oldToNew);
        oldToNew.map(op.getResults(), newAlloc->getResults());
      } else if (isa<MOGG::OutputPlaceholdersOp>(op)) {
        // OutputPlaceholdersOp is the first operator in the user kernel to
        // clone.
        SmallVector<Type> results(originalOuts->getResultTypes());
        results[outputIdx] = op.getResultTypes().front();
        newOuts = builder.create<OutputPlaceholdersOp>(op.getLoc(), results);
        oldToNew.map(op.getResult(0), newOuts.getResult(outputIdx));
      } else if (isa<KGEN::ParamDeclareOp>(op)) {
        auto newOp = cast<KGEN::ParamDeclareOp>(builder.clone(op));
        if (tensorSpecAttrNames.contains(newOp.getParamDecl().getName()))
          findParamDeclMapping(newOp);
        newParamDeclOps.push_back(newOp);
      } else if (isa<MOGG::OutputOp, KGEN::ParamDeclareRegionOp>(op)) {
        // Do nothing.
      } else {
        oldToNew.map(op.getResults(),
                     builder.clone(op, oldToNew)->getResults());
      }
    }

    if (!userIsInPlace) {
      ASSERT_STREAM(
          newOuts,
          "Failed to create output placeholders for non-in-place kernel "
          "fusion. "
          "Output placeholders are required for proper tensor allocation.");
    }
    ASSERT_STREAM(
        fusionMoggToKgen,
        "Failed to create ToKGENOp for fusion block argument. "
        "This is required for output lambda fusion to work correctly.");

    // Step 3: Clone all input lamba functions in the user kernel into the src
    // kernel. For the fusion input lambda, we treat it specially by nesting it
    // inside the output lambda of the src kernel.

    mlir::SymbolTable srcSymTable{kernelToFuseInto};
    mlir::SymbolTable userSymTable{userToFuseIn};

    // It is guarded that the input and output fusion lambda name exist.
    auto srcOutputLambdaOp =
        cast<KGEN::ParamDeclareRegionOp>(srcSymTable.lookup(outputLambdaName));
    StringAttr inputLambdaName =
        userToFuseIn.getInputFusionInterfaceAt(fusionOperandIdx);
    auto userInputLambdaOp =
        cast<KGEN::ParamDeclareRegionOp>(userSymTable.lookup(inputLambdaName));

    // Clone all input lambdas except for the special one (the fusion one).
    builder.setInsertionPoint(srcOutputLambdaOp);
    llvm::DenseSet<StringRef> lambdaSkipSet;

    StringAttr userOutputLambdaName =
        userToFuseIn.getOutputFusionInterfaceAt(0);

    KGEN::ParamDeclareRegionOp userOutputLambdaOp;
    if (userIsInPlace) {
      // Get the output lambda using the fusionBodyOutputLambda attribute
      StringAttr fusionBodyOutputLambdaAttr =
          cast<StringAttr>(userToFuseIn->getDiscardableAttr(
              KGEN::MOGGPreElab::kMOGGFusionBodyOutputLambda));
      ASSERT_STREAM(fusionBodyOutputLambdaAttr,
                    << KGEN::MOGGPreElab::kMOGGFusionBodyOutputLambda
                    << " attribute must be present on "
                       "elementwise kernel");

      userOutputLambdaOp = cast<KGEN::ParamDeclareRegionOp>(
          userSymTable.lookup(fusionBodyOutputLambdaAttr));
    } else {
      userOutputLambdaOp = cast<KGEN::ParamDeclareRegionOp>(
          userSymTable.lookup(userOutputLambdaName));
      lambdaSkipSet.insert(userOutputLambdaName);
    }

    for (auto attr : userToFuseIn.getInputFusionInterfaceAttr()) {
      StringAttr userInputLambdaNameAttr = cast<StringAttr>(attr);
      lambdaSkipSet.insert(userInputLambdaNameAttr);

      if (userInputLambdaNameAttr != inputLambdaName) {
        if (auto lambdaOp = userSymTable.lookup(userInputLambdaNameAttr))
          // For non-fusion input lambda, we don't remove its symbol attribute
          // so other passes can still use SymbolTable to visit them.
          builder.clone(*lambdaOp, oldToNew);
      }
    }

    // Clone and modify the fusion input lambda (the input lambda in the user
    // kernel). We put it inside the output lambda of the src kernel. We remove
    // the simd_load_internal inside it and pass the fused SIMD value directly.
    builder.setInsertionPointToStart(
        &srcOutputLambdaOp.getBodyRegion().front());

    auto newInputLambdaOp = cast<KGEN::ParamDeclareRegionOp>(
        builder.clone(*userInputLambdaOp.getOperation()));
    newInputLambdaOp->removeAttr(mlir::SymbolTable::getSymbolAttrName());

    // Move the related declare spec op if needed.
    if (auto paramDeclIt = lambdaToParamDeclareOpMapping.find(
            newInputLambdaOp.getParamDeclAttr().getName());
        paramDeclIt != lambdaToParamDeclareOpMapping.end()) {
      for (auto declOp : paramDeclIt->second) {
        declOp->moveAfter(newInputLambdaOp);
      }
    }

    Value originOutValue = srcOutputLambdaOp.getBodyRegion().getArgument(1);
    Type bindType = newInputLambdaOp.getFunctionType().getResult(0);

    KGEN::ReturnOp returnOp =
        *newInputLambdaOp.getOps<KGEN::ReturnOp>().begin();
    assert(returnOp);
    builder.setInsertionPoint(returnOp);

    KGEN::RebindOp rebind = builder.create<KGEN::RebindOp>(
        newInputLambdaOp->getLoc(), bindType, originOutValue);
    KGEN::CallOp simdLoadOp = *newInputLambdaOp.getOps<KGEN::CallOp>().begin();
    assert(simdLoadOp);
    simdLoadOp->replaceAllUsesWith(rebind);
    simdLoadOp->erase();

    // Step 4: We handle all the compute lambdas here. This includes the final
    // elementwise call and all the compute lambdas get fused into. For the
    // final elementwise lambda, we insert the COMPUTE_OUTLINED call op
    // directly. For the rest of the compute lambdas, we clone them into the
    // output lambda of the src kernel.
    StringAttr elementwiseLambdaName = userToFuseIn.getFusionBodyLambdaAttr();
    if (!userIsInPlace)
      lambdaSkipSet.insert(elementwiseLambdaName);

    // It is guaranteed that the user kernel is an elementwise op.
    auto elementwiseOp = userToFuseIn.findLambda(elementwiseLambdaName);

    // Assume the last call in each lambda is the store op

    KGEN::CallOp userSimdStoreOp;
    for (auto callOp : userOutputLambdaOp.getOps<KGEN::CallOp>())
      userSimdStoreOp = callOp;

    KGEN::CallOp srcSimdStoreOp;
    for (auto callOp : srcOutputLambdaOp.getOps<KGEN::CallOp>())
      srcSimdStoreOp = callOp;

    // Assert that the last operator are the store function we expect.
    [[maybe_unused]] auto simdStoreFunc = analysis.getRegisteredMojoKernelFor(
        Utils::DPSManagedTensorSliceStoreSIMD);
    if (!userIsInPlace) {
      ASSERT_STREAM(
          srcSimdStoreOp.getCallee().getSymbol().getLeafReference() ==
              simdStoreFunc.getSymNameAttr(),
          << "Expected last call of output lambda of src to be "
          << simdStoreFunc.getSymNameAttr() << " but got "
          << srcSimdStoreOp.getCallee().getSymbol().getLeafReference());
      ASSERT_STREAM(
          userSimdStoreOp.getCallee().getSymbol().getLeafReference() ==
              simdStoreFunc.getSymNameAttr(),
          << "Expected last call of output lambda of user to be "
          << simdStoreFunc.getSymNameAttr() << " but got "
          << userSimdStoreOp.getCallee().getSymbol().getLeafReference());
    }

    // Remap to_kgen to the new one, which is corresponding to the output
    // placeholder in the src kernel.
    for (auto toKgenOp : userToFuseIn.getOps<MOGG::ToKGENOp>()) {
      if (toKgenOp.getInput() == userBlock.getArgument(fusionOperandIdx)) {
        oldToNew.map(toKgenOp->getResult(0), fusionMoggToKgen->getResult(0));
      }
    }

    // Remap the SIMD indices of the simd_store_internal in the output lambda.
    if (!userIsInPlace)
      oldToNew.map(userSimdStoreOp.getOperand(1), srcSimdStoreOp.getOperand(1));

    // Remap SIMD indices for the COMPUTE_OUTLINED call.
    // The body should only have a single argument (the SIMD indices).
    assert(elementwiseOp.getBody()->getNumArguments() == 1);
    constexpr unsigned simdStoreIndicesOperandIdx = 1;
    if (!userIsInPlace) {
      oldToNew.map(elementwiseOp.getBody()->getArgument(0),
                   srcSimdStoreOp.getOperand(simdStoreIndicesOperandIdx));
    }

    // Clone all the elementwise lambdas except the final one.
    builder.setInsertionPointAfter(newInputLambdaOp);
    for (auto declRegion : userToFuseIn.getOps<KGEN::ParamDeclareRegionOp>()) {
      StringAttr regionName = declRegion.getParamDeclAttr().getName();
      if (!lambdaSkipSet.contains(regionName)) {
        auto newRegion = cast<KGEN::ParamDeclareRegionOp>(
            builder.clone(*declRegion.getOperation(), oldToNew));
        newRegion->removeAttr(mlir::SymbolTable::getSymbolAttrName());

        // Move the related declare spec op if needed.
        if (auto paramDeclIt = lambdaToParamDeclareOpMapping.find(
                newRegion.getParamDeclAttr().getName());
            paramDeclIt != lambdaToParamDeclareOpMapping.end()) {
          for (auto declOp : paramDeclIt->second) {
            declOp->moveAfter(newInputLambdaOp);
          }
        }
      }
    }

    builder.setInsertionPoint(
        builder.getInsertionPoint()->getBlock()->getTerminator());

    if (!userIsInPlace) {
      // Insert the elementwise body for the final elementwise lambda. We
      // clone the user kernel's SIMD store into the src kernel to account for
      // the mixed precision case.
      SmallVector<Operation *> clonedElementwiseBodyOps;
      for (Operation &elemwOp : elementwiseOp.getBody()->without_terminator()) {
        auto clonedOp = builder.clone(elemwOp, oldToNew);
        clonedElementwiseBodyOps.push_back(clonedOp);
      }

      Operation *elemwiseBodyReturnOp =
          elementwiseOp.getBody()->getTerminator();
      assert(elemwiseBodyReturnOp->getNumOperands() == 1);
      constexpr unsigned simdStoreValuesOperandIdx = 2;
      oldToNew.map(userSimdStoreOp.getOperand(simdStoreValuesOperandIdx),
                   oldToNew.lookup(elemwiseBodyReturnOp->getOperand(0)));
      clonedElementwiseBodyOps.push_back(
          builder.clone(*userSimdStoreOp.getOperation(), oldToNew));

      // Step 5: Remap all the template parameters of the two new calls inserted
      // (COMPUTE_OUTLINED and SIMD store).
      DenseMap<KGEN::ParamDeclRefAttr, TypedAttr> paramRemaps;
      auto oldElementwiseParams = elementwiseOp.getInputParams();
      auto oldStoreParams = userOutputLambdaOp.getInputParams();
      auto srcOutputLambdaParams = srcOutputLambdaOp.getInputParams();

      auto constructParamRemapFunc = [&](ArrayRef<KGEN::ParamDeclAttr> params) {
        for (auto [oldParam, newParam] :
             llvm::zip(params, srcOutputLambdaParams)) {
          auto oldDeclRef = KGEN::ParamDeclRefAttr::get(oldParam.getName(),
                                                        oldParam.getType());
          auto newDeclRef = KGEN::ParamDeclRefAttr::get(newParam.getName(),
                                                        newParam.getType());
          paramRemaps[oldDeclRef] = newDeclRef;
        }
      };

      constructParamRemapFunc(oldElementwiseParams);
      constructParamRemapFunc(oldStoreParams);

      mlir::AttrTypeReplacer walker;
      walker.addReplacement(
          [&](KGEN::ParamDeclRefAttr attr) -> std::optional<TypedAttr> {
            auto itr = paramRemaps.find(attr);
            if (itr != paramRemaps.end())
              return itr->second;
            return std::nullopt;
          });

      for (Operation *newElemwOp : clonedElementwiseBodyOps) {
        walker.recursivelyReplaceElementsIn(newElemwOp,
                                            /*replaceAttrs=*/true,
                                            /*replaceLocs=*/false,
                                            /*replaceTypes=*/true);
      }
    } else {
      // If we have an in-place kernel we've cloned ALL the fusion body
      // elementwise lambdas (including the final one as opposed to the standard
      // case)
      // We've also cloned the lambda that's actually responsible for calling
      // that final elmewise body lambda and writing to the actual outpuut
      // (userOutputLambdaOp) now we simply have to synthesize a kgen.call_param
      // to the userOutputLambdaOp with the single index parameter of that bound
      // to the first parameter of the enclosing region.

      auto simdWidthParam = srcOutputLambdaOp.getInputParams().front();

      auto simdWidthParamRef = KGEN::ParamDeclRefAttr::get(
          simdWidthParam.getName(), simdWidthParam.getType());

      auto generatorAttrRef =
          KGEN::ParamDeclRefAttr::get(userOutputLambdaOp.getParamDeclAttr());

      auto bindParamAttr = KGEN::BindParamsAttr::get(
          generatorAttrRef, ArrayRef<TypedAttr>{simdWidthParamRef});

      auto noneType = KGEN::NoneType::get(builder.getContext());

      builder.create<KGEN::CallParamOp>(
          userOutputLambdaOp.getLoc(), noneType, bindParamAttr,
          ArrayRef<Value>{srcOutputLambdaOp.getBodyRegion().getArgument(0)});
    }

    // Step 6: Finally, we clean up the kernel metadata, eliminate the user
    // kernel, and handle the mixed precision case.

    // Disable future output fusion.
    kernelToFuseInto.removeOutputFusionInterfaceAttr();

    // Update the lambdas_have_fusion parameter for the extensibility kernels
    setKGENKernelParamValueIfPresent(
        kernelToFuseInto, analysis,
        KGEN::MOGGPreElab::kMOGGLambdasHaveFusionLabel,
        BoolAttr::get(&getContext(), true));

    // Get rid of the unused declarations for some of the removed lambdas.
    clearUnusedParamDecls(kernelToFuseInto, newParamDeclOps);

    // Matmul with mixed precision (bad case): Matmul kernels actually take in
    // an output buffer despite having a lambda. Therefore, in mixed
    // precision, we need another buffer to do the accumulation. This results
    // in the kernel op having two return values to trigger allocations in
    // later passes.
    if (isMatmulOrConvOrConcat && requiresMixedPrecisionScratchPlaceholder) {
      ASSERT_STREAM(outputIdx == 0,
                    "matmul or conv should only have one result.");
      auto loc = kernelToFuseInto.getLoc();

      builder.setInsertionPoint(originalOuts);

      auto newPlaceholders = builder.create<MOGG::OutputPlaceholdersOp>(
          loc, ArrayRef<Type>{userOuts->getResult(0).getType()});

      auto scratchPlaceholder = builder.create<MOGG::ScratchPlaceholdersOp>(
          loc, originalOuts->getResult(outputIdx).getType());

      MOGG::OutputOp originOutputOp = kernelToFuseInto.getOutputOp();
      builder.setInsertionPoint(originOutputOp);
      builder.create<MOGG::OutputOp>(loc, newPlaceholders->getResults());

      newOuts.getResult(outputIdx).replaceAllUsesWith(
          newPlaceholders.getResult(0));
      originalOuts.getResult(outputIdx).replaceAllUsesWith(
          scratchPlaceholder.getResult(0));

      originOutputOp.erase();
      newOuts->erase();
      originalOuts.erase();
      srcSimdStoreOp.erase();

      builder.setInsertionPoint(kernelToFuseInto);
      auto newDstKernel = builder.create<MOGG::KernelOp>(
          loc,
          /*results=*/newPlaceholders.getResultTypes(),
          /*operands=*/kernelToFuseInto.getOperands(),
          /*outputParamDecls=*/kernelToFuseInto.getOutputParamDeclsAttr(),
          /*inputFusionInterface=*/
          kernelToFuseInto.getInputFusionInterfaceAttr(),
          /*outputFusionInterface=*/
          kernelToFuseInto.getOutputFusionInterfaceAttr(),
          /*fusionBodyLambda=*/kernelToFuseInto.getFusionBodyLambdaAttr(),
          /*isViewOp=*/kernelToFuseInto.getIsViewOpAttr(),
          /*summary=*/
          getSummaryString({kernelToFuseInto, userToFuseIn}, FusionType::HARD),
          /*isAsync=*/kernelToFuseInto.getIsAsyncAttr(),
          /*computationallyTrivial=*/
          kernelToFuseInto.getComputationallyTrivialAttr(),
          /*allocIndices=*/ArrayAttr{},
          /*hasReadEffects*/ kernelToFuseInto.getHasReadEffect() ||
              userToFuseIn.getHasReadEffect(),
          /*hasWriteEffects*/ kernelToFuseInto.getHasWriteEffect() ||
              userToFuseIn.getHasWriteEffect(),
          /*device*/ kernelToFuseInto.getDeviceRef());
      newDstKernel.getBody().takeBody(kernelToFuseInto.getBody());

      userToFuseIn.getResult(0).replaceAllUsesWith(newDstKernel.getResult(0));
      opsToErase.push_back(userToFuseIn);
      opsToErase.push_back(kernelToFuseInto);

      return newDstKernel;
    }

    kernelToFuseInto.setSummaryAttr(
        getSummaryString({kernelToFuseInto, userToFuseIn}, FusionType::HARD));

    // For in place consumer kernels we need to create a scratch buffer for
    // intermediate results and erase the old output from the output placeholder
    // op
    if (userIsInPlace) {
      // Create the scratch placeholder op.
      builder.setInsertionPointAfter(originalOuts);
      auto scratchPlaceholder = builder.create<MOGG::ScratchPlaceholdersOp>(
          kernelToFuseInto.getLoc(), originalOuts->getResultTypes());
      originalOuts.replaceAllUsesWith(scratchPlaceholder.getResults());

      // Create the output chain for the kernel
      auto outChain = builder.create<MOGG::CreateReadyChainOp>(
          kernelToFuseInto.getLoc(), MO::ChainType::get(ctx));

      auto outputOp = kernelToFuseInto.getOutputOp();
      builder.setInsertionPoint(outputOp);

      builder.create<MOGG::OutputOp>(kernelToFuseInto.getLoc(),
                                     outChain.getResult());
      outputOp.erase();

      kernelToFuseInto.setHasReadEffect(kernelToFuseInto.getHasReadEffect() ||
                                        userToFuseIn.getHasReadEffect());

      kernelToFuseInto.setHasWriteEffect(kernelToFuseInto.getHasWriteEffect() ||
                                         userToFuseIn.getHasWriteEffect());

    } else {
      // If the user is not in place, we just replace the original output op's
      // inputs with the new outputs.
      originalOuts.replaceAllUsesWith(newOuts);
    }

    originalOuts.erase();
    srcSimdStoreOp.erase();

    userToFuseIn->getResult(0).replaceAllUsesWith(
        kernelToFuseInto->getResult(outputIdx));
    opsToErase.push_back(userToFuseIn);

    // Merge any CreateEmptyTensorsOp instances at the end
    llvm::SmallVector<MOGG::CreateEmptyTensorsOp> emptyTensorOps;
    for (auto op : kernelToFuseInto.getOps<MOGG::CreateEmptyTensorsOp>())
      emptyTensorOps.push_back(op);

    if (emptyTensorOps.size() > 1) {
      // Collect all result types
      SmallVector<Type> allTypes;
      for (auto op : emptyTensorOps) {
        allTypes.append(op->getResultTypes().begin(),
                        op->getResultTypes().end());
      }

      // Create a new merged operation
      auto outputPlaceholdersOps =
          kernelToFuseInto.getOps<MOGG::OutputPlaceholdersOp>();
      if (!outputPlaceholdersOps.empty())
        builder.setInsertionPointAfter(*outputPlaceholdersOps.begin());
      else
        kernelToFuseInto.setInsertionPointToTop(builder);

      auto mergedOp = builder.create<MOGG::CreateEmptyTensorsOp>(
          kernelToFuseInto.getLoc(), allTypes);

      // Replace all uses
      unsigned resultIndex = 0;
      for (auto op : emptyTensorOps) {
        for (auto [i, result] : llvm::enumerate(op.getResults()))
          result.replaceAllUsesWith(mergedOp.getResult(resultIndex++));
        op.erase();
      }
    }

    // This is hacky as all hell, but since this particular lowering path will
    // go away soon I don't feel *too* bad about it.
    //
    // The issue is that if the producer kernel has chains earlier will have it
    // end up with two chain operand and two chain results.
    // The logic here just re-writes the kernel so that there is just one chain
    // operand/result and rewires chains as appropriate.
    if (maybeSrcInChain && maybeSrcOutChain && userIsInPlace) {
      ASSERT_STREAM(
          kernelToFuseInto->getNumResults() == 2,
          << "Expected kernelToFuseInto to have exactly 2 results, but got "
          << kernelToFuseInto->getNumResults());
      ASSERT_STREAM(
          isa<MO::ChainType>(kernelToFuseInto->getResult(0).getType()) &&
              isa<MO::ChainType>(kernelToFuseInto->getResult(1).getType()),
          << "Both results of kernelToFuseInto should be of chain type");

      MO::ChainType chainType = MO::ChainType::get(ctx);
      auto newResults = SmallVector<Type>{chainType};

      auto newOperands =
          SmallVector<Value>(kernelToFuseInto.getOperands().drop_back());

      builder.setInsertionPoint(kernelToFuseInto);
      auto newKernelOp = builder.create<MOGG::KernelOp>(
          kernelToFuseInto.getLoc(),
          /*results=*/newResults,
          /*operands=*/newOperands,
          /*outputParamDecls=*/kernelToFuseInto.getOutputParamDeclsAttr(),
          /*inputFusionInterface=*/
          kernelToFuseInto.getInputFusionInterfaceAttr(),
          /*outputFusionInterface=*/
          kernelToFuseInto.getOutputFusionInterfaceAttr(),
          /*fusionBodyLambda=*/kernelToFuseInto.getFusionBodyLambdaAttr(),
          /*isViewOp=*/kernelToFuseInto.getIsViewOpAttr(),
          /*summary=*/kernelToFuseInto.getSummaryAttr(),
          /*isAsync=*/kernelToFuseInto.getIsAsyncAttr(),
          /*computationallyTrivial=*/
          kernelToFuseInto.getComputationallyTrivialAttr(),
          /*allocIndices=*/kernelToFuseInto.getAllocIndicesAttr(),
          /*hasReadEffects=*/kernelToFuseInto.getHasReadEffect(),
          /*hasWriteEffects=*/kernelToFuseInto.getHasWriteEffect(),
          /*device=*/kernelToFuseInto.getDeviceRef());

      auto &oldBody = kernelToFuseInto.getBody();
      auto &oldBlock = oldBody.front();

      auto lastArg = oldBlock.getArgument(oldBlock.getNumArguments() - 1);
      ASSERT_STREAM(isa<MO::ChainType>(lastArg.getType()),
                    << "Last block is expected to be a chain type");
      ASSERT_STREAM(lastArg.use_empty(),
                    << "Last block argument (chain) must have no uses before "
                       "dropping");

      ASSERT_STREAM(oldBlock.getNumArguments() >= 2,
                    << "Block should have at least 2 arguments");
      auto secondLastArg = oldBlock.getArgument(oldBlock.getNumArguments() - 2);
      ASSERT_STREAM(
          isa<MO::ChainType>(secondLastArg.getType()),
          << "Second-to-last block argument is expected to be a chain type");

      // Drop the last block argument before moving the body (the extraneous
      // chain operand)
      oldBlock.eraseArgument(oldBlock.getNumArguments() - 1);
      newKernelOp.getBody().takeBody(oldBody);

      // Route the output chain of newKernelOp to all consumers of
      // userToFuseIn's output chain
      Value newOutChain = newKernelOp.getResult(0);
      kernelToFuseInto.getResult(0).replaceAllUsesWith(newOutChain);

      // Replace the in chain of newKernel op with maybeSrcInChain
      Value inChain = *maybeSrcInChain;
      newKernelOp->setOperand(newKernelOp->getNumOperands() - 1, inChain);

      // The way the code before this block is written, leaves an output chain
      // looping back to kernelToFuseInto. Break the cycle so we can erase
      // kernelToFuseInto later.
      kernelToFuseInto->setOperand(kernelToFuseInto->getNumOperands() - 2,
                                   newKernelOp.getInChain());
      opsToErase.push_back(kernelToFuseInto);

      kernelToFuseInto = newKernelOp;
    }

    return kernelToFuseInto;
  }

  void runOnOperation() override {
    MO::GraphOp moGraph = getOperation();

    if (MOGG::useExperimentalMOGGPath()) {
      fuseExpKernelOutLambda(moGraph);
      fuseExpKernelInLambda(moGraph);
      return;
    }

    auto &loadAnalysis = getAnalysis<MOGG::LoadMojoAnalysisPass>();
    auto res = loadAnalysis.initIfNeeded(&getContext(), kernelLibraryPaths);
    if (res.isError()) {
      mlir::emitError(moGraph->getLoc(),
                      "Unable to initialize MojoLibraryAnalysis");
      signalPassFailure();
    }
    MojoLibraryAnalysis *analysis = res.get();

    SmallVector<MOGG::KernelOp> kernelOpsToErase;

    moGraph.walk([&](MOGG::KernelOp kernelToFuseInto) {
      [[maybe_unused]] DeviceRefAttr beforeDevice =
          MO::getDeviceOrDefault(kernelToFuseInto);
      std::optional<ArrayAttr> outputLambdas =
          kernelToFuseInto.getOutputFusionInterface();
      // Try fuse into the output lambda if we have a fusion interface for that
      // output.
      if (outputLambdas.has_value()) {
        ArrayAttr lambdas = *outputLambdas;
        llvm::BitVector fusableOuts =
            canDoOutputFusion(kernelToFuseInto, lambdas);
        // Fuse into the output lambdas if possible.
        for (unsigned idx : fusableOuts.set_bits()) {
          StringAttr outputLambdaName = cast<StringAttr>(lambdas[idx]);
          kernelToFuseInto =
              fuseIntoOutputLambdaKGEN(kernelToFuseInto, outputLambdaName, idx,
                                       kernelOpsToErase, *analysis);
        }
      }
      ASSERT_STREAM(beforeDevice == MO::getDeviceOrDefault(kernelToFuseInto),
                    << "Must be same device after output lambda fusion");

      auto inputLambdas = kernelToFuseInto.getInputFusionInterface();
      if (!inputLambdas) {
        return;
      }

      uint64_t numOperands = kernelToFuseInto.getNumOperands();
      for (int64_t inputIdx = numOperands - 1; inputIdx >= 0; --inputIdx) {
        if (canDoInputFusion(kernelToFuseInto,
                             kernelToFuseInto.getInputFusionInterfaceAttr(),
                             inputIdx)) {
          fuseIntoInputLambdaKGEN(
              kernelToFuseInto, kernelToFuseInto.getInputFusionInterfaceAttr(),
              inputIdx, kernelOpsToErase, *analysis);
        }
      }
      ASSERT_STREAM(beforeDevice == MO::getDeviceOrDefault(kernelToFuseInto),
                    << "Must be same device after input lambda fusion");
    });

    for (KernelOp kernelOp : kernelOpsToErase)
      kernelOp.erase();

    // FIXME(GEX-865): This pass is generating invalid IR. Fix it up here.
    getOperation().walk([](KGEN::CallOp call) {
      OpBuilder b(call);
      for (auto [arg, opOperand, type] :
           llvm::zip(call.getOperands(), call->getOpOperands(),
                     call.getCallee().getType().getBody().getArguments())) {
        if (arg.getType() != type) {
          Value rebound = b.create<KGEN::RebindOp>(call.getLoc(), type, arg);
          opOperand.set(rebound);
        }
      }
    });
  }
};

} // namespace

} // namespace M::MOGG
