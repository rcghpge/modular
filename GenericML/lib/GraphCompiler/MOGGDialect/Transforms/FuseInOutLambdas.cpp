//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoTargetFunctions.h"
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

struct FuseInOutLambdas : public impl::FuseInOutLambdasBase<FuseInOutLambdas> {
  using FuseInOutLambdasBase::FuseInOutLambdasBase;

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

    for (auto &op : inputKernel.getOps()) {
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
      } else if (auto callop = dyn_cast<KGEN::CallOp>(op)) {
        StringRef funcName =
            cast<mlir::FlatSymbolRefAttr>(callop.getCalleeSymbol()).getValue();
        // Clone every function call except the one has the `elementwise`
        // decorator.
        if (!funcName.starts_with(Utils::DPSTensorForEach)) {
          oldToNew.map(op.getResults(),
                       builder.clone(op, oldToNew)->getResults());
        }
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
    kernelToFuseInto.setKGENKernelParamValueIfPresent(
        analysis, KGEN::MOGGPreElab::kMOGGLambdasHaveFusionLabel,
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
  bool canDoOutputFusion(MOGG::KernelOp kernel, ArrayAttr fusionInterface) {
    // Unreachable: Ambigious what we should actually fuse. Previous passes
    // should have maintained 1:1 relationship of output to output interfaces.
    if (fusionInterface.size() != kernel.getNumResults())
      return false;

    // Restrictive guard for now.
    if (kernel.getNumResults() != 1)
      return false;

    // For in-place allocated kernel, we disable for now.
    if (kernel.isInPlaceAlloc())
      return false;

    // Should be unreachable sanity check. Ensure this is not an elementwise
    // kernel (as if it where after fusion we would trigger more fusions).
    auto isElemwise = kernel.getFusionBodyLambda();
    if (isElemwise)
      return false;

    // Output can only have one user and that user must be a fusable elementwise
    // kernel.
    Value result = kernel.getResult(0);
    if (llvm::range_size(result.getUsers()) > 1)
      return false;

    // Check it's a kernel.
    MOGG::KernelOp user = dyn_cast<MOGG::KernelOp>(*result.user_begin());
    if (!user)
      return false;

    // Only fuse within the same block and not across, i.e, control flow.
    if (kernel->getBlock() != user->getBlock())
      return false;

    // Check that it has a single output.
    if (user->getNumResults() != 1)
      return false;

    // Check the kernel is a fusable elementwise.
    if (!user.getFusionBodyLambda())
      return false;

    // Kernel must expose input fusion contract.
    std::optional<ArrayAttr> inputInterfaces = user.getInputFusionInterface();
    if (!inputInterfaces.has_value())
      return false;

    // It must expose a valid input lambda attribute for every operand which
    // uses this.
    for (auto [operand, attr] : llvm::zip(
             user.getOperands(), inputInterfaces->getAsRange<StringAttr>())) {
      if (operand == result && attr.strref().empty())
        return false;
    }

    // Check the output lambda fusion hook.
    if (cast<StringAttr>(fusionInterface[0]).strref().empty())
      return false;

    // Must have the same device
    if (kernel.getDeviceRef() != user.getDeviceRef())
      return false;

    return true;
  }

  // Fuse the user of `kernelToFuseInto` into the src kernel with KGEN.
  KernelOp fuseIntoOutputLambdaKGEN(KernelOp kernelToFuseInto,
                                    StringAttr outputLambdaName,
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

    MLIRContext *ctx = kernelToFuseInto.getContext();

    OpBuilder builder{ctx};

    // Maps kernel's old operand to new operand.
    IRMapping oldToNew;

    // Guarded to only fuse 1 output kernels.
    Value originalResult = kernelToFuseInto.getResult(0);

    // We've already guarded that we only have one user.
    MOGG::KernelOp userToFuseIn =
        cast<MOGG::KernelOp>(*originalResult.user_begin());

    Value newExpectedResult = userToFuseIn->getResult(0);

    OutputPlaceholdersOp originalOuts = kernelToFuseInto.getOutputPlaceholder();
    OutputPlaceholdersOp userOuts = userToFuseIn.getOutputPlaceholder();

    const bool isMatmulOrConvOrConcat =
        containsMatmulOrConvOrConcat(kernelToFuseInto);

    const bool isMixedPrecision =
        originalOuts.getResult(0).getType() != userOuts.getResult(0).getType();

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
    MOGG::ToKGENOp newOutsToKgenOp;
    OutputPlaceholdersOp newOuts;

    SmallVector<KGEN::ParamDeclareOp> newParamDeclOps;

    // Find all the ParamDeclRef that corresponds to staticTensorSpec attrs.
    DenseSet<StringAttr> tensorSpecAttrNames;
    for (auto toKgenOp : userToFuseIn.getOps<MOGG::ToKGENOp>()) {
      StringAttr tensorSpecName = toKgenOp.getStaticTensorSpecNameAttr();
      if (tensorSpecName)
        tensorSpecAttrNames.insert(tensorSpecName);
    }

    // Build the mapping from the lambda to tensor spec param decl that refers
    // to it. When we move a lambda inside another lambda, we also need to move
    // the corresponding tensor spec param decl. Otherwise, it will complain
    // that the lambda isn't defined (different scope).
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

    kernelToFuseInto.setInsertionPointToTop(builder);
    for (auto &op : userToFuseIn.getOps()) {
      if (auto toKgen = dyn_cast<MOGG::ToKGENOp>(op)) {
        // We clone ToKGENOp except for the one related to the fusion hook.
        if (auto blockArg = dyn_cast<mlir::BlockArgument>(toKgen.getInput()))
          if (blockArg.getArgNumber() == fusionOperandIdx)
            continue;

        auto newToKgen = cast<MOGG::ToKGENOp>(builder.clone(op, oldToNew));
        // When we visit ToKGENOp, newOuts placeholder has been assigned.
        if (newToKgen.getInput() == newOuts.getResult(0))
          newOutsToKgenOp = newToKgen;

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
        newOuts = cast<OutputPlaceholdersOp>(builder.clone(op));
        oldToNew.map(op.getResult(0), newOuts.getResult(0));
      } else if (isa<KGEN::ParamDeclareOp>(op)) {
        auto newOp = cast<KGEN::ParamDeclareOp>(builder.clone(op));
        if (tensorSpecAttrNames.contains(newOp.getParamDecl().getName()))
          findParamDeclMapping(newOp);
        newParamDeclOps.push_back(newOp);
      } else if (isa<MOGG::OutputOp>(op) ||
                 isa<KGEN::ParamDeclareRegionOp>(op)) {
        // Do nothing.
      } else if (auto callop = dyn_cast<KGEN::CallOp>(op)) {
        StringRef funcName =
            cast<mlir::FlatSymbolRefAttr>(callop.getCalleeSymbol()).getValue();
        // After Tensor APIs becomes the de-factor standard APIs (builtin
        // library), we will use MojoLibararyAnalysis to check this. Clone every
        // function call except the one has the `elementwise` decorator.
        if (!funcName.starts_with(Utils::DPSTensorForEach)) {
          oldToNew.map(op.getResults(),
                       builder.clone(op, oldToNew)->getResults());
        }
      } else {
        oldToNew.map(op.getResults(),
                     builder.clone(op, oldToNew)->getResults());
      }
    }
    assert(newOutsToKgenOp);
    assert(newOuts);

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
    lambdaSkipSet.insert(userOutputLambdaName);

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
    lambdaSkipSet.insert(elementwiseLambdaName);

    // It is guaranteed that the user kernel is an elementwise op.
    auto elementwiseOp = userToFuseIn.findLambda(elementwiseLambdaName);

    auto userOutputLambdaOp = cast<KGEN::ParamDeclareRegionOp>(
        userSymTable.lookup(userOutputLambdaName));

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
    ASSERT_STREAM(srcSimdStoreOp.getCallee().getSymbol().getLeafReference() ==
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

    // Remap to_kgen to the new one, which is corresponding to the output
    // placeholder in the src kernel.
    for (auto toKgenOp : userToFuseIn.getOps<MOGG::ToKGENOp>()) {
      if (toKgenOp.getInput() == userBlock.getArgument(fusionOperandIdx))
        oldToNew.map(toKgenOp->getResult(0), newOutsToKgenOp->getResult(0));
    }

    // Remap the SIMD indices of the simd_store_internal in the output lambda.
    oldToNew.map(userSimdStoreOp.getOperand(1), srcSimdStoreOp.getOperand(1));

    // Remap SIMD indices for the COMPUTE_OUTLINED call.
    // The body should only have a single argument (the SIMD indices).
    assert(elementwiseOp.getBody()->getNumArguments() == 1);
    constexpr unsigned simdStoreIndicesOperandIdx = 1;
    oldToNew.map(elementwiseOp.getBody()->getArgument(0),
                 srcSimdStoreOp.getOperand(simdStoreIndicesOperandIdx));

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

    // Insert the elementwise body for the final elementwise lambda. We
    // clone the user kernel's SIMD store into the src kernel to account for the
    // mixed precision case.
    SmallVector<Operation *> clonedElementwiseBodyOps;
    for (Operation &elemwOp : elementwiseOp.getBody()->without_terminator()) {
      clonedElementwiseBodyOps.push_back(builder.clone(elemwOp, oldToNew));
    }

    Operation *elemwiseBodyReturnOp = elementwiseOp.getBody()->getTerminator();
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
        auto oldDeclRef =
            KGEN::ParamDeclRefAttr::get(oldParam.getName(), oldParam.getType());
        auto newDeclRef =
            KGEN::ParamDeclRefAttr::get(newParam.getName(), newParam.getType());
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

    // Step 6: Finally, we clean up the kernel metadata, eliminate the user
    // kernel, and handle the mixed precision case.

    // Disable future output fusion.
    kernelToFuseInto.removeOutputFusionInterfaceAttr();

    // Update the lambdas_have_fusion parameter for the extensibility kernels
    kernelToFuseInto.setKGENKernelParamValueIfPresent(
        analysis, KGEN::MOGGPreElab::kMOGGLambdasHaveFusionLabel,
        BoolAttr::get(&getContext(), true));

    // Get rid of the unused declarations for some of the removed lambdas.
    clearUnusedParamDecls(kernelToFuseInto, newParamDeclOps);

    // Matmul with mixed precision (bad case): Matmul kernels actually take in
    // an output buffer despite having a lambda. Therefore, in mixed
    // precision, we need another buffer to do the accumulation. This results
    // in the kernel op having two return values to trigger allocations in
    // later passes.
    if (isMatmulOrConvOrConcat && isMixedPrecision) {
      auto loc = kernelToFuseInto.getLoc();

      builder.setInsertionPoint(originalOuts);
      auto combinedPlaceholders = builder.create<MOGG::OutputPlaceholdersOp>(
          loc, ArrayRef<Type>{userOuts->getResult(0).getType(),
                              originalOuts->getResult(0).getType()});

      MOGG::OutputOp originOutputOp = kernelToFuseInto.getOutputOp();
      builder.setInsertionPoint(originOutputOp);
      builder.create<MOGG::OutputOp>(loc, combinedPlaceholders->getResults());

      newOuts.getResult(0).replaceAllUsesWith(
          combinedPlaceholders.getResult(0));
      originalOuts.getResult(0).replaceAllUsesWith(
          combinedPlaceholders.getResult(1));

      originOutputOp.erase();
      newOuts->erase();
      originalOuts.erase();
      srcSimdStoreOp.erase();

      builder.setInsertionPoint(kernelToFuseInto);
      auto newDstKernel = builder.create<MOGG::KernelOp>(
          loc,
          /*results=*/combinedPlaceholders.getResultTypes(),
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

    originalOuts->replaceAllUsesWith(newOuts);
    srcSimdStoreOp.erase();
    originalOuts.erase();

    userToFuseIn.replaceAllUsesWith(kernelToFuseInto);
    opsToErase.push_back(userToFuseIn);

    return kernelToFuseInto;
  }

  void runOnOperation() override {
    MO::GraphOp moGraph = getOperation();

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

        // Fuse into the output lambdas if possible.
        if (canDoOutputFusion(kernelToFuseInto, lambdas)) {
          StringAttr outputLambdaName = cast<StringAttr>(lambdas[0]);
          kernelToFuseInto = fuseIntoOutputLambdaKGEN(
              kernelToFuseInto, outputLambdaName, kernelOpsToErase, *analysis);
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
