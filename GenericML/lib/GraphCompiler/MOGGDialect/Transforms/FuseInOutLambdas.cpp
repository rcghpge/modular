//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoTargetFunctions.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/AssertStream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEINOUTLAMBDAS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// TODO(#22349): Remove checks when matmul/conv kernels respect the fusion
// contract.
static bool containsMatmulOrConvolution(MOGG::KernelOp kernel) {
  if (kernel.getIsKGEN().value_or(false)) {
    StringRef summary = kernel.getSummaryAttr().strref();
    if (summary.contains(MO::MatmulOp::getOperationName()) ||
        summary.contains(MO::BatchMatmulOp::getOperationName()) ||
        summary.contains(MO::ConvOp::getOperationName()) ||
        summary.contains(MO::ConvTransposeOp::getOperationName()))
      return true;
  } else {
    for (auto call : kernel.getOps<MOGG::CallOp>()) {
      auto target = call.getMojoKernelTarget();
      if (target == MO::MatmulOp::getOperationName() ||
          target == MO::BatchMatmulOp::getOperationName() ||
          target == MO::ConvOp::getOperationName() ||
          target == MO::ConvTransposeOp::getOperationName())
        return true;
    }
  }
  return false;
}

struct FuseInOutLambdas : public impl::FuseInOutLambdasBase<FuseInOutLambdas> {
  using FuseInOutLambdasBase::FuseInOutLambdasBase;

  bool canDoInputFusion(MOGG::KernelOp kernel, ArrayAttr fusionInterfaces,
                        uint64_t inputIdx) {
    // Disable fusion on experimental new KGEN in MOGG for now.
    if (kernel.getIsKGEN().value_or(false))
      return false;

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

    if (parentKernel.getIsKGEN() != kernel.getIsKGEN())
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

  void fuseIntoInputLambda(MOGG::KernelOp dstKernel, ArrayAttr fusionInterfaces,
                           uint64_t inputIdx,
                           SmallVectorImpl<KernelOp> &opsToErase) {
    MLIRContext *ctx = dstKernel.getContext();

    // We've already guarded that we only have one user. The kernel is cloned in
    // case it is needed elsewhere.
    auto srcKernel =
        cast<MOGG::KernelOp>(dstKernel.getOperand(inputIdx).getDefiningOp())
            .clone();

    Block &dstKernelBody = dstKernel.getBody().front();
    Block &srcKernelBody = srcKernel.getBody().front();

    auto fusionArgument = dstKernelBody.getArgument(inputIdx);

    // Remap references to the old outputs to the new values.
    IRMapping oldToNew;
    oldToNew.map(srcKernel.getOutputPlaceholder()->getResult(0),
                 fusionArgument);
    for (auto [operand, blockArg] :
         llvm::zip(srcKernel.getOperands(), srcKernelBody.getArguments())) {
      Value newBlockArg = dstKernel.addOperand(operand);
      oldToNew.map(blockArg, newBlockArg);
    }

    // Inline all the lambdas in the op we are pulling in.
    inlineLambdasInKernel(srcKernel);

    OpBuilder builder{ctx};
    dstKernel.setInsertionPointToTop(builder);
    mlir::SymbolTable symTable{dstKernel};

    IRMapping &remapper = oldToNew;

    // Don't fuse in the elementwise kernel invoke directly.
    auto skip = [](Operation *op) {
      if (auto call = dyn_cast<MOGG::CallOp>(op)) {
        if (call.getMojoKernelTarget() == Utils::MOGGElementwiseTarget)
          return true;
      }

      if (isa<MOGG::OutputPlaceholdersOp>(op))
        return true;

      return false;
    };

    llvm::StringMap<StringRef> oldNameToNew;
    srcKernel.fuseBodyIntoNewWithSkip(builder, symTable, remapper, oldNameToNew,
                                      skip);

    // The kernel we fused in will expose its fusion via the fusion body
    // attribute. That lambda will now exist in our function with a possibly
    // remapped name (but probably not) so look through the map to find the new
    // name.
    StringRef oldName = *srcKernel.getFusionBodyLambda();
    auto fusedLambdaName = oldNameToNew[oldName];

    // Find the lambda inside the new body.
    auto fusedLambda = cast<LambdaOp>(symTable.lookup(fusedLambdaName));

    // Invoke the fused lambda inside the body of the input lambda.
    auto dstInputLambdaName = SmallVector<StringAttr>(
        fusionInterfaces.getAsRange<StringAttr>())[inputIdx];
    auto dstInputLambda = cast<LambdaOp>(symTable.lookup(dstInputLambdaName));
    Block &lambdaBody = dstInputLambda.getRegion().front();
    Value indices = lambdaBody.getArgument(0);

    // Invoke the new lambda.
    builder.setInsertionPoint(&lambdaBody, lambdaBody.begin());
    auto invoke = builder.create<MOGG::LambdaInvokeOp>(
        fusedLambda.getLoc(), TypeRange{},
        SymbolRefAttr::get(ctx, fusedLambdaName, {}), ValueRange{indices},
        mlir::BoolAttr{});

    // Inline it.
    auto newValues = inlineInvoke(invoke, fusedLambda);
    fusedLambda.erase();
    invoke.erase();

    // Find all the stores that write through the argument that is being fused.
    SmallVector<MOGG::CallOp> stores;
    for (auto op : dstInputLambda.getOps<MOGG::CallOp>()) {
      if (op.getMojoKernelTarget() == Utils::MOGGSimdStoreTarget &&
          op.getOperand(0) == fusionArgument) {
        stores.push_back(op);
      }
    }

    Value simdValue = stores.back()->getOperand(2);

    // Erase all store ops.
    for (auto storeOp : stores)
      storeOp.erase();

    // Replace all load ops that read from the argument that is being fused.
    for (auto load :
         llvm::make_early_inc_range(dstInputLambda.getOps<MOGG::CallOp>())) {
      // Skip non simd loads.
      if (load.getMojoKernelTarget() != Utils::MOGGSimdLoadTarget)
        continue;

      if (load.getOperand(0) == fusionArgument) {
        load.replaceAllUsesWith(ValueRange{simdValue});
        load.erase();
      }
    }

    // If we have the `lambdas_have_fusion` parameter, a.k.a matmul,
    // then we should set it to true now.
    for (auto call : dstKernel.getOps<MOGG::CallOp>()) {
      BoolAttr trueAttr = BoolAttr::get(ctx, true);
      call.setKGENParamToValueIfPresent(Utils::LambdasHaveFusion, trueAttr);
    }

    // Delete the <operand, argument> pair from which we fused.
    dstKernel.removeOperandAtIndex(inputIdx);

    // Delete the cloned kernel.
    opsToErase.push_back(srcKernel);
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

    // Disable fusion on experimental new KGEN in MOGG for now.
    if (user.getIsKGEN() != kernel.getIsKGEN())
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
                                    SmallVector<KernelOp> &opsToErase) {

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

    OutputPlaceholdersOp originalOuts = kernelToFuseInto.getOutputPlaceholder();
    OutputPlaceholdersOp userOuts = userToFuseIn.getOutputPlaceholder();

    const bool isMatmulOrConvolution =
        containsMatmulOrConvolution(kernelToFuseInto);

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

    // Step 2: We clone all non-trivial operators in the user kernel into the
    // src kernel. Here, we ignore all the ParamDeclareRegionOp, which we will
    // take care of later in a more specific way.

    // Major mapper we use to clone all kinds of the operators in the user
    // kernel.
    IRMapping kgenOldToNew;
    MOGG::ToKGENOp newOutsToKgenOp;
    OutputPlaceholdersOp newOuts;

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

        kgenOldToNew.map(toKgen.getResults(), newToKgen.getResults());
        oldToNew.map(toKgen.getResults(), newToKgen.getResults());
      } else if (isa<KGEN::POP::StackAllocationOp>(op)) {
        // We clone StackAllocationOp and track its mapping. Both
        // StackAllocationOp and ToKGENOp can produce outputs directly, feeding
        // the KGEN calls.
        Operation *newAlloc = builder.clone(op, oldToNew);
        oldToNew.map(op.getResults(), newAlloc->getResults());
        kgenOldToNew.map(op.getResults(), newAlloc->getResults());
      } else if (isa<MOGG::OutputPlaceholdersOp>(op)) {
        // OutputPlaceholdersOp is the first operator in the user kernel to
        // clone.
        newOuts = cast<OutputPlaceholdersOp>(builder.clone(op));
        oldToNew.map(op.getResult(0), newOuts.getResult(0));
      } else if (isa<KGEN::ParamDeclareOp>(op)) {
        builder.clone(op);
      } else if (isa<MOGG::OutputOp>(op) ||
                 isa<KGEN::ParamDeclareRegionOp>(op)) {
        // Do nothing.
      } else if (auto callop = dyn_cast<KGEN::CallOp>(op)) {
        StringRef funcName =
            cast<mlir::FlatSymbolRefAttr>(callop.getCalleeSymbol()).getValue();
        // After Tensor APIs becomes the de-factor standard APIs (builtin
        // library), we will use MojoLibararyAnalysis to check this. Clone every
        // function call except the one has the `elementwise` decorator.
        if (!funcName.starts_with(Utils::MOGGTensorForEach)) {
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
          builder.clone(*lambdaOp, kgenOldToNew);
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
    KGEN::CallOp outlinedCallOp = *elementwiseOp.getOps<KGEN::CallOp>().begin();
    assert(outlinedCallOp);

    auto userOutputLambdaOp = cast<KGEN::ParamDeclareRegionOp>(
        userSymTable.lookup(userOutputLambdaName));
    KGEN::CallOp userSimdStoreOp =
        *userOutputLambdaOp.getOps<KGEN::CallOp>().begin();
    KGEN::CallOp srcSimdStoreOp =
        *srcOutputLambdaOp.getOps<KGEN::CallOp>().begin();

    // Remap to_kgen to the new one, which is corresponding to the output
    // placeholder in the src kernel.
    for (auto toKgenOp : userToFuseIn.getOps<MOGG::ToKGENOp>()) {
      if (toKgenOp.getInput() == userBlock.getArgument(fusionOperandIdx))
        kgenOldToNew.map(toKgenOp->getResult(0), newOutsToKgenOp->getResult(0));
    }

    // Remap the SIMD indices of the simd_store_internal in the output lambda.
    kgenOldToNew.map(userSimdStoreOp.getOperand(1),
                     srcSimdStoreOp.getOperand(1));

    // Remap SIMD indices for the COMPUTE_OUTLINED call.
    for (Value operand : outlinedCallOp->getOperands()) {
      if (isa<mlir::BlockArgument>(operand))
        kgenOldToNew.map(operand, srcSimdStoreOp.getOperand(1));
    }

    // Clone all the elementwise lambdas except the final one.
    builder.setInsertionPointAfter(newInputLambdaOp);
    for (auto declRegion : userToFuseIn.getOps<KGEN::ParamDeclareRegionOp>()) {
      StringAttr regionName = declRegion.getParamDeclAttr().getName();
      if (!lambdaSkipSet.contains(regionName)) {
        auto newRegion = cast<KGEN::ParamDeclareRegionOp>(
            builder.clone(*declRegion.getOperation(), kgenOldToNew));
        newRegion->removeAttr(mlir::SymbolTable::getSymbolAttrName());
      }
    }

    // Insert the COMPUTE_OUTLINED call for the final elementwise lambda. We
    // clone the user kernel's SIMD store into the src kernel to account for the
    // mixed precision case.
    KGEN::CallOp newOutlinedCallOp = cast<KGEN::CallOp>(
        builder.clone(*outlinedCallOp.getOperation(), kgenOldToNew));

    kgenOldToNew.map(userSimdStoreOp.getOperand(2),
                     newOutlinedCallOp.getResult(0));
    auto newSimdStoreOp = cast<KGEN::CallOp>(
        builder.clone(*userSimdStoreOp.getOperation(), kgenOldToNew));

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

    walker.recursivelyReplaceElementsIn(newOutlinedCallOp,
                                        /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
    walker.recursivelyReplaceElementsIn(newSimdStoreOp,
                                        /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);

    // Step 6: Finally, we clean up the kernel metadata, eliminate the user
    // kernel, and handle the mixed precision case.

    // Disable future output fusion.
    kernelToFuseInto.removeOutputFusionInterfaceAttr();

    // Matmul with mixed precision (bad case): Matmul kernels actually take in
    // an output buffer despite having a lambda. Therefore, in mixed
    // precision, we need another buffer to do the accumulation. This results
    // in the kernel op having two return values to trigger allocations in
    // later passes.
    if (isMatmulOrConvolution && isMixedPrecision) {
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
          /*isKGEN=*/kernelToFuseInto.getIsKGENAttr(),
          /*summary=*/getSummaryString({kernelToFuseInto, userToFuseIn}),
          /*isAsync=*/kernelToFuseInto.getIsAsyncAttr(),
          /*allocIndices=*/ArrayAttr{});

      if (kernelToFuseInto.getDeviceRef())
        newDstKernel.setDeviceRef(kernelToFuseInto.getDeviceRef());

      newDstKernel.getBody().takeBody(kernelToFuseInto.getBody());

      userToFuseIn.getResult(0).replaceAllUsesWith(newDstKernel.getResult(0));
      opsToErase.push_back(userToFuseIn);
      opsToErase.push_back(kernelToFuseInto);

      return newDstKernel;
    }

    kernelToFuseInto.setSummaryAttr(
        getSummaryString({kernelToFuseInto, userToFuseIn}));

    originalOuts->replaceAllUsesWith(newOuts);
    srcSimdStoreOp.erase();
    originalOuts.erase();

    userToFuseIn.replaceAllUsesWith(kernelToFuseInto);
    opsToErase.push_back(userToFuseIn);

    return kernelToFuseInto;
  }

  // Fuse the use of this MOGG kernel into the output lambda.
  KernelOp fuseIntoOutputLambdaMOGG(KernelOp kernelToFuseInto,
                                    StringAttr outputLambdaName,
                                    SmallVector<KernelOp> &opsToErase) {
    MLIRContext *ctx = kernelToFuseInto.getContext();

    // Guarded to only fuse 1 output kernels.
    constexpr size_t outputIdx = 0;
    Value originalResult = kernelToFuseInto.getResult(outputIdx);

    // We've already guarded that we only have one user.
    MOGG::KernelOp userToFuseIn =
        cast<MOGG::KernelOp>(*originalResult.user_begin());

    Block &userBlock = userToFuseIn.getBody().front();

    OutputPlaceholdersOp originalOuts = kernelToFuseInto.getOutputPlaceholder();
    OutputPlaceholdersOp oldOuts = userToFuseIn.getOutputPlaceholder();

    IRMapping oldToNew;

    const bool isMatmulOrConvolution =
        containsMatmulOrConvolution(kernelToFuseInto);
    // Elementwise fusion expects shapes to be identical but data types can
    // change.
    const bool isMixedPrecision =
        originalOuts.getResult(0).getType() != oldOuts.getResult(0).getType();

    // TODO(#22349)
    // If this is not a matmul/convolution kernel or if it isn't mixed
    // precision, remap references to the old outputs to the new values. This
    // effectively gets rid of the output placeholder of `kernelToFuseInto`
    // after mergning the kernels.
    if (!isMatmulOrConvolution)
      oldToNew.map(originalOuts.getResult(0), oldOuts.getResult(0));
    else if (!isMixedPrecision)
      oldToNew.map(oldOuts.getResult(0), originalOuts.getResult(0));

    // Mapping of the graph level kernel argument to the inner block
    // arg. `mogg.kernel(%graph) : (%inner : tensor<...>) {
    DenseMap<Value, Value> seenOperands;

    // First add each operand and its body from the kernel we are fusing
    // into to the map.
    for (auto [operand, blockArg] :
         llvm::zip(userToFuseIn.getOperands(), userBlock.getArguments())) {
      // Check if it's already in just so we are deterministic in which
      // operand we unique.
      if (!seenOperands.contains(operand))
        seenOperands[operand] = blockArg;
    }

    // Next look at the kernel we are fusing in, either add the operand
    // if it is new or use the existing argument.
    for (auto [operand, blockArg] :
         llvm::zip(userToFuseIn.getOperands(), userBlock.getArguments())) {
      // We don't want to fuse ourselves! For each block argument which
      // we are replacing, replace it with the output placeholder. (i.e
      // each operand which was produced by the original kernel).
      if (operand == originalResult) {
        oldToNew.map(blockArg, originalOuts->getResult(0));
        continue;
      }

      // Either add a new operand to the kernel or reuse the existing.
      auto itr = seenOperands.find(operand);
      if (itr != seenOperands.end()) {
        Value newBlockArg = kernelToFuseInto.addOperand(operand);
        oldToNew.map(blockArg, newBlockArg);
        seenOperands[operand] = newBlockArg;
      }
    }

    // Inline all the lambdas in the op we are pulling in.
    inlineLambdasInKernel(userToFuseIn);

    OpBuilder builder{ctx};
    kernelToFuseInto.setInsertionPointToTop(builder);

    mlir::SymbolTable symTable{kernelToFuseInto};
    llvm::StringMap<StringRef> oldNameToNew;

    IRMapping &remapper = oldToNew;

    // Before we fuse remove any store to the output in the lambda we
    // are fusing into. The lambda coming in already does that.
    auto outputLambdaToFuseInto =
        cast<LambdaOp>(symTable.lookup(outputLambdaName));
    for (auto store : llvm::make_early_inc_range(
             outputLambdaToFuseInto.getOps<MOGG::CallOp>())) {
      if (store.getMojoKernelTarget() == Utils::MOGGSimdStoreTarget &&
          store.getOperand(0) == originalOuts->getResult(0)) {
        store.erase();
      }
    }

    // Don't fuse in the elementwise kernel invoke directly.
    auto skipElementwise = [&](Operation *op) {
      if (auto call = dyn_cast<MOGG::CallOp>(op)) {
        if (call.getMojoKernelTarget() == Utils::MOGGElementwiseTarget)
          return true;
      }

      // TODO(#22349)
      // If we are fusing into a matmul/conv and it isn't mixed precision, then
      // skip the placeholder.
      if (isMatmulOrConvolution && !isMixedPrecision &&
          isa<MOGG::OutputPlaceholdersOp>(op)) {
        return true;
      }

      return false;
    };

    userToFuseIn.fuseBodyIntoNewWithSkip(builder, symTable, remapper,
                                         oldNameToNew, skipElementwise);

    // The kernel we fused in will expose its fusion via the fusion body
    // attribute. That lambda will now exist in our function with a possibly
    // remapped name (but probably not) so look through the map to find the new
    // name.
    StringRef oldName = *userToFuseIn.getFusionBodyLambda();
    auto fusedLambdaName = oldNameToNew[oldName];

    // Find the lambda inside the new body.
    auto fusedLambda = cast<LambdaOp>(symTable.lookup(fusedLambdaName));

    // Invoke the fused lambda inside the body of the new lambda.
    Block &lambdaBody = outputLambdaToFuseInto.getRegion().front();
    Value indices = lambdaBody.getArgument(0);
    Value simdValue = lambdaBody.getArgument(1);

    builder.setInsertionPoint(&lambdaBody, lambdaBody.begin());

    // Invoke the new lambda.
    auto invoke = builder.create<MOGG::LambdaInvokeOp>(
        fusedLambda.getLoc(), TypeRange{},
        SymbolRefAttr::get(ctx, fusedLambdaName, {}), ValueRange{indices},
        mlir::BoolAttr{});

    // Inline it.
    auto newValues = inlineInvoke(invoke, fusedLambda);
    fusedLambda.erase();
    invoke.erase();

    // Replace any SIMD loads of the output with the value.
    for (auto load : llvm::make_early_inc_range(
             outputLambdaToFuseInto.getOps<MOGG::CallOp>())) {
      // Skip non simd loads.
      if (load.getMojoKernelTarget() != Utils::MOGGSimdLoadTarget)
        continue;

      // Anything which loads the output is a reference to lambda simd
      // value parameter.
      if (load.getOperand(0) == originalOuts->getResult(0)) {
        load.replaceAllUsesWith(ValueRange{simdValue});
        load.erase();
      }
    }

    // If we have the `lambdas_have_fusion` parameter, a.k.a matmul,
    // then we should set it to true now.
    for (auto call : kernelToFuseInto.getOps<MOGG::CallOp>()) {
      BoolAttr trueAttr = BoolAttr::get(ctx, true);
      call.setKGENParamToValueIfPresent(Utils::LambdasHaveFusion, trueAttr);
    }

    // Disable future output fusion.
    kernelToFuseInto.removeOutputFusionInterfaceAttr();

    // Finding the old output op.
    MOGG::OutputOp oldOutputOp = nullptr;
    for (auto op : kernelToFuseInto.getOps<MOGG::OutputOp>()) {
      if (op->getOperand(0).getDefiningOp() == originalOuts) {
        oldOutputOp = op;
        break;
      }
    }
    assert(oldOutputOp);

    // Finding the new placeholder that was inserted earlier.
    MOGG::OutputPlaceholdersOp newPlaceholder = nullptr;
    for (auto op : kernelToFuseInto.getOps<MOGG::OutputPlaceholdersOp>()) {
      if (op != originalOuts) {
        newPlaceholder = op;
        break;
      }
    }
    assert((isMatmulOrConvolution && !isMixedPrecision) || newPlaceholder);

    Location loc = kernelToFuseInto.getLoc();
    // TODO(#22349)
    // Three cases:
    // Not a matmul/conv (good case): Simply create a new output op (in case
    // data types change) and change the output type of `kernelToFuseInto`.
    if (!isMatmulOrConvolution) {
      builder.setInsertionPoint(oldOutputOp);
      builder.create<MOGG::OutputOp>(loc, newPlaceholder->getResult(0));
      kernelToFuseInto.getResult(0).setType(
          newPlaceholder->getResult(0).getType());
      oldOutputOp.erase();
      originalOuts.erase();
      userToFuseIn.replaceAllUsesWith(kernelToFuseInto);

      // Capture the fusion within the summary.
      kernelToFuseInto.setSummaryAttr(
          getSummaryString({kernelToFuseInto, userToFuseIn}));

      return kernelToFuseInto;
    }
    // Matmul with mixed precision (bad case): Matmul kernels actually take in
    // an output buffer despite having a lambda. Therefore, in mixed
    // precision, we need another buffer to do the accumulation. This results
    // in the kernel op having two return values to trigger allocations in
    // later passes.
    if (isMixedPrecision) {
      builder.setInsertionPoint(originalOuts);
      auto combinedPlaceholders = builder.create<MOGG::OutputPlaceholdersOp>(
          loc, ArrayRef<Type>{newPlaceholder->getResult(0).getType(),
                              originalOuts->getResult(0).getType()});

      builder.setInsertionPoint(oldOutputOp);
      builder.create<MOGG::OutputOp>(loc, combinedPlaceholders->getResults());
      newPlaceholder.getResult(0).replaceAllUsesWith(
          combinedPlaceholders.getResult(0));
      originalOuts.getResult(0).replaceAllUsesWith(
          combinedPlaceholders.getResult(1));
      newPlaceholder.erase();
      oldOutputOp.erase();
      originalOuts.erase();

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
          /*isKGEN=*/kernelToFuseInto.getIsKGENAttr(),
          /*summary=*/getSummaryString({kernelToFuseInto, userToFuseIn}),
          /*isAsync=*/kernelToFuseInto.getIsAsyncAttr(),
          /*allocIndices=*/ArrayAttr{});
      if (kernelToFuseInto.getDeviceRef())
        newDstKernel.setDeviceRef(kernelToFuseInto.getDeviceRef());
      newDstKernel.getBody().takeBody(kernelToFuseInto.getBody());
      userToFuseIn.getResult(0).replaceAllUsesWith(newDstKernel.getResult(0));

      opsToErase.push_back(userToFuseIn);
      opsToErase.push_back(kernelToFuseInto);

      return newDstKernel;
    }
    // Matmul without mixed precision (ugly case): Return value is unchanged
    // and there is a single output placeholder. Nothing to do. The reason
    // there is a third case at all is because performance is slightly better
    // than having two outputs.
    userToFuseIn.replaceAllUsesWith(kernelToFuseInto);
    kernelToFuseInto.setSummaryAttr(
        getSummaryString({kernelToFuseInto, userToFuseIn}));

    return kernelToFuseInto;
  }

  void runOnOperation() override {
    MO::GraphOp moGraph = getOperation();

    SmallVector<MOGG::KernelOp> kernelOpsToErase;
    for (MOGG::KernelOp kernelToFuseInto : moGraph.getOps<MOGG::KernelOp>()) {
      DeviceRefAttr beforeDevice = MO::getDeviceOrDefault(kernelToFuseInto);
      std::optional<ArrayAttr> outputLambdas =
          kernelToFuseInto.getOutputFusionInterface();
      // Try fuse into the output lambda if we have a fusion interface for that
      // output.
      if (outputLambdas.has_value()) {
        ArrayAttr lambdas = *outputLambdas;

        // Fuse into the output lambdas if possible.
        if (canDoOutputFusion(kernelToFuseInto, lambdas)) {
          StringAttr outputLambdaName = cast<StringAttr>(lambdas[0]);
          if (kernelToFuseInto.getIsKGEN().value_or(false))
            kernelToFuseInto = fuseIntoOutputLambdaKGEN(
                kernelToFuseInto, outputLambdaName, kernelOpsToErase);
          else
            kernelToFuseInto = fuseIntoOutputLambdaMOGG(
                kernelToFuseInto, outputLambdaName, kernelOpsToErase);
          ;
        }
      }
      ASSERT_STREAM(beforeDevice == MO::getDeviceOrDefault(kernelToFuseInto),
                    << "Must be same device after output lambda fusion");

      auto inputLambdas = kernelToFuseInto.getInputFusionInterface();
      if (!inputLambdas) {
        continue;
      }

      uint64_t numOperands = kernelToFuseInto.getNumOperands();
      for (int64_t inputIdx = numOperands - 1; inputIdx >= 0; --inputIdx) {
        if (canDoInputFusion(kernelToFuseInto,
                             kernelToFuseInto.getInputFusionInterfaceAttr(),
                             inputIdx)) {
          fuseIntoInputLambda(kernelToFuseInto,
                              kernelToFuseInto.getInputFusionInterfaceAttr(),
                              inputIdx, kernelOpsToErase);
        }
      }
      ASSERT_STREAM(beforeDevice == MO::getDeviceOrDefault(kernelToFuseInto),
                    << "Must be same device after input lambda fusion");
    }

    for (KernelOp kernelOp : kernelOpsToErase)
      kernelOp.erase();
  }
};

} // namespace

} // namespace M::MOGG
