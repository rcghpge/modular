//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/Utils.h"

#include "mlir/IR/IRMapping.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEINOUTLAMBDAS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// TODO(19870): Remove checks when matmul supports full fusion
bool hasMatmulCall(MOGG::KernelOp kernel) {
  for (auto call : kernel.getOps<MOGG::CallOp>()) {
    if (call.getMojoKernelTarget() == "mo.matmul")
      return true;
    if (call.getMojoKernelTarget() == "mo.batch_matmul")
      return true;
  }
  return false;
}

struct FuseInOutLambdas : public impl::FuseInOutLambdasBase<FuseInOutLambdas> {
  using FuseInOutLambdasBase::FuseInOutLambdasBase;

  bool canDoInputFusion(MOGG::KernelOp kernel, ArrayAttr fusionInterfaces,
                        uint64_t inputIdx) {

    // Should be unreachable sanity check. Ensure this is not an elementwise
    // kernel (as if it where after fusion we would trigger more fusions).
    auto isElemwise = kernel.getFusionBodyLambda();
    if (isElemwise)
      return false;

    // Input fusion interface must be non null.
    auto inputFusion = SmallVector<StringAttr>(
        fusionInterfaces.getAsRange<StringAttr>())[inputIdx];
    if (!inputFusion.size())
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

    // Can only fuse with kernels which maintain the same output type for
    // matmul epilogue fusion.
    if ((hasMatmulCall(kernel) || hasMatmulCall(parentKernel)) &&
        parentKernel->getResultTypes() != kernel->getResultTypes())
      return false;

    // Check that the kernel has a single output.
    if (parentKernel->getNumResults() != 1)
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
    if (!outputInterface.size())
      return false;

    return true;
  }

  void fuseIntoInputLambda(MOGG::KernelOp dstKernel, ArrayAttr fusionInterfaces,
                           uint64_t inputIdx) {
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
    srcKernel.inlineLambdas();

    OpBuilder builder{ctx};
    dstKernel.setInsertionPointToTop(builder);
    mlir::SymbolTable symTable{dstKernel};

    IRMapping &remapper = oldToNew;

    // Don't fuse in the elementwise kernel invoke directly.
    auto skipElementwise = [](Operation *op) {
      if (auto call = dyn_cast<MOGG::CallOp>(op)) {
        if (call.getMojoKernelTarget() == "elementwise")
          return true;
      }
      return false;
    };

    llvm::StringMap<StringRef> oldNameToNew;
    srcKernel.fuseBodyIntoNewWithSkip(builder, symTable, remapper, oldNameToNew,
                                      skipElementwise);

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
      if (op.getOperand(0) == fusionArgument) {
        stores.push_back(op);
      }
    }

    Value simdValue = stores.back()->getOperand(2);

    // Erase all store ops.
    for (auto storeOp : stores)
      storeOp.erase();

    // Replace all load ops that read from the argument that is being fused.
    for (auto load : llvm::make_early_inc_range(
             dstInputLambda.getOps<MOGG::SimdLoadOp>())) {
      if (load.getTensor() == fusionArgument) {
        load.replaceAllUsesWith(simdValue);
        load.erase();
      }
    }

    // If we have the `lambdas_have_fusion` parameter, a.k.a matmul,
    // then we should set it to true now.
    for (auto call : dstKernel.getOps<MOGG::CallOp>()) {
      BoolAttr trueAttr = BoolAttr::get(ctx, true);
      call.setKGENParamToValueIfPresent("lambdas_have_fusion", trueAttr);
    }

    // Delete the <operand, argument> pair from which we fused.
    dstKernel.removeOperandAtIndex(inputIdx);

    // Delete the cloned kernel.
    srcKernel.erase();
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

    // Should be unreachable sanity check. Ensure this is not an elementwise
    // kernel (as if it where after fusion we would trigger more fusions).
    auto isElemwise = kernel.getFusionBodyLambda();
    if (isElemwise)
      return false;

    // Output can only have one user and that user must be a fusable elementwise
    // kernel.
    for (Value result : kernel.getResults()) {
      if (!result.hasOneUse())
        return false;

      // Check it's a kernel.
      MOGG::KernelOp user = dyn_cast<MOGG::KernelOp>(*result.user_begin());
      if (!user)
        return false;

      // Can only fuse with kernels which maintain the same output type.
      if ((hasMatmulCall(kernel) || hasMatmulCall(user)) &&
          user->getResultTypes() != kernel->getResultTypes())
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
        if (operand == kernel.getResult(0) && attr.strref() == "")
          return false;
      }
    }

    return true;
  }

  // Fuse the use of this kernel into the output lambda.
  void fuseIntoOutputLambda(KernelOp kernelToFuseInto,
                            StringAttr outputLambdaName) {
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

    // Remap references to the old outputs to the new values.
    for (auto [oldOut, newOut] :
         llvm::zip(oldOuts.getResults(), originalOuts.getResults()))
      oldToNew.map(oldOut, newOut);

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
    userToFuseIn.inlineLambdas();

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
      if (store.getMojoKernelTarget() == "simd_store" &&
          store.getOperand(0) == originalOuts->getResult(0)) {
        store.erase();
      }
    }

    // Don't fuse in the elementwise kernel invoke directly.
    auto skipElementwise = [](Operation *op) {
      if (auto call = dyn_cast<MOGG::CallOp>(op)) {
        if (call.getMojoKernelTarget() == "elementwise")
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
             outputLambdaToFuseInto.getOps<MOGG::SimdLoadOp>())) {
      // Anything which loads the output is a reference to lambda simd
      // value parameter.
      if (load.getTensor() == originalOuts->getResult(0)) {
        load.replaceAllUsesWith(simdValue);
        load.erase();
      }
    }

    // If we have the `lambdas_have_fusion` parameter, a.k.a matmul,
    // then we should set it to true now.
    for (auto call : kernelToFuseInto.getOps<MOGG::CallOp>()) {
      BoolAttr trueAttr = BoolAttr::get(ctx, true);
      call.setKGENParamToValueIfPresent("lambdas_have_fusion", trueAttr);
    }
    // Finally replace all uses of the thing we fused in with this
    // kernel.
    userToFuseIn.replaceAllUsesWith(kernelToFuseInto);

    // Disable future output fusion.
    kernelToFuseInto.removeOutputFusionInterfaceAttr();
  }

  void runOnOperation() override {
    MO::GraphOp moGraph = getOperation();

    for (MOGG::KernelOp kernelToFuseInto : moGraph.getOps<MOGG::KernelOp>()) {
      std::optional<ArrayAttr> outputLambdas =
          kernelToFuseInto.getOutputFusionInterface();
      // Try fuse into the output lambda if we have a fusion interface for that
      // output.
      if (outputLambdas.has_value()) {
        ArrayAttr lambdas = *outputLambdas;

        // Fuse into the output lambdas if possible.
        if (canDoOutputFusion(kernelToFuseInto, lambdas)) {
          StringAttr outputLambdaName = cast<StringAttr>(lambdas[0]);
          fuseIntoOutputLambda(kernelToFuseInto, outputLambdaName);
        }
      }

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
                              inputIdx);
        }
      }
    }
  }
};

} // namespace

} // namespace M::MOGG
