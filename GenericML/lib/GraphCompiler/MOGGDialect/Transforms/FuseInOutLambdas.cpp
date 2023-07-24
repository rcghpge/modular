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

struct FuseInOutLambdas : public impl::FuseInOutLambdasBase<FuseInOutLambdas> {
  using FuseInOutLambdasBase::FuseInOutLambdasBase;

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

    // Fusion interface is expected to be a string attr.
    for (Attribute attr : fusionInterface) {
      if (!isa<StringAttr>(attr))
        return false;
    }

    // Output can only have one user and that user must be a fusable elementwise
    // kernel.
    for (Value result : kernel.getResults()) {
      if (!result.hasOneUse())
        return false;

      // Check it's a kernel.
      MOGG::KernelOp user = dyn_cast<MOGG::KernelOp>(*result.user_begin());
      if (!user)
        return false;

      // Check the kernel is a fusable elementwise.
      auto attr = user.getFusionBodyLambda();
      if (!attr)
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
      } else {
        oldToNew.map(blockArg, itr->second);
      }
    }

    // Inline all the lambdas in the op we are pulling in.
    userToFuseIn.inlineLambdas();

    OpBuilder builder{ctx};
    builder.setInsertionPointAfter(originalOuts);

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
    }
  }
};

} // namespace

} // namespace M::MOGG
