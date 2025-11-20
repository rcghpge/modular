//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/Interfaces/LoweringState.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGInterfaces.h" // IWYU: needed for type declarations used via MOGG::
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_ADDTRYEXCEPTBLOCKS
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

namespace M::MGP {

namespace {

struct AddTryExceptBlocks
    : public MGP::impl::AddTryExceptBlocksBase<AddTryExceptBlocks> {
  using AddTryExceptBlocksBase::AddTryExceptBlocksBase;

  void runOnOperation() override {
    if (!MOGG::useExperimentalMojoPath())
      return;

    OpBuilder builder(getOperation()->getContext());
    for (auto kernelOp :
         getOperation().getBody(0)->getOps<MOGG::KernelStubOp>()) {
      if (kernelOp.getAbi() != MOGG::ABIKind::MGP || !kernelOp.isThrows())
        continue;

      auto &kernelBody = kernelOp.getBodyBlock();
      Location loc = kernelOp->getLoc();
      auto terminator = kernelBody.getTerminator();

      // Seek the first non unpacking op.
      builder.setInsertionPointToStart(&kernelBody);
      for (auto &&op : kernelBody.getOperations()) {
        if (!isa<MOGG::KernelArgumentConsumer>(op)) {
          builder.setInsertionPointAfter(&op);
          break;
        }
      }

      // Move all ops inside the Try clause.
      auto tryOp =
          MOGG::TryOp::create(builder, loc, terminator->getOperandTypes());
      Block &tryBlock = tryOp.getTryRegion().emplaceBlock();
      SmallVector<Value> tryPackedValues;
      for (Operation &op :
           llvm::make_early_inc_range(kernelBody.getOperations())) {
        if (!isa<MOGG::TryOp, MOGG::OutputOp, MOGG::AsyncDelOp,
                 MOGG::KernelArgumentConsumer>(op)) {
          op.moveBefore(&tryBlock, tryBlock.end());

          if (isa<MOGG::KernelTerminatorProvider>(op))
            tryPackedValues.push_back(op.getResult(0));
        }
      }

      builder.setInsertionPointToEnd(&tryBlock);
      auto tryYieldOp = MOGG::YieldOp::create(builder, loc, tryPackedValues);

      // The except block returns a chain error and dummy values for other
      // outputs.
      Block &exceptBlock = tryOp.getExceptRegion().emplaceBlock();
      Value errArg =
          exceptBlock.addArgument(MOGG::ErrorType::get(loc->getContext()), loc);
      builder.setInsertionPointToStart(&exceptBlock);

      // Create individual AsyncErrorOp for each output (chain and non-chain).
      // Each AsyncErrorOp represents setting an error on one specific output.
      SmallVector<Value> exceptYieldValues;
      exceptYieldValues.reserve(tryYieldOp.getOperands().size());
      for (Value tryYieldOperand : tryYieldOp.getOperands()) {
        Value asyncError = MOGG::AsyncErrorOp::create(
            builder, loc, tryYieldOperand.getType(), errArg);
        exceptYieldValues.push_back(asyncError);
      }

      MOGG::YieldOp::create(builder, loc, exceptYieldValues);

      // Update the final terminator outputs to the try op results.
      terminator->setOperands(tryOp.getOutputs());
    }
  }
};

} // namespace

} // namespace M::MGP
