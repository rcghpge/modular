//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Interfaces/LoweringState.h"
#include "GraphCompiler/MOGGDialect/MOGGInterfaces.h" // IWYU: needed for type declarations used via MOGG::
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_ADDTRYEXCEPTBLOCKS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

namespace {

struct AddTryExceptBlocks
    : public MOGG::impl::AddTryExceptBlocksBase<AddTryExceptBlocks> {
  using AddTryExceptBlocksBase::AddTryExceptBlocksBase;

  void runOnOperation() override {
    auto kernelOp = getOperation();
    OpBuilder builder(kernelOp->getContext());
    auto &kernelBody = kernelOp.getBodyBlock();
    auto terminator = kernelBody.getTerminator();
    Location loc = kernelOp->getLoc();

    // Handle MOGG ABI stubs: wrap all ops (except TryOp and OutputOp) in a
    // TryOp. The try block contains the setup ops and WithTraceOp (added by
    // createKernelStub). The except block contains a RaiseOp that formats
    // the error with full kernel context (name, fusion info, traceback).
    //
    // MOGG stubs take this arm, or nothing renders their error context. The
    // terminator stays outside the try, so its values are forwarded through.
    if (kernelOp.getAbi() == MOGG::ABIKind::MOGG) {
      builder.setInsertionPointToStart(&kernelBody);
      auto tryOp =
          MOGG::TryOp::create(builder, loc, terminator->getOperandTypes());
      Block &tryBlock = tryOp.getTryRegion().emplaceBlock();
      for (Operation &op :
           llvm::make_early_inc_range(kernelBody.getOperations())) {
        if (!isa<MOGG::TryOp, MOGG::OutputOp>(op))
          op.moveBefore(&tryBlock, tryBlock.end());
      }
      builder.setInsertionPointToEnd(&tryBlock);
      MOGG::YieldOp::create(builder, loc, terminator->getOperands());

      Block &exceptBlock = tryOp.getExceptRegion().emplaceBlock();
      Value errArg =
          exceptBlock.addArgument(MOGG::ErrorType::get(loc->getContext()), loc);
      builder.setInsertionPointToStart(&exceptBlock);
      MOGG::RaiseOp::create(builder, loc, errArg);

      terminator->setOperands(tryOp.getOutputs());
      return;
    }

    if (kernelOp.getAbi() != MOGG::ABIKind::MGP || !kernelOp.isThrows())
      return;

    // For throwing kernels, wrap the compute ops in a TryOp. Local
    // destructors still run during exception unwinding, so cleanup is handled
    // implicitly in both paths.
    builder.setInsertionPointToStart(&kernelBody);
    auto tryOp =
        MOGG::TryOp::create(builder, loc, terminator->getOperandTypes());
    Block &tryBlock = tryOp.getTryRegion().emplaceBlock();
    for (Operation &op :
         llvm::make_early_inc_range(kernelBody.getOperations())) {
      if (!isa<MOGG::TryOp, MOGG::OutputOp>(op))
        op.moveBefore(&tryBlock, tryBlock.end());
    }

    builder.setInsertionPointToEnd(&tryBlock);
    MOGG::YieldOp::create(builder, loc, terminator->getOperands());

    // The except block returns a chain error and dummy values for other
    // outputs.
    Block &exceptBlock = tryOp.getExceptRegion().emplaceBlock();
    Value errArg =
        exceptBlock.addArgument(MOGG::ErrorType::get(loc->getContext()), loc);
    builder.setInsertionPointToStart(&exceptBlock);

    // Create individual AsyncErrorOp for each output (chain and non-chain).
    // Each AsyncErrorOp represents setting an error on one specific output.
    SmallVector<Value> exceptYieldValues;
    exceptYieldValues.reserve(tryOp.getOutputs().size());
    for (Value tryYieldOperand : tryOp.getOutputs()) {
      Value asyncError = MOGG::AsyncErrorOp::create(
          builder, loc, tryYieldOperand.getType(), errArg);
      exceptYieldValues.push_back(asyncError);
    }

    MOGG::YieldOp::create(builder, loc, exceptYieldValues);

    // Update the final terminator outputs to the try op results.
    terminator->setOperands(tryOp.getOutputs());
  }
};

} // namespace

} // namespace M::MOGG
