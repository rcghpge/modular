//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/Interfaces/LoweringState.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGInterfaces.h" // IWYU: needed for type declarations used via MOGG::
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/FusionUtils.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_ADDTRYEXCEPTBLOCKS
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

namespace {

struct AddTryExceptBlocks
    : public MOGG::impl::AddTryExceptBlocksBase<AddTryExceptBlocks> {
  using AddTryExceptBlocksBase::AddTryExceptBlocksBase;

  /// Helper to set the builder insertion point for AsyncDelOp operations.
  /// Sets insertion point before the first AsyncReadyOp if one exists in the
  /// block, otherwise before the fallback operation (or at block end if
  /// nullptr).
  void setDelInsertionPoint(OpBuilder &builder, Block &block,
                            Operation *fallback) {
    auto asyncReadyOps = block.getOps<MOGG::AsyncReadyOp>();
    if (!asyncReadyOps.empty()) {
      builder.setInsertionPoint((*asyncReadyOps.begin()).getOperation());
    } else if (fallback) {
      builder.setInsertionPoint(fallback);
    } else {
      builder.setInsertionPointToEnd(&block);
    }
  }

  /// Helper to insert AsyncDelOp for all kernel arguments at the current
  /// insertion point.
  void insertAsyncDelOps(OpBuilder &builder, Location loc,
                         ArrayRef<BlockArgument> args) {
    if (!args.empty())
      MOGG::AsyncDelOp::create(builder, loc, args);
  }

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
    if (kernelOp.getAbi() == MOGG::ABIKind::MOGG) {
      if (!kernelOp.isThrows())
        return;

      builder.setInsertionPointToStart(&kernelBody);
      auto tryOp = MOGG::TryOp::create(builder, loc, TypeRange{});
      Block &tryBlock = tryOp.getTryRegion().emplaceBlock();
      for (Operation &op :
           llvm::make_early_inc_range(kernelBody.getOperations())) {
        if (!isa<MOGG::TryOp, MOGG::OutputOp>(op))
          op.moveBefore(&tryBlock, tryBlock.end());
      }
      builder.setInsertionPointToEnd(&tryBlock);
      MOGG::YieldOp::create(builder, loc, ValueRange{});

      Block &exceptBlock = tryOp.getExceptRegion().emplaceBlock();
      Value errArg =
          exceptBlock.addArgument(MOGG::ErrorType::get(loc->getContext()), loc);
      builder.setInsertionPointToStart(&exceptBlock);
      MOGG::RaiseOp::create(builder, loc, errArg);
      return;
    }

    if (kernelOp.getAbi() != MOGG::ABIKind::MGP)
      return;

    // Collect arguments that need cleanup. We need to do this before moving
    // ops around.
    SmallVector<BlockArgument> argsToDelete(kernelBody.getArguments().begin(),
                                            kernelBody.getArguments().end());

    // For non-throwing kernels, insert AsyncDelOp before AsyncReadyOp.
    // AsyncReadyOp signals that the output chain is ready, which tells
    // downstream consumers they can start. By inserting cleanup before
    // AsyncReadyOp, we ensure that when await() returns for downstream
    // consumers, all cleanup from this kernel has completed.
    //
    // Note: We specifically look for AsyncReadyOp and not AsyncPackOp because
    // pack operations may still reference values derived from arguments (like
    // memory storage handles from cached buffers). The chain ready is what
    // triggers downstream execution.
    if (!kernelOp.isThrows()) {
      setDelInsertionPoint(builder, kernelBody, terminator);
      insertAsyncDelOps(builder, loc, argsToDelete);
      return;
    }

    // For throwing kernels, we need AsyncDelOp in BOTH the try and except
    // blocks to ensure cleanup happens before signaling ready/error to
    // downstream consumers, regardless of whether an exception is thrown.

    // Move all ops inside the Try clause.
    builder.setInsertionPointToStart(&kernelBody);
    auto tryOp =
        MOGG::TryOp::create(builder, loc, terminator->getOperandTypes());
    Block &tryBlock = tryOp.getTryRegion().emplaceBlock();
    for (Operation &op :
         llvm::make_early_inc_range(kernelBody.getOperations())) {
      if (!isa<MOGG::TryOp, MOGG::OutputOp>(op))
        op.moveBefore(&tryBlock, tryBlock.end());
    }

    // Insert AsyncDelOp in try block before AsyncReadyOp (or at the end if
    // no AsyncReadyOp found).
    setDelInsertionPoint(builder, tryBlock, /*fallback=*/nullptr);
    insertAsyncDelOps(builder, loc, argsToDelete);

    builder.setInsertionPointToEnd(&tryBlock);
    auto tryYieldOp =
        MOGG::YieldOp::create(builder, loc, terminator->getOperands());

    // The except block returns a chain error and dummy values for other
    // outputs.
    Block &exceptBlock = tryOp.getExceptRegion().emplaceBlock();
    Value errArg =
        exceptBlock.addArgument(MOGG::ErrorType::get(loc->getContext()), loc);
    builder.setInsertionPointToStart(&exceptBlock);

    // Insert AsyncDelOp in except block too - cleanup must happen before
    // signaling error to downstream.
    insertAsyncDelOps(builder, loc, argsToDelete);

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
};

} // namespace

} // namespace M::MOGG
