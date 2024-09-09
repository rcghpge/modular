//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Fuses connected GenericCompiledRegion.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"

#include "GraphRT/GRTDialect/GRTOps.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_REMOVEREDUNDANTOPERANDS
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

namespace M::MGP {

namespace {

// This pass will remove any redundant operands within a
// generic-compiled region and collapse the usage.  This reduces the number of
// async values crossing the generic-region boundary
struct RemoveRedundantOperands
    : public MGP::impl::RemoveRedundantOperandsBase<RemoveRedundantOperands> {
  using RemoveRedundantOperandsBase::RemoveRedundantOperandsBase;

  void runOnOperation() override {
    auto modelOp = getOperation();
    auto ctx = modelOp->getContext();
    OpBuilder builder(ctx);
    modelOp.walk([&](MOGG::GenericCompiledRegionOp regionOp) {
      SmallDenseMap<Value, Value> operandToFirstArg;
      llvm::BitVector operandsToRemove(regionOp.getNumOperands());
      // Collapse all OpOperands that are duplicates and collapse all arguments
      // within region to use the same one, go in reverse order
      for (auto &operand : llvm::reverse(regionOp->getOpOperands())) {
        auto arg =
            regionOp.getBody().front().getArgument(operand.getOperandNumber());
        if (operandToFirstArg.contains(operand.get())) {
          // Replace all uses of argument, then remove it from the body.  Store
          // the OpOperand to remove after the fact
          arg.replaceAllUsesWith(operandToFirstArg[operand.get()]);
          regionOp.getBody().eraseArgument(operand.getOperandNumber());
          operandsToRemove.set(operand.getOperandNumber());
        } else
          operandToFirstArg[operand.get()] = arg;
      }

      regionOp->eraseOperands(operandsToRemove);

      regionOp.setInOutSignatureAttr(
          regionOp.getInOutSignature().remove(operandsToRemove));
    });
  }
};

} // namespace

} // namespace M::MGP
