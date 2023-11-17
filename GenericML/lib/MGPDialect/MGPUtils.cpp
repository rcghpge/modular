//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPUtils.h"

#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTypes.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/ErrorHandling.h"

using namespace M;
using namespace MGP;

namespace {
/// Name of attribute holding the 'temporary' #M.inout_sig.
constexpr StringRef kInOutSigAttribute = "tempInOutSig";
} // namespace

TypedValue<MGP::ContextType> MGP::getContextForBlock(Block *block) {
  while (true) {
    // First look for a block argument.
    for (Value arg : block->getArguments()) {
      if (isa<MGP::ContextType>(arg.getType()))
        return cast<TypedValue<MGP::ContextType>>(arg);
    }
    // Next look for a mgp.context.create op.
    for (auto createOp : block->getOps<MGP::ContextCreateOp>()) {
      return createOp.getResult();
    }
    // Next, try one level up.
    assert(!isa<ModelOp>(block->getParentOp()) &&
           "unable to locate an !mgp.context");
    block = block->getParentOp()->getBlock();
  }
  assert(false && "unable to locate an !mgp.context");
}

void MGP::setInOutSignature(Operation *op, InOutSignatureAttr inOutSig) {
  assert(!isa<MGPInOutSemanticsOpInterface>(op) &&
         "op already implements MGPInOutSemanticsOpInterface");
  assert(inOutSig.size() == op->getNumOperands() &&
         "mismatched operation and in/out signature arity");
  op->setAttr(kInOutSigAttribute, inOutSig);
}

void MGP::removeInOutSignature(Operation *op) {
  op->removeAttr(kInOutSigAttribute);
}

InOutSignatureAttr::InOutSemantics
MGP::getInOutSemantics(OpOperand &opOperand) {
  Operation *op = opOperand.getOwner();

  // Ops with ODS definitions (the majority) convey in/out semantics via the
  // interface.
  if (auto inOutSemanticsOp = dyn_cast<MGPInOutSemanticsOpInterface>(op))
    return inOutSemanticsOp.getInOutSemantics(opOperand.getOperandNumber());

  // All other ops may supply an #M.InOutSignature directly.
  if (auto inOutAttr =
          op->getAttrOfType<InOutSignatureAttr>(kInOutSigAttribute)) {
    assert(inOutAttr.size() == op->getNumOperands() &&
           "mismatched operation and in/out signature arity");
    return inOutAttr[opOperand.getOperandNumber()];
  }

  // Assume 'in' only.
  return InOutSignatureAttr::kIn;
}

bool MGP::isMutated(OpOperand &opOperand) {
  InOutSignatureAttr::InOutSemantics inOut = getInOutSemantics(opOperand);
  return inOut == InOutSignatureAttr::kOut || inOut == InOutSignatureAttr::kMut;
}

bool MGP::isRead(OpOperand &opOperand) {
  InOutSignatureAttr::InOutSemantics inOut = getInOutSemantics(opOperand);
  return inOut == InOutSignatureAttr::kIn || inOut == InOutSignatureAttr::kMut;
}
