//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/Support/MOGGTypeTraits.h"
#include "GenericML/Support/Tensor.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSECONSTANTS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// The max number of elements in a constant that we will allow to be fused into
// the body of a kernel.
constexpr int64_t kMaxConstantSize = 2048;

struct FuseConstants : public impl::FuseConstantsBase<FuseConstants> {
  using FuseConstantsBase::FuseConstantsBase;

  void runOnOperation() override {
    MO::GraphOp graph = getOperation();

    for (MOGG::KernelOp kernel : graph.getOps<MOGG::KernelOp>()) {
      int32_t preFuseNumOperands = kernel.getNumOperands();

      // Traverse from the back to simplify removing operands.
      for (int32_t index = preFuseNumOperands - 1; index >= 0; --index) {
        if (canFuseOperand(kernel, index))
          fuseInputAtIndex(kernel, index);
      }
    }
  }

  /// We only fuse constants here.
  bool canFuseOperand(MOGG::KernelOp kernel, int32_t index) {
    Operation *edge = kernel.getOperand(index).getDefiningOp();
    if (!edge)
      return false;

    // Check it is a small constant with a attribute we understand.
    if (auto cst = dyn_cast<MO::ConstantOp>(edge)) {
      ElementsAttr elems = cst.getValueAttr();
      if (elems.getNumElements() > kMaxConstantSize)
        return false;

      Type dtype = elems.getElementType();
      if (!MOGGTypeTraits::isSupportedMOGGType(dtype))
        return false;

      if (!isa<M::ArrayElementsAttr, DenseResourceElementsAttr>(elems))
        return false;

      return true;
    }

    return false;
  }

  /// Fold the constant into the operation.
  void fuseInputAtIndex(MOGG::KernelOp kernel, int32_t idx) {
    auto cst = cast<MO::ConstantOp>(kernel.getOperand(idx).getDefiningOp());

    Block &block = kernel.getBody().front();

    mlir::OpBuilder builder{kernel->getContext()};

    // We insert after the output placeholders
    MOGG::OutputPlaceholdersOp outputPh;
    for (auto ph : kernel.getOps<MOGG::OutputPlaceholdersOp>()) {
      assert(!outputPh && "Only one placeholder expected");
      outputPh = ph;
    }

    // Not everything has to have an input placeholder, e.g a shape function.
    if (outputPh)
      builder.setInsertionPointAfter(outputPh);
    else
      builder.setInsertionPointToStart(&block);

    // Clone the constant and add it to the kernel.
    auto newOp = cst->clone();
    builder.insert(newOp);

    // Replace all uses of it with the now in-kernel constant.
    block.getArgument(idx).replaceAllUsesWith(newOp->getResult(0));

    // Remove the argument.
    kernel.removeOperandAtIndex(idx);

    // Kill the constant if that was the last use.
    if (cst->use_empty())
      cst->erase();
  }
};

} // namespace

} // namespace M::MOGG
