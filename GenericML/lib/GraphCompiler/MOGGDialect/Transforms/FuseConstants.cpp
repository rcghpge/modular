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
#include "GenericML/Support/Tensor.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSECONSTANTS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

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
      auto attr = dyn_cast<M::ArrayElementsAttr>(cst.getValueAttr());
      if (!attr)
        return false;

      if (!attr.getElementType().isF32())
        return false;

      if (attr.getNumElements() > 16)
        return false;

      return true;
    }

    return false;
  }

  /// Fold the constant into the operation.
  void fuseInputAtIndex(MOGG::KernelOp kernel, int32_t idx) {
    auto cst = cast<MO::ConstantOp>(kernel.getOperand(idx).getDefiningOp());

    Block &block = kernel.getBody().front();

    // We insert after the output placeholders
    MOGG::OutputPlaceholdersOp outputPh;
    for (auto ph : kernel.getOps<MOGG::OutputPlaceholdersOp>()) {
      assert(!outputPh && "Only one placeholder expected");
      outputPh = ph;
    }
    mlir::OpBuilder builder{outputPh->getContext()};
    builder.setInsertionPointAfter(outputPh);

    // Clone the constant and add it to the kernel.
    auto newOp = cst->clone();
    builder.insert(newOp);

    // Replace all uses of it with the now in-kernel constant.
    block.getArgument(idx).replaceAllUsesWith(newOp->getResult(0));

    bool canDropArg = true;

    // This is really stupid and can go away when we have shape inference on by
    // default. Without shape inference we need to carry around a shape function
    // for the bcast which looks at some of the operands. Because we remove
    // operands we have to: 1. not remove an operand used in the bcast. and 2.
    // update the indices if needed to reflect the new position of that operand.
    std::optional<ArrayAttr> bcastShapeInd = kernel.getBcastInputIndicesAttr();
    if (bcastShapeInd && *bcastShapeInd && !bcastShapeInd->empty()) {
      auto op1 = cast<IntegerAttr>((*bcastShapeInd)[0]);
      auto op2 = cast<IntegerAttr>((*bcastShapeInd)[1]);
      int32_t original1 =
          static_cast<int32_t>(op1.getValue().getLimitedValue());
      int32_t original2 =
          static_cast<int32_t>(op2.getValue().getLimitedValue());

      llvm::SmallVector<int32_t> indices{original1, original2};

      // Firstly if either are the this index we can't drop it.
      if (indices[0] == idx || indices[1] == idx) {
        canDropArg = false;
      } else {
        // Otherwise we need to shift them to the left to account for the
        // removed operand.
        if (indices[0] > idx)
          --indices[0];
        if (indices[1] > idx)
          --indices[1];

        if (indices[0] != original1 || indices[1] != original2)
          kernel.setBcastInputIndicesAttr(builder.getI32ArrayAttr(indices));
      }
    }

    if (canDropArg) {
      // Remove the argument.
      kernel->eraseOperand(idx);
      block.eraseArgument(idx);

      // Kill the constant if that was the last use.
      if (cst->use_empty())
        cst->erase();
    }
  }
};

} // namespace

} // namespace M::MOGG
