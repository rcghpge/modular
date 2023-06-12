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

#include "mlir/IR/IRMapping.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEBROADCASTS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct FuseBroadcastsPass
    : public impl::FuseBroadcastsBase<FuseBroadcastsPass> {
  using FuseBroadcastsBase::FuseBroadcastsBase;

  void runOnOperation() override {
    MO::GraphOp graph = getOperation();

    for (MOGG::KernelOp kernel : graph.getOps<MOGG::KernelOp>()) {
      // Skip non elementwise kernels. We are fusing broadcasts *into* those
      // kernels.
      if (!kernel.getFusionBodyLambda() || kernel.getIsViewOp().value_or(false))
        continue;

      // For catching the rare case of a broadcast taking a broadcast as arg AND
      // this kernel taking both as arg.
      DenseMap<Value, Value> alreadyFused;

      // Fuse broadcasts into this block. We only iterate over one layer. I.E we
      // don't fuse the inputs of the broadcasts as well due to the present
      // danger.
      int32_t preFuseNumOperands = kernel.getNumOperands();

      // Traverse from the back to simplify removing operands.
      for (int32_t index = preFuseNumOperands - 1; index >= 0; --index) {
        if (!canFuseOperand(index, kernel))
          continue;

        // All legality checking should be performed above so we assume this
        // will perform the fusion or produce a fatal failure.
        fuseInputAtIndex(kernel, index, alreadyFused);
      }
    }
  }

  /// Return the index of the input to fuse or -1 if nothing can be fused.
  bool canFuseOperand(int32_t index, MOGG::KernelOp kernel) {
    // Block arguments have no defining op and obviously cant be fused.
    Operation *edge = kernel.getOperand(index).getDefiningOp();
    if (!edge)
      return false;

    // Only fuse within the same region and not across, i.e, control flow.
    if (edge->getParentRegion() != kernel->getParentRegion())
      return false;

    if (auto otherKernel = dyn_cast<KernelOp>(edge)) {
      // Only fuse view operations.
      if (otherKernel.getIsViewOp().value_or(false))
        return true;
    }
    return false;
  }

  // Fuse at the given index and return the pair of operands used in broadcast.
  void fuseInputAtIndex(MOGG::KernelOp leaf, int32_t idx,
                        DenseMap<Value, Value> &alreadyFused) {
    IRMapping remapper;

    Value operandBeingReplaced = leaf.getOperand(idx);

    KernelOp producerKernel =
        cast<KernelOp>(operandBeingReplaced.getDefiningOp());
    Block &producerBlock = producerKernel.getBody().front();
    Block &leafBlock = leaf.getBody().front();

    // Make the outputPlaceholders remap to each other. The semantics of the
    // broadcast are such that a broadcast can be fused an operation when the
    // output matches the target. Therefor a broadcast to the output is
    // semantically the same.
    MOGG::OutputPlaceholdersOp leafOuts;
    for (auto ph : leaf.getOps<MOGG::OutputPlaceholdersOp>()) {
      assert(!leafOuts && "Only one placeholder expected");
      leafOuts = ph;
    }

    MOGG::OutputPlaceholdersOp producerOuts;
    for (auto ph : producerKernel.getOps<MOGG::OutputPlaceholdersOp>()) {
      assert(!producerOuts && "Only one placeholder expected");
      producerOuts = ph;
    }

    // Remap the outputs.
    for (auto [leafOut, producerOut] :
         llvm::zip(leafOuts.getResults(), producerOuts.getResults())) {
      remapper.map(producerOut, leafOut);
    }

    // A view op kernel should contain a view transforming operation followed by
    // a copy into the output. When we fuse we just hoist the view operation and
    // not the copy. This is to find that view operation.
    MOGG::CallOp viewOp;

    // Add the operands of the old kernel to the new one.
    for (size_t i = 0, e = producerKernel->getNumOperands(); i < e; ++i) {
      // Add operand to the kernel.
      Value otherOperand = producerKernel->getOperand(i);
      Value otherBlockArg = producerBlock.getArgument(i);

      bool alreadyHasOperand = false;

      // Check if we already have this value as input to avoid adding multiple
      // copies.
      for (size_t leafI = 0, e = leaf->getNumOperands(); leafI < e; ++leafI) {
        if (otherOperand == leaf->getOperand(leafI)) {
          alreadyHasOperand = true;
          remapper.map(otherBlockArg, leafBlock.getArgument(leafI));
          break;
        }
      }

      // If we already fused this argument then just reuse it.
      auto itr = alreadyFused.find(otherOperand);
      if (itr != alreadyFused.end()) {
        alreadyHasOperand = true;
        remapper.map(otherBlockArg, itr->second);
      }

      // Add if we don't already have an operand for this.
      if (!alreadyHasOperand) {
        leaf->insertOperands(leaf->getNumOperands(), ValueRange{otherOperand});

        Value leafBlockArg = leafBlock.addArgument(otherBlockArg.getType(),
                                                   otherBlockArg.getLoc());
        remapper.map(otherBlockArg, leafBlockArg);
      }

      // The mogg kernel has been marked as a view. Search the kernel for the
      // broadcast function.
      for (Operation *user : otherBlockArg.getUsers()) {
        auto call = dyn_cast<MOGG::CallOp>(user);
        if (call) {
          if (viewOp) {
            assert(viewOp == call &&
                   "More than one view op in kernel being fused.");
          }
          viewOp = call;
        }
      }
    }

    assert(viewOp && "Kernel was marked as a view, it must have a tensor "
                     "semantic view operation.");

    // Insert at the top just after the output placeholders.
    mlir::OpBuilder builder{leafOuts->getContext()};
    builder.setInsertionPointAfter(leafOuts);
    mlir::Operation *insertPoint = nullptr;

    auto newOp = viewOp->clone(remapper);

    // If the tensor isn't a kernel input we need to insert after the last
    // occuring input.
    for (Value operand : newOp->getOperands()) {
      Operation *definingOp = operand.getDefiningOp();
      if (definingOp == nullptr)
        continue;

      // If we don't have an insert point use this otherwise check the order.
      if (!insertPoint || definingOp->isBeforeInBlock(insertPoint))
        insertPoint = definingOp;
    }

    if (insertPoint)
      builder.setInsertionPointAfter(insertPoint);

    builder.insert(newOp);
    // Remove the argument and replace all uses of it with the broadcast.
    leaf->eraseOperand(idx);
    leafBlock.getArgument(idx).replaceAllUsesWith(newOp->getResult(0));
    leafBlock.eraseArgument(idx);

    alreadyFused[operandBeingReplaced] = newOp->getResult(0);
  }
};

} // namespace

} // namespace M::MOGG
