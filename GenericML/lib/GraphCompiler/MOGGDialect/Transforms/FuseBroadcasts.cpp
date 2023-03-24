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

      bool hasFused = false;

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
        hasFused = true;
      }

      // Fix up the attributes on this kernel.
      if (hasFused) {
        Block &leafBlock = kernel.getBody().front();

        MOGG::CallOp bcast;

        // Find a broadcast.
        for (Value arg : leafBlock.getArguments()) {
          for (Operation *user : arg.getUsers()) {
            auto call = dyn_cast<MOGG::CallOp>(user);
            if (call && call.getLitKernelTarget() == MOGG::BroadcastToTensorOp::getOperationName()) {
              // At least one of the broadcasts must use only the inputs. That
              // is the one we want as we need to tell the shape inference
              // function which of our inputs can be used to deduce the shape.
              if (call->getOperand(0).getDefiningOp() == nullptr &&
                  call->getOperand(1).getDefiningOp() == nullptr) {
                bcast = call;
                break;
              }
            }
          }

          // We only need one.
          if (bcast)
            break;
        }

        assert(bcast && "Broadcast not found but fusion was performed.");

        // Block all fusion on this operation going forward for safety.
        kernel.removeInputFusionInterfaceAttr();
        kernel.removeOutputFusionInterfaceAttr();
        kernel.removeFusionBodyLambdaAttr();

        // The indices of the original tensor and the target tensor of the
        // broadcast. These are needed for the shape inference function.
        int32_t i = 0, fromIndex = -1, toIndex = -1;
        for (Value arg : leafBlock.getArguments()) {
          if (bcast->getOperand(0) == arg)
            fromIndex = i;
          else if (bcast->getOperand(1) == arg)
            toIndex = i;

          i++;
        }

        assert(fromIndex != -1 && "From index could not be identified.");
        assert(toIndex != -1 && "To index could not be identified.");

        mlir::OpBuilder builder{kernel.getContext()};

        // Tell the shape function which of the indices should be used to deduce
        // the shape.
        kernel.setBcastInputIndicesAttr(
            builder.getI32ArrayAttr({fromIndex, toIndex}));
        kernel.setShapeFunc("broadcast_to_tensor");
      }
    }
  }

  /// Return the index of the input to fuse or -1 if nothing can be fused.
  bool canFuseOperand(int32_t index, MOGG::KernelOp kernel) {
    // Block arguments have no defining op and obviously cant be fused.
    Operation *edge = kernel.getOperand(index).getDefiningOp();
    if (!edge)
      return false;

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

    MOGG::CallOp bcast;

    // Add the operands of the old kernel to the new one.
    for (size_t i = 0, e = producerKernel->getNumOperands(); i < e; ++i) {
      // Add operand to the kernel.
      Value otherOperand = producerKernel->getOperand(i);
      Value otherBlockArg = producerBlock.getArgument(i);

      bool alreadyHasOperand = false;

      // Check if we already take this value as input to avoid adding multiple
      // copies.
      for (size_t leafI = 0, e = leaf->getNumOperands(); leafI < e; ++leafI) {
        if (otherOperand == leaf->getOperand(leafI)) {
          alreadyHasOperand = true;
          remapper.map(otherBlockArg, leafBlock.getArgument(leafI));
          break;
        }
      }

      // If we already fused the argument
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

      // The mogg kernel has been marked as a view, this means one of the arguments must be broadcasted. Search the kernel for the broadcast function.
      for (Operation *user : otherBlockArg.getUsers()) {
        auto call = dyn_cast<MOGG::CallOp>(user);
        if (call && call.getLitKernelTarget() == MOGG::BroadcastToTensorOp::getOperationName()) {
          if (bcast) {
            assert(bcast == call &&
                   "More than one broadcast in kernel being fused.");
          }
          bcast = call;
        }
      }
    }

    assert(bcast && "Kernel was marked as a view, it must have a broadcast");

    mlir::OpBuilder builder{&leafBlock, leafBlock.begin()};
    mlir::Operation *insertPoint = nullptr;

    auto newOp = bcast->clone(remapper);

    // If the tensor isn't a kernel input we need to insert after the input of
    // this kernel.
    if (newOp->getOperand(0).getDefiningOp() != nullptr)
      insertPoint = newOp->getOperand(0).getDefiningOp();

    auto rhs = newOp->getOperand(1).getDefiningOp();
    if (rhs != nullptr) {
      if (!insertPoint || !rhs->isBeforeInBlock(insertPoint))
        insertPoint = rhs;
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
