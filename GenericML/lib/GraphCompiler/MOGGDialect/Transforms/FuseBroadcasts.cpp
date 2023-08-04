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
#include "GenericML/GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/Utils.h"
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
    MLIRContext *ctx = graph.getContext();

    for (MOGG::KernelOp kernel : graph.getOps<MOGG::KernelOp>()) {
      // Only allow fusion if the kernel exposes an input lambda.
      // This isn't mechanically necessary. Everything will still work without
      // this. However the current story is that a kernel without a lambda will
      // disallow fusion on those inputs. We might want to make this more fine
      // grained in the future.
      if (!kernel.getInputFusionInterfaceAttr())
        continue;

      // Also avoid view ops into view ops for now.
      if (kernel.getIsViewOp().value_or(false))
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

    auto replaceArg = [](KernelOp kernel, size_t index, Value newVal) {
      Block &thisBlock = kernel.getBody().front();
      thisBlock.getArgument(index).replaceAllUsesWith(newVal);
      kernel.removeOperandAtIndex(index);
    };

    auto hasCompoundParameters = [](MO::TensorType t) {
      for (TypedAttr symbolicDim : t.getDims()) {
        if (!(isa<KGEN::ParamDeclRefAttr>(symbolicDim) ||
              isa<IntegerAttr>(symbolicDim)))
          return true;
      }
      return false;
    };

    // Once we have fused the views we can fuse reshapes into the kernel. This
    // is technically a view but it must come last as it only can reshape
    // contigious inputs and can always be fused regardless of whether or not
    // the kernel has opted into fusion or not as it returns a contigious tensor
    // with the same number of elements.
    for (MOGG::KernelOp kernel : graph.getOps<MOGG::KernelOp>()) {
      // Don't fuse kernels with parameter decls.
      if (!kernel.getOutputParamDecls().empty())
        continue;

      int32_t preFuseNumOperands = kernel.getNumOperands();

      // For safety we check which have already been fused and won't fuse
      // something again.
      DenseMap<Value, Value> alreadyFused;

      // Start from the end so we can safely remove and append arguments.
      for (int32_t index = preFuseNumOperands - 1; index >= 0; --index) {
        // Get the creator of this operand.
        Value operand = kernel.getOperand(index);
        Operation *op = operand.getDefiningOp();

        auto itr = alreadyFused.find(operand);
        if (itr != alreadyFused.end()) {
          replaceArg(kernel, index, itr->second);
          continue;
        }

        if (auto reshape = dyn_cast_or_null<MO::Static::ReshapeOp>(op)) {
          MO::TensorType outType = reshape.getResult().getType();
          MO::TensorType inType = reshape.getInput().getType();

          // We can't fuse this if there are any fancy symbolic dims.
          if (hasCompoundParameters(inType) || hasCompoundParameters(outType))
            continue;

          // Insert after the output placeholders.
          mlir::OpBuilder builder{ctx};
          kernel.setInsertionPointToTop(builder);

          // Add the old reshape argument to this kernel.
          Value reshapeInput = kernel.addOperand(reshape.getInput());

          // Add the new reshape.
          MOGG::CallOp newReshape = createMOGGReshape(builder, reshape.getLoc(),
                                                      reshapeInput, outType);

          // Replace all uses of the old operand and then remove it from this
          // kernel.
          replaceArg(kernel, index, newReshape->getResult(0));

          alreadyFused[reshape.getResult()] = newReshape->getResult(0);
        }
      }
    }
  }

  /// Return the index of the input to fuse or -1 if nothing can be fused.
  bool canFuseOperand(int32_t index, MOGG::KernelOp kernel) {
    // Block arguments have no defining op and obviously cant be fused.
    Operation *edge = kernel.getOperand(index).getDefiningOp();
    if (!edge)
      return false;

    ArrayAttr inputFusions = kernel.getInputFusionInterfaceAttr();
    // Unreachable unsafe state.
    if (inputFusions.size() != kernel.getNumOperands())
      return false;

    // Skip empty input lambdas.
    if (cast<StringAttr>(inputFusions[index]).strref() == "")
      return false;

    // Only fuse within the same region and not across, i.e, control flow.
    if (edge->getParentRegion() != kernel->getParentRegion())
      return false;
    if (auto otherKernel = dyn_cast<KernelOp>(edge)) {
      // Should be unreachable.
      if (otherKernel.getOutputPlaceholder() == nullptr)
        return false;

      // Only fuse view operations.
      if (otherKernel.getIsViewOp().value_or(false))
        return true;
    }
    return false;
  }

  // Fuse at the given index.
  void fuseInputAtIndex(MOGG::KernelOp leaf, int32_t idx,
                        DenseMap<Value, Value> &alreadyFused) {
    IRMapping remapper;
    Value operandBeingReplaced = leaf.getOperand(idx);
    mlir::SymbolTable symTable{leaf};
    llvm::StringMap<StringRef> oldNameToNew;
    KernelOp producerKernel =
        cast<KernelOp>(operandBeingReplaced.getDefiningOp());
    Block &producerBlock = producerKernel.getBody().front();
    Block &leafBlock = leaf.getBody().front();

    // A view op kernel should contain a view transforming operation followed by
    // a copy into the output. When we fuse we just hoist the view operation and
    // not the copy.
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
      if (!alreadyHasOperand)
        remapper.map(otherBlockArg, leaf.addOperand(otherOperand));

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

    mlir::OpBuilder builder{leaf.getContext()};
    // Insert at the top just after the output placeholders.
    leaf.setInsertionPointToTop(builder);

    // Don't fuse in the elementwise kernel invoke or any lambdas. We just
    // want the view ops at the top.
    auto skipElementwiseAndLambdas = [](Operation *op) {
      if (isa<MOGG::LambdaOp>(op))
        return true;
      if (auto call = dyn_cast<MOGG::CallOp>(op)) {
        if (call.getMojoKernelTarget() == "elementwise")
          return true;
      }
      return false;
    };

    // Copy the body into the new one.
    producerKernel.fuseBodyIntoNewWithSkip(
        builder, symTable, remapper, oldNameToNew, skipElementwiseAndLambdas);

    // Remove the argument and replace all uses of it with the broadcast.
    Operation *newView = remapper.lookup(viewOp);

    // For incredibly contrived tests we need to check we are in dominator
    // order. I.E %0 = bcast, %1 = slice(..., %0), %2 = some_kernel(%1, %0)
    for (Value operand : newView->getOperands()) {
      Operation *definingOp = operand.getDefiningOp();
      if (definingOp && newView->isBeforeInBlock(definingOp))
        newView->moveAfter(definingOp);
    }

    leafBlock.getArgument(idx).replaceAllUsesWith(newView->getResult(0));
    leaf.removeOperandAtIndex(idx);
    alreadyFused[operandBeingReplaced] = newView->getResult(0);
  }
};

} // namespace

} // namespace M::MOGG
