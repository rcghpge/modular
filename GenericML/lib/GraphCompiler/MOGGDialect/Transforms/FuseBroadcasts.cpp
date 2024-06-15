//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "ArraySupport/Tensor.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENUtils.h"
#include "Support/AssertStream.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
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
      bool isView = kernel.getIsViewOp().value_or(false);

      // Only allow fusion if the kernel exposes an input lambda or is a view.
      // This isn't mechanically necessary. Everything will still work without
      // this. However the current story is that a kernel without a lambda will
      // disallow fusion on those inputs. We might want to make this more fine
      // grained in the future.
      if (!kernel.getInputFusionInterfaceAttr() && !isView)
        continue;

      // For catching the rare case of a broadcast taking a broadcast as arg AND
      // this kernel taking both as arg.
      DenseMap<Value, Value> alreadyFused;

      // Fuse broadcasts into this block.
      int32_t index = kernel.getNumOperands() - 1;
      bool fusionOccurred = false;

      // Traverse from the back to simplify removing operands.
      while (index >= 0) {
        if (!canFuseOperand(index, kernel)) {
          index--;
          continue;
        }

        // All legality checking should be performed above so we assume this
        // will perform the fusion or produce a fatal failure.
        fuseInputAtIndex(kernel, index, alreadyFused);
        index = kernel.getNumOperands() - 1;
        fusionOccurred = true;
      }

      // Strip all temporarily added fusion hooks.
      if (fusionOccurred && !isView)
        kernel.removeTemporaryFusionMarks();
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

      // Don't fuse reshapes into new API kernels because we will handle it in
      // another pass.
      if (kernel.getIsKGEN().value_or(false))
        continue;

      if (kernel.isInPlaceAlloc())
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
          MOGG::CallOp newReshape =
              createMOGGReshape(builder, reshape.getLoc(), reshapeInput,
                                outType, "ndbuffer_reshape");

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

    if (kernel.isInPlaceAlloc())
      return false;

    // Currently view ops don't have to have the input fusion interface.
    if (!kernel.getIsViewOp().value_or(false)) {
      ArrayAttr inputFusions = kernel.getInputFusionInterfaceAttr();
      // Unreachable unsafe state.
      if (inputFusions.size() != kernel.getNumOperands())
        return false;

      // Skip empty input lambdas.
      if (cast<StringAttr>(inputFusions[index]).strref().empty())
        return false;
    }

    // Only fuse within the same region and not across, i.e, control flow.
    if (edge->getParentRegion() != kernel->getParentRegion())
      return false;

    if (auto otherKernel = dyn_cast<KernelOp>(edge)) {
      // Should be unreachable.
      if (otherKernel.getOutputPlaceholder() == nullptr)
        return false;

      // Make sure both kernels are compatible regarding KGEN.
      if (kernel.getIsKGENAttr() != otherKernel.getIsKGENAttr())
        return false;

      // For KGEN view fusion, the src kernel must only have 1 output.
      if (kernel.getIsKGEN().value_or(false) &&
          otherKernel.getNumResults() != 1)
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
    KernelOp producerKernel =
        cast<KernelOp>(operandBeingReplaced.getDefiningOp());
    Block &producerBlock = producerKernel.getBody().front();
    Block &leafBlock = leaf.getBody().front();

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
        remapper.map(otherBlockArg, leaf.addOperand(otherOperand));

        // In order to "trick" the fusion algorithm to run on the inputs of the
        // thing we have just fused in we pretend it has a fusion lambda. We
        // will remove this at the end this is just so the fusion runs again.
        // This is so we can distinguish between views which we know we can
        // fuse again and ops which have opted out of fusion.
        if (!leaf.getIsViewOp().value_or(false)) {
          leaf.setInputFusionInterface(leaf.getNumOperands() - 1,
                                       KernelOp::getTempLambdaName());
        }
      }
    }

    mlir::OpBuilder builder{leaf.getContext()};
    // Insert at the top just after the output placeholders.
    leaf.setInsertionPointToTop(builder);

    // Fusion helper function for KGEN version.
    auto fuseInputImplKGEN = [&](KernelOp producerKernel, KernelOp leaf,
                                 Value operand) {
      stageKernelsForFusion({producerKernel, leaf});
      Value target = leafBlock.getArgument(idx);
      llvm::StringMap<TypedAttr> paramMapping;

      MOGG::ToKGENOp producerKernelToKGEN = nullptr;
      for (Operation &op : producerKernel.getOps()) {
        if (isa<MOGG::OutputPlaceholdersOp>(op)) {
          remapper.map(op.getResult(0), target);
          // Find the ToKGEN of the producer kernel.
          for (auto user : op.getUsers()) {
            ASSERT_STREAM(!producerKernelToKGEN,
                          << "Expecting a single ToKGENOp");
            if (auto toKGENOp = dyn_cast<MOGG::ToKGENOp>(user))
              producerKernelToKGEN = toKGENOp;
          }
        } else if (isa<MOGG::OutputOp>(op)) {
          // Do nothing.
        } else
          MOGG::cloneKgenOpsIntoMOGG(builder, &op, paramMapping, remapper);
      }
      ASSERT_STREAM(producerKernelToKGEN,
                    << "Could not find ToKGENOp for producer kernel");

      // We find the mogg.to_kgen that relates to the output of the
      // `producerKernel`.
      llvm::SmallPtrSet<Operation *, 2> toDelete;
      llvm::DenseSet<Value> kgenOut;
      leaf.walk([&](MOGG::ToKGENOp toKgen) {
        if (toKgen.getInput() == target) {
          kgenOut.insert(toKgen.getResults());
          toDelete.insert(toKgen);
        }
      });

      // We find the pop.stack_allocation that relates to the output of the
      // `producerKernel`.
      Value stackAlloc;
      leaf.walk([&](MOGG::TensorCopyOp tensorCopy) {
        if (kgenOut.contains(tensorCopy.getDst())) {
          stackAlloc = tensorCopy.getSrc();
          toDelete.insert(tensorCopy);
        }
      });

      target.replaceAllUsesWith(stackAlloc);
      leaf.removeOperandAtIndex(idx);
      alreadyFused[operand] = stackAlloc;

      Type bindType;
      for (auto value : kgenOut) {
        // If we find a mogg.to_kgen that connects to a mogg.tensor_copy,
        // then it is the one we just cloned into the leaf which requires
        // rebinding of its output type.
        if (llvm::all_of(value.getUsers(), [](Operation *user) {
              return !isa<MOGG::TensorCopyOp>(user);
            })) {
          bindType = producerKernelToKGEN.getType();
        }
      }

      KGEN::RebindOp rebind =
          builder.create<KGEN::RebindOp>(leaf->getLoc(), bindType, stackAlloc);

      // When fusing views, we want to ensure the consumer operation is not
      // assuming contiguous memory layout. To do so, we change the stride
      // information to be statically unknown. It is achieved by replacing old
      // static strides which can be known to the new, unknown own. This assumes
      // LoweringHelper.cpp lowered unknown strides for view op outputs.
      auto newStrides = dyn_cast<KGEN::ParamDeclRefAttr>(
          producerKernelToKGEN.getStridesAttr());
      ASSERT_STREAM(
          newStrides,
          << "Expected strides to be a parameter declaration reference. Maybe "
             "the view kernel was not explicitly parametrized on strides?");
      DenseSet<KGEN::ParamDeclRefAttr> oldStrideParams;
      for (auto value : kgenOut) {
        auto oldStrides = dyn_cast<KGEN::ParamDeclRefAttr>(
            value.getDefiningOp<ToKGENOp>().getStridesAttr());
        ASSERT_STREAM(
            oldStrides,
            << "All kernels should have explicitly parametrized strides");
        oldStrideParams.insert(oldStrides);
      }
      mlir::AttrTypeReplacer walker;
      walker.addReplacement([&](KGEN::ParamDeclRefAttr attr)
                                -> std::optional<KGEN::ParamDeclRefAttr> {
        if (oldStrideParams.contains(attr))
          return newStrides;
        return std::nullopt;
      });
      walker.recursivelyReplaceElementsIn(leaf, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);

      for (auto value : kgenOut)
        value.replaceAllUsesWith(rebind);

      // Clean-up un-necessary mogg.to_kgen and mogg.tensor_copy in the
      // leaf that are cloned from the `producerKernel`.
      for (Operation *op : toDelete)
        op->erase();

      MOGG::remapTypesAndAttrs(leaf, paramMapping);
    };

    // Fusion helper function for MOGG version.
    auto fuseInputImplMOGG = [&](KernelOp producerKernel, KernelOp leaf,
                                 Value operand) {
      mlir::SymbolTable symTable{leaf};
      llvm::StringMap<StringRef> oldNameToNew;
      // Don't fuse in the elementwise kernel invoke or any lambdas. We just
      // want the view ops at the top.
      auto skipElementwiseAndLambdas = [](Operation *op) {
        if (isa<MOGG::LambdaOp, MOGG::OutputPlaceholdersOp>(op))
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

      // Find "The view" being fused in. The view we just fused in will be the
      // operation which has a return but that has no use (since it was
      // previously used in a lambda which we haven't fused). With improved
      // composability in the future we won't need this hack.
      Operation *newView = nullptr;
      for (auto op : leaf.getOps<MOGG::CallOp>()) {
        if (op->getNumResults() == 1 && op->getResult(0).use_empty()) {
          ASSERT_STREAM(
              newView == nullptr,
              << "More than one view op detected in kernel being fused.");
          newView = op;
        }
      }
      ASSERT_STREAM(
          newView,
          << "Kernel was marked as a view but could not find a view op");

      leafBlock.getArgument(idx).replaceAllUsesWith(newView->getResult(0));
      leaf.removeOperandAtIndex(idx);
      alreadyFused[operand] = newView->getResult(0);
    };

    if (leaf.getIsKGEN().value_or(false)) {
      fuseInputImplKGEN(producerKernel, leaf, operandBeingReplaced);
    } else {
      fuseInputImplMOGG(producerKernel, leaf, operandBeingReplaced);
    }

    // Ensure the block is in topologically sorted order. This is needed for
    // somewhat contrived test cases.
    mlir::sortTopologically(&leafBlock);

    // Update the leaf's summary to capture the producer's summary.
    leaf.setSummaryAttr(getSummaryString({producerKernel, leaf}));
  }
};

} // namespace

} // namespace M::MOGG
