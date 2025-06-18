//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "GenericML/GraphCompiler/Analysis/MODialect/ParametricTopologyAnalysis.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MOToKGENLowering.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MOToMOGGLowering.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoTargetFunctions.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Runtime/Tensor/Tensor.h"
#include "Support/AssertStream.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/IRMapping.h"
using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEVIEWS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

/**
1. Fuse compatible view ops into other kernels.
2. Lower `mo.static.reshape` ops into a kernel and fuse them into other kernels.
The reason this is done as a separate step is because reshape ops are not
lowered and they must be the last view op fused into a kernel because the kernel
implementation expects contiguous strides. Therefore, bcast -> reshape will
yield invalid results.
3. Raise mogg.kernels that used to be mo.static.reshape back into reshapes. This
step is optional for correctness but necessary for performance. Reshapes, as
with other views, copy data. A later pass, MOToPrimitives, will turn the reshape
into a primitive that does not copy.

All in all, this pass fuses views into other views or kernels that aren't
reshapes. Then, reshapes are fused and the remaining ones will be later lowered
to a fast primitive.

GUARANTEES:
- All remaining reshapes are `mo.static.reshape.
- No view fuses into a reshape kernel.
 */

struct FuseViewsPass : public impl::FuseViewsBase<FuseViewsPass> {
  using FuseViewsBase::FuseViewsBase;

  FuseViewsPass() = default;
  FuseViewsPass(const FuseViewsPass &pass) {}

  Statistic numberOfFusedViews{this, "numberOfFusedViews",
                               "Number of fused views"};

  static bool hasCompoundParameters(MO::TensorType t) {
    for (TypedAttr symbolicDim : t.getDims()) {
      if (!(isa<KGEN::ParamDeclRefAttr>(symbolicDim) ||
            isa<IntegerAttr>(symbolicDim)))
        return true;
    }
    return false;
  };

  void runOnOperationImpl(
      MO::GraphOp graph, MLIRContext *ctx,
      const DenseMap<Operation *, MO::Static::ReshapeOp> &reshapeKernels,
      MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
      TypeManager &typeManager) {

    // TODO(GEX-2300): we do not need the helper lambda after fully migration.
    auto isViewFusible = [](Operation *op) {
      if (auto k = dyn_cast<MOGG::KernelOp>(op)) {
        // Only allow fusion if the kernel exposes an input lambda or is a
        // view. This isn't mechanically necessary. Everything will still
        // work without this. However the current story is that a kernel
        // without a lambda will disallow fusion on those inputs. We might
        // want to make this more fine grained in the future.
        return k.getInputFusionInterfaceAttr() ||
               k.getIsViewOp().value_or(false);
      } else {
        auto ek = cast<MOGG::ExpKernelOp>(op);
        return ek.hasFusibleOperand(true) || ek.isViewOp();
      }
    };

    graph.walk([&](Operation *op) {
      if (!isa<MOGG::KernelOp, MOGG::ExpKernelOp>(op))
        return;

      if (reshapeKernels.contains(op))
        return;

      if (!isViewFusible(op))
        return;

      // For catching the rare case of a broadcast taking a broadcast as arg
      // AND this kernel taking both as arg.
      DenseMap<Value, Value> alreadyFused;

      // Fuse broadcasts into this block.
      int32_t index = op->getNumOperands() - 1;
      bool fusionOccurred = false;

      // Traverse from the back to simplify removing operands.
      while (index >= 0) {
        if (!canFuseOperand(index, op)) {
          index--;
          continue;
        }

        // If we are fusing in reshapes, then we skip things that aren't
        // reshapes. This avoids cases like:
        //   non-reshape -> reshape -> non-reshape
        //     non-reshape -> reshape+non-reshape (first iteration)
        //       non-reshape+reshape+non-reshape  (second iteration)
        if (!reshapeKernels.empty() &&
            !reshapeKernels.contains(op->getOperand(index).getDefiningOp())) {
          index--;
          continue;
        }

        // All legality checking should be performed above so we assume this
        // will perform the fusion or produce a fatal failure.
        fuseInputAtIndex(op, index, alreadyFused, mojoLibraryAnalysis,
                         typeManager);
        index = op->getNumOperands() - 1;
        fusionOccurred = true;
      }

      // Strip all temporarily added fusion hooks.

      if (auto kernel = dyn_cast<MOGG::KernelOp>(op);
          kernel && fusionOccurred && !kernel.getIsViewOp().value_or(false))
        kernel.removeTemporaryFusionMarks();

      ModuleOp topLevelModule = op->getParentOfType<ModuleOp>();
      MOGG::addForwardDeclsOfReferencedFuncs(op, topLevelModule,
                                             mojoLibraryAnalysis);
    });

    // FIXME(GEX-865): This pass is generating invalid IR. Fix it up here.
    getOperation().walk([](KGEN::CallOp call) {
      OpBuilder b(call);
      for (auto [arg, opOperand, type] :
           llvm::zip(call.getOperands(), call->getOpOperands(),
                     call.getCallee().getType().getBody().getArguments())) {
        if (arg.getType() != type) {
          Value rebound = b.create<KGEN::RebindOp>(call.getLoc(), type, arg);
          opOperand.set(rebound);
        }
      }
    });
  }

  /// Return the index of the input to fuse or -1 if nothing can be fused.
  bool canFuseOperand(int32_t index, Operation *op) {
    // Block arguments have no defining op and obviously cant be fused.
    Operation *edge = op->getOperand(index).getDefiningOp();
    if (!edge)
      return false;

    // Only fuse within the same block and not across, i.e, control flow.
    if (edge->getBlock() != op->getBlock())
      return false;

    if (auto expKernel = dyn_cast<MOGG::ExpKernelOp>(op)) {
      if (auto alloc = expKernel.getAllocIndices(); alloc && !alloc.empty())
        return false;

      auto otherKernel = dyn_cast<MOGG::ExpKernelOp>(edge);
      if (!otherKernel)
        return false;

      // Only fuse view operations (when will a view kernel returns more than
      // one result, probably dead code, or should we instead verify it during
      // IR construction?)
      if (!otherKernel.isViewOp() || otherKernel.getNumResults() != 1)
        return false;

      // if the kernel is not a view kernel nor has a view fusible input at the
      // given position.
      if (!expKernel.isViewOp() && !expKernel.isOperandFusible(index, true))
        return false;

      return true;
    }

    // TODO(GEX-2300): dead code after fully migration.
    auto kernel = cast<MOGG::KernelOp>(op);
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

    auto otherKernel = dyn_cast<KernelOp>(edge);
    if (!otherKernel)
      return false;

    // Should be unreachable.
    if (otherKernel.getOutputPlaceholder() == nullptr)
      return false;

    // For KGEN view fusion, the src kernel must only have 1 output.
    if (otherKernel.getNumResults() != 1)
      return false;

    // Only fuse view operations.
    if (!otherKernel.getIsViewOp().value_or(false))
      return false;

    return true;
  }

  void fuseInputAtIndex(MOGG::ExpKernelOp kernel, int32_t idx,
                        DenseMap<Value, Value> &alreadyFused,
                        MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                        TypeManager &typeManager) {
    Value toFuse = kernel.getOperand(idx);
    auto viewProducer = cast<MOGG::ExpKernelOp>(toFuse.getDefiningOp());
    kernel.insertInputsAt(idx + 1, viewProducer.getInputs());

    IRRewriter rewriter(kernel->getContext());
    rewriter.cloneRegionBefore(viewProducer.getExecBody(),
                               &kernel.getExecBody().front());

    Block *src = &kernel.getExecBody().front();
    Block *dst = &kernel.getExecBody().back();
    auto srcTerm = cast<MOGG::CallMaterializeOp>(src->getTerminator());

    ValueRange inlinedArgs = dst->getArguments().slice(
        idx + 1, viewProducer.getExecBody().getNumArguments());

    rewriter.inlineBlockBefore(src, dst, dst->begin(), inlinedArgs);

    // Clean up.
    dst->getArgument(idx).replaceAllUsesWith(srcTerm.getInputs().front());
    kernel.eraseInputsAt(idx);
    srcTerm.erase();

    kernel.setSummaryAttr(
        getSummaryString({viewProducer, kernel}, FusionType::HARD));
  }

  // Fuse at the given index.
  void fuseInputAtIndex(Operation *op, int32_t idx,
                        DenseMap<Value, Value> &alreadyFused,
                        MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                        TypeManager &typeManager) {
    ++numberOfFusedViews;
    if (auto expKernel = dyn_cast<MOGG::ExpKernelOp>(op))
      return fuseInputAtIndex(expKernel, idx, alreadyFused, mojoLibraryAnalysis,
                              typeManager);

    // TODO(GEX-2300): to remove after fully migrated.
    auto leaf = cast<MOGG::KernelOp>(op);
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
        // This is so we can distinguish between views which we know we can fuse
        // again and ops which have opted out of fusion.
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
      llvm::DenseMap<StringAttr, TypedAttr> paramMapping;

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
        } else {
          MOGG::cloneKgenOpsIntoMOGG(builder, &op, paramMapping, remapper);
        }
      }
      ASSERT_STREAM(producerKernelToKGEN,
                    << "Could not find ToKGENOp for producer kernel");

      // Operations in this vector should be in topological order.
      llvm::SmallVector<Operation *> toDeleteTopoOrder;

      // We find the mogg.to_kgen that relates to the output of the
      // `producerKernel`.
      llvm::DenseSet<Value> kgenOut;
      leaf.walk([&](MOGG::ToKGENOp toKgen) {
        if (toKgen.getInput() == target) {
          kgenOut.insert(toKgen.getResults());
          toDeleteTopoOrder.push_back(toKgen);
        }
      });

      KGEN::CallOp viewCopyCallOp;

      Value stackAlloc;
      leaf.walk([&](KGEN::CallOp callOp) {
        StringRef funcName =
            cast<mlir::FlatSymbolRefAttr>(callOp.getCalleeSymbol()).getValue();
        if (funcName.starts_with(Utils::DPSTensorCopy)) {
          viewCopyCallOp = callOp;
          stackAlloc = callOp->getOperand(1);

          // Assumptions, any users of `callOp` can now be safely deleted.
          toDeleteTopoOrder.append(MO::getRootedComponents(callOp));

          // Always remove the first one because in case we fuse a view into a
          // view, we want to maintain the second one.
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      ASSERT_STREAM(stackAlloc != nullptr, "stack view not found !");

      alreadyFused[operand] = stackAlloc;

      KGEN::ParamDeclRefAttr newStaticTensorSpec = nullptr;
      DenseSet<StringAttr> oldStrideParams;
      // Iterating through all the mogg.to_kgen that relate to the output of the
      // view we are fusing in.
      for (auto value : kgenOut) {
        StringAttr oldStaticTensorSpecName =
            value.getDefiningOp<ToKGENOp>().getStaticTensorSpecNameAttr();
        if (oldStaticTensorSpecName)
          oldStrideParams.insert(oldStaticTensorSpecName);
      }

      // Build a new StaticTensorSpec parameter, changing only the strides.
      auto buildTensorSpecWithViewStrides =
          [&](TypedAttr oldTensorSpecttr,
              StringRef oldAttrName) -> ErrorOr<KGEN::ParamDeclRefAttr> {
        // Get The view strides from the viewcopy call.
        constexpr unsigned kViewCopyTensorSpecIndex = 2;
        TypedAttr viewTensorSpecAttr =
            viewCopyCallOp.getParamValues()[kViewCopyTensorSpecIndex];

        auto newParamName = mlir::StringAttr::get(
            viewCopyCallOp.getContext(), oldAttrName + "_MOGG_FUSE_VIEWS");
        auto newParam = KGEN::ParamDeclAttr::get(newParamName,
                                                 viewTensorSpecAttr.getType());
        auto paramDeclarationOp = builder.create<KGEN::ParamDeclareOp>(
            operand.getLoc(), newParam, viewTensorSpecAttr);

        return KGEN::ParamDeclRefAttr::get(
            paramDeclarationOp.getParamDeclAttr());
      };

      mlir::AttrTypeReplacer walker;
      walker.addReplacement([&](KGEN::ParamDeclRefAttr attr)
                                -> std::optional<KGEN::ParamDeclRefAttr> {
        if (oldStrideParams.contains(attr.getName())) {
          ASSERT_STREAM(
              producerKernelToKGEN.getStaticTensorSpecNameAttr() != nullptr,
              << "Expected strides to be a parameter declaration reference. "
                 "Maybe the view kernel was not explicitly parametrized on "
                 "strides?");
          if (newStaticTensorSpec == nullptr) {
            auto specOrErr = buildTensorSpecWithViewStrides(
                producerKernelToKGEN.getStaticTensorRefAttr(), attr.getName());
            ASSERT_STREAM(!specOrErr.isError(), "failed to build the new spec");
            newStaticTensorSpec = specOrErr.takeValue();
          }
          return newStaticTensorSpec;
        }
        return std::nullopt;
      });
      walker.recursivelyReplaceElementsIn(leaf, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);

      // Don't forget to update the annotation.
      if (newStaticTensorSpec) {
        for (auto toKGENOp : leafBlock.getOps<ToKGENOp>()) {
          StringAttr oldTensorSpec = toKGENOp.getStaticTensorSpecNameAttr();
          if (!oldStrideParams.contains(oldTensorSpec))
            continue;

          toKGENOp.setStaticTensorSpecNameAttr(newStaticTensorSpec.getName());
        }
      }

      Type bindType = producerKernelToKGEN.getType();
      KGEN::RebindOp rebind =
          builder.create<KGEN::RebindOp>(leaf->getLoc(), bindType, stackAlloc);
      for (auto value : kgenOut)
        value.replaceAllUsesWith(rebind);

      // Clean-up un-necessary mogg.to_kgen and mogg.tensor_copy in the
      // leaf that are cloned from the `producerKernel`.
      for (Operation *op : llvm::reverse(toDeleteTopoOrder))
        op->erase();

      leaf.removeOperandAtIndex(idx);

      MOGG::remapTypesAndAttrs(leaf, paramMapping);
    };

    fuseInputImplKGEN(producerKernel, leaf, operandBeingReplaced);

    // Ensure the block is in topologically sorted order. This is needed for
    // somewhat contrived test cases.
    mlir::sortTopologically(&leafBlock);

    // Update the leaf's summary to capture the producer's summary.
    leaf.setSummaryAttr(
        getSummaryString({producerKernel, leaf}, FusionType::HARD));
  }

  DenseMap<Operation *, MO::Static::ReshapeOp>
  lowerReshapeOps(MO::GraphOp graph,
                  MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis) {
    auto ctx = graph.getContext();
    DenseMap<Operation *, MO::Static::ReshapeOp> reshapeKernels;
    mlir::OpBuilder builder{ctx};
    graph.walk([&](MO::Static::ReshapeOp reshapeOp) {
      builder.setInsertionPoint(reshapeOp);
      if (MOGG::useExperimentalMOGGPath()) {
        // TODO: we should probably lower static.reshape during mo-to-mogg.
        MOGG::MOGGExpKernelBuilder lowerer(reshapeOp, builder, false,
                                           mojoLibraryAnalysis);

        auto expKernelOp = lowerer.lowerToKernel();
        ASSERT_STREAM(!expKernelOp.isError(), << "Could not lower reshape op");

        reshapeKernels[*expKernelOp] = reshapeOp;
        reshapeOp->replaceAllUsesWith(*expKernelOp);
      } else {
        MO::MOGGKernelBuilder lowerer(reshapeOp, builder, false,
                                      mojoLibraryAnalysis);

        auto kernelOp = lowerer.lowerToKernel();
        ASSERT_STREAM(!kernelOp.isError(), << "Could not lower reshape op");

        reshapeKernels[*kernelOp] = reshapeOp;
        reshapeOp->replaceAllUsesWith(*kernelOp);
      }
    });

    return reshapeKernels;
  }

  static void raiseBackReshapeKernels(
      DenseMap<Operation *, MO::Static::ReshapeOp> &reshapeOps) {
    for (auto &[kernelOp, reshapeOp] : reshapeOps) {
      kernelOp->replaceAllUsesWith(reshapeOp);
      kernelOp->erase();
    }
  }

  void runOnOperation() override {
    auto graph = getOperation();
    auto ctx = graph.getContext();

    // Get the LibraryAnalysis
    auto &analysis = getAnalysis<MOGG::LoadMojoAnalysisPass>();
    ErrorOr<MOGG::MojoLibraryAnalysis *> errOr =
        analysis.initIfNeeded(ctx, kernelLibraryPaths);
    if (errOr.isError()) {
      signalPassFailure();
      return;
    }
    MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis = *errOr.takeValue();

    // Get the type manager
    auto typeManager = MOGG::TypeManager::get(ctx, mojoLibraryAnalysis);
    ASSERT_STREAM(!typeManager.isError(), << typeManager.takeError());

    // First fuse all kernels that are view ops.
    DenseMap<Operation *, MO::Static::ReshapeOp> reshapeKernels;
    runOnOperationImpl(graph, ctx, reshapeKernels, mojoLibraryAnalysis,
                       *typeManager);

    reshapeKernels = lowerReshapeOps(graph, mojoLibraryAnalysis);

    // Do not attempt the next step if there are no reshape kernels (probably
    // because they would not be lowered to the extensibility/kernel API).
    if (!reshapeKernels.empty()) {
      // Then, lower mo.static.reshape to mogg.kernel and fuse things into them.
      runOnOperationImpl(graph, ctx, reshapeKernels, mojoLibraryAnalysis,
                         *typeManager);
      // Then, bring back the mo.static.reshape.
      raiseBackReshapeKernels(reshapeKernels);
    }
  }
};

} // namespace

} // namespace M::MOGG
