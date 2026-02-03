//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "GenericML/GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/SequenceFusionAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoTargetFunctions.h"
#include "KGEN/KGENDialect/KGENOps.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"

using namespace M;

#define DEBUG_TYPE "fuse-elementwise"

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEELEMENTWISE
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

static bool checkMutableKernelFusible(std::optional<Value> srcInChain,
                                      std::optional<Value> srcOutChain,
                                      std::optional<Value> dstInChain,
                                      std::optional<Value> dstOutChain) {
  LLVM_DEBUG({
    if (dstInChain.has_value() || dstOutChain.has_value()) {
      llvm::dbgs() << "  Destination kernel has ";
      if (dstInChain.has_value() && dstOutChain.has_value())
        llvm::dbgs() << "input and output chains\n";
    }
  });

  if (srcInChain.has_value() && srcOutChain.has_value() &&
      dstInChain.has_value() && dstOutChain.has_value()) {
    LLVM_DEBUG(llvm::dbgs() << "  Both kernels have input and output chains\n");

    // Ensure the input chain to dstKernel is output chain from srcKernel
    if (*dstInChain != *srcOutChain) {
      LLVM_DEBUG({
        llvm::dbgs() << "  FAILED: Chain mismatch - source output chain and "
                        "destination input chain are different values\n";
      });
      return false;
    }

    // Ensure that dstKernel is the sole consumer of the output chain from
    // srcKernel.
    // Strictly speaking srcKernel having more than one consumer of
    // its output chain would be UB when those consumers perform writes to
    // mo.buffers -- skip fusion here to make the presence of that situation
    // clearer.
    if (!srcOutChain->hasOneUse()) {
      LLVM_DEBUG({
        llvm::dbgs() << "  FAILED: Source output chain has multiple uses: "
                     << std::distance(srcOutChain->getUses().begin(),
                                      srcOutChain->getUses().end())
                     << "\n";
      });
      return false;
    }
  }

  // Ensure if the kernel has an input chain, it has only one use
  // TODO(GEX-2105): This check is slightly stricter than it needs to be, as we
  // kind of assume that the kernel in question potentially mutates all of its
  // buffer operands.
  if (dstInChain.has_value()) {
    if (!dstInChain->hasOneUse()) {
      LLVM_DEBUG({
        llvm::dbgs() << "  FAILED: Destination input chain has multiple uses: "
                     << std::distance(dstInChain->getUses().begin(),
                                      dstInChain->getUses().end())
                     << "\n";
      });
      return false;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "  Kernels are compatible for fusion\n");
  return true;
}

static inline bool isAnElementwiseKernel(Operation *op) {
  if (auto kernelOp = dyn_cast<KernelOp>(op)) {
    return isa<MOGG::CallForeachOp>(kernelOp.getTerminator());
  }

  return false;
}

static inline bool canDuplicateIfOnNotOnHost(Operation *op) {
  return !MO::isOnHost(op);
}

//-------------------------------------------------
// mogg.kernel fusion utils.
//-------------------------------------------------

static std::optional<SmallVector<int>> getFusionIndices(KernelOp src,
                                                        KernelOp dst) {
  LLVM_DEBUG({
    llvm::dbgs() << "Examining fusion indices between experimental kernels:\n";
    llvm::dbgs() << "  Source: " << src.getSummaryAttr().getValue() << " at ";
    llvm::dbgs() << "\n  Destination: " << dst.getSummaryAttr().getValue()
                 << " at ";
    llvm::dbgs() << "\n";
  });

  SmallVector<int> indices;
  for (auto [i, operand] : llvm::enumerate(dst.getOperands())) {
    /// TODO: Support fusion in the presence of chains.
    if (isa<MO::ChainType>(operand.getType()))
      continue;

    if (operand.getDefiningOp() != src)
      continue;

    if (!dst.isOperandFusible(i)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  FAILED: Operand at index " << i << " is not fusible\n");
      return std::nullopt;
    }

    indices.push_back(i);
  }

  if (indices.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "  FAILED: No operands of destination kernel "
                               "come from source kernel\n");
    return std::nullopt;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "  SUCCESS: Found " << indices.size()
                 << " valid fusion indices: [";
    llvm::interleaveComma(indices, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });
  return indices;
}

static inline bool expKernelsAreCompatible(Operation *src, Operation *dst) {
  auto srcKernel = cast<KernelOp>(src);
  auto dstKernel = cast<KernelOp>(dst);

  if (!getFusionIndices(srcKernel, dstKernel).has_value())
    return false;

  auto srcInChain = srcKernel.maybeGetInChain();
  auto srcOutChain = srcKernel.maybeGetOutChain();
  auto dstInChain = dstKernel.maybeGetInChain();
  auto dstOutChain = dstKernel.maybeGetOutChain();

  return checkMutableKernelFusible(srcInChain, srcOutChain, dstInChain,
                                   dstOutChain);
}

static void fuseAndReplaceOperandWith(MOGG::KernelOp dst, unsigned opIdx,
                                      MOGG::KernelOp src) {
  Value toReplace = dst.getExecBody().getArgument(opIdx);

  dst.insertInputsAt(opIdx, src.getInputs());

  // All uses.
  unsigned numSrcVal = src->getNumOperands();
  ValueRange fusedArgs =
      dst.getExecBody().getArguments().slice(opIdx, numSrcVal);

  // erase below might corrupt use-def list, extract to a vector first.
  auto uses = llvm::map_to_vector(toReplace.getUses(),
                                  [](OpOperand &op) { return &op; });
  for (auto use : uses) {
    unsigned insertIdx = use->getOperandNumber();
    // It is actually necessary to have a a tmp *op here to avoid UAF (since
    // insert below might trigger a resize, making `use` a dangling pointer).
    Operation *op = use->getOwner();
    op->insertOperands(insertIdx, fusedArgs);
    op->eraseOperand(insertIdx + numSrcVal);
  }
  // cleanup
  dst.getExecBody().eraseArgument(opIdx + numSrcVal);
  dst.getInputsMutable().erase(opIdx + numSrcVal);
}

static void fuseKernelsIntoLastKernel(ArrayRef<Operation *> subgraph) {
  auto dst = cast<KernelOp>(subgraph.back());
  PatternRewriter rewriter(dst.getContext());
  MOGG::BindLambdaOp dstBindOp = dst.getElemwiseBodyLambda();
  Block *dstFuseBody = &dstBindOp.getLambda().front();
  for (auto op : llvm::reverse(subgraph.drop_back())) {
    auto src = cast<KernelOp>(op);

    SmallVector<int> indices = *getFusionIndices(src, dst);
    SmallVector<Value> fusedArgs(src->getNumOperands() + 1);
    Region &srcFuseRegion = src.getElemwiseBodyLambda().getBodyRegion();

    for (int kernelOperandIdx : llvm::reverse(indices)) {
      // Make a clone first, since one region might be used multiple time.
      // Otherwise, the `rewriter.inlineBlockBefore` below would delete the
      // block.
      rewriter.cloneRegionBefore(srcFuseRegion, *(dstFuseBody->getParent()),
                                 dstFuseBody->getParent()->begin());
      Block *srcFuseBody = &dstFuseBody->getParent()->front();

      auto yieldOp = cast<MOGG::OutputOp>(srcFuseBody->getTerminator());
      Value fusedVal = yieldOp->getOperand(0);
      yieldOp->erase();

      // Find the block argument in bindop corresponding to the value being
      // fused, replace the block argument to fused result.
      BlockArgument dstArg;
      for (auto ba : dstFuseBody->getArguments()) {
        if (dstBindOp->getOperand(ba.getArgNumber()) ==
            dst.getExecBody().getArgument(kernelOperandIdx)) {
          dstArg = ba;
          break;
        }
      }
      assert(dstArg);

      int bindOperandIdx = dstArg.getArgNumber();
      dstArg.replaceAllUsesWith(fusedVal);
      dstFuseBody->eraseArgument(bindOperandIdx);
      for (auto ba : llvm::reverse(srcFuseBody->getArguments().drop_back())) {
        fusedArgs[ba.getArgNumber()] = dstFuseBody->insertArgument(
            bindOperandIdx, ba.getType(), ba.getLoc());
      }
      // Indices are reused.
      fusedArgs.back() = dstFuseBody->getArguments().back();
      rewriter.inlineBlockBefore(srcFuseBody, dstFuseBody, dstFuseBody->begin(),
                                 fusedArgs);

      fuseAndReplaceOperandWith(dst, kernelOperandIdx, src);
    }

    dst.setSummaryAttr(getSummaryString({src, dst}, FusionType::HARD));
    dst.setHasReadEffect(src.getHasReadEffect() || dst.getHasReadEffect());
    dst.setHasWriteEffect(src.getHasWriteEffect() || dst.getHasWriteEffect());
  }
}

struct FuseElementwisePass
    : public impl::FuseElementwiseBase<FuseElementwisePass> {
  using FuseElementwiseBase::FuseElementwiseBase;

  FuseElementwisePass() = default;

  void runOnOperation() override {
    MO::GraphOp graphOp = getOperation();
    MO::ParametricTopologyAnalysis topologyAnalysis(graphOp);
    MOGG::CompatibilityFunction compatibilityCheck = expKernelsAreCompatible;

    /// TODO: MOCO-1469 Remove this limit.
    static constexpr uint64_t kMaximumFusionSetSize = 64;
    auto subgraphs = sourceDestinationSequenceFusionAnalysis(
        graphOp.getBody(), topologyAnalysis, isAnElementwiseKernel,
        compatibilityCheck, canDuplicateIfOnNotOnHost, kMaximumFusionSetSize);

    for (auto &subgraph : subgraphs)
      topologyAnalysis.computeTopologicalSorting(subgraph);

    for (ArrayRef subgraph : subgraphs) {
      numberOfFusedKernels += subgraph.size();
      fuseKernelsIntoLastKernel(subgraph);
    }

    // Erase operations that were fused. The DenseSet is necessary because when
    // duplication is allowed, an op will appear in more than one subgraph.
    DenseSet<Operation *> erased;
    for (ArrayRef subgraph : subgraphs) {
      for (auto op : llvm::reverse(subgraph.drop_back())) {
        if (!erased.contains(op) && op->getUsers().empty()) {
          erased.insert(op);
          op->erase();
        }
      }
    }
  }
};

} // namespace

} // namespace M::MOGG
