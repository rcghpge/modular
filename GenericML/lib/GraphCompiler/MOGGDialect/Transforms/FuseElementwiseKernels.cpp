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
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/SequenceFusionAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoTargetFunctions.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/TypeManager.h"
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

static std::optional<SmallVector<int>> getFusionIndices(KernelOp src,
                                                        KernelOp dst) {
  if (src.getOutputFusionInterfaceAt(0).empty()) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "  FAILED: Source kernel has empty output fusion interface\n");
    return std::nullopt;
  }

  SmallVector<int> indices;
  for (auto [i, operand] : llvm::enumerate(dst.getOperands())) {
    if (operand.getDefiningOp() != src)
      continue;

    if (isa<MO::ChainType>(operand.getType())) {
      LLVM_DEBUG(llvm::dbgs() << "  Skipping operand " << i
                              << " because it is a chain type\n");
      continue;
    }

    if (dst.getInputFusionInterfaceAt(i).empty()) {
      LLVM_DEBUG(llvm::dbgs() << "  FAILED: Destination kernel has empty input "
                                 "fusion interface at index "
                              << i << "\n");
      return std::nullopt;
    }

    indices.push_back(i);
  }

  return indices;
}

static inline bool isAnElementwiseKernel(Operation *op) {
  if (auto kernelOp = dyn_cast<ExpKernelOp>(op)) {
    return kernelOp.getElemwiseBodyLambda() != nullptr;
  }

  auto kernelOp = dyn_cast<KernelOp>(op);
  if (!kernelOp)
    return false;

  LLVM_DEBUG({
    llvm::dbgs() << "\n==================================================\n";
    llvm::dbgs() << "Checking if kernel is elementwise: "
                 << kernelOp.getSummaryAttr().getValue() << "\n";
  });

  if (kernelOp.getIsViewOp().value_or(false)) {
    LLVM_DEBUG(llvm::dbgs() << "  FAILED: Kernel is a view op\n");
    return false;
  }

  if (!kernelOp.getFusionBodyLambda()) {
    LLVM_DEBUG(llvm::dbgs() << "  FAILED: Kernel has no fusion body lambda\n");
    return false;
  }

  if (!kernelOp.getOutputParamDecls().empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "  FAILED: Kernel has output parameter declarations\n");
    return false;
  }

  // We disable fusion for in-place allocated kernel.
  if (kernelOp.isInPlaceAlloc()) {
    LLVM_DEBUG(llvm::dbgs() << "  FAILED: Kernel has in-place allocation\n");
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << "  SUCCESS: Kernel is elementwise\n");
  return true;
}

static inline bool kernelsAreCompatible(Operation *src, Operation *dst) {
  auto srcKernel = cast<KernelOp>(src);
  auto dstKernel = cast<KernelOp>(dst);

  LLVM_DEBUG({
    llvm::dbgs() << "\n==================================================\n";
    llvm::dbgs() << "Checking compatibility for kernels:\n";
    llvm::dbgs() << "  Source: " << srcKernel.getSummaryAttr().getValue();
    llvm::dbgs() << "\n  Destination: "
                 << dstKernel.getSummaryAttr().getValue();
    llvm::dbgs() << "\n";
  });

  auto fusionIndices = getFusionIndices(srcKernel, dstKernel);
  if (!fusionIndices.has_value()) {
    LLVM_DEBUG(llvm::dbgs()
               << "  FAILED: No valid fusion indices found. Source kernel may "
                  "have empty output fusion interface or destination kernel "
                  "may have empty input fusion interface.\n");
    return false;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "  Found fusion indices: [";
    llvm::interleaveComma(fusionIndices.value(), llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  // Check chain compatibility if both kernels have input and output chains
  auto srcInChain = srcKernel.maybeGetInChain();
  auto srcOutChain = srcKernel.maybeGetOutChain();
  auto dstInChain = dstKernel.maybeGetInChain();
  auto dstOutChain = dstKernel.maybeGetOutChain();

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

    // Get actual chain values
    Value srcOutChain = srcKernel.getOutChain();
    Value dstInChain = dstKernel.getInChain();

    // Ensure the input chain to dstKernel is output chain from srcKernel
    if (dstInChain != srcOutChain) {
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
    if (!srcOutChain.hasOneUse()) {
      LLVM_DEBUG({
        llvm::dbgs() << "  FAILED: Source output chain has multiple uses: "
                     << std::distance(srcOutChain.getUses().begin(),
                                      srcOutChain.getUses().end())
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
    Value chainVal = dstKernel.getInChain();
    if (!chainVal.hasOneUse()) {
      LLVM_DEBUG({
        llvm::dbgs() << "  FAILED: Destination input chain has multiple uses: "
                     << std::distance(chainVal.getUses().begin(),
                                      chainVal.getUses().end())
                     << "\n";
      });
      return false;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "  Kernels are compatible for fusion\n");
  return true;
}

// Check if applyExpr is of the form `apply(get_kernel_simd_width, ...)`.
static bool isCallToGetKernelSimdWidth(KGEN::ParamOperatorAttr applyExpr) {
  if (!applyExpr || applyExpr.getOpcode() != KGEN::POC::Apply)
    return false;

  auto applyCallee =
      dyn_cast<KGEN::SymbolConstantAttr>(applyExpr.getOperand(0));
  if (!applyCallee)
    return false;
  StringRef applySymName =
      cast<mlir::FlatSymbolRefAttr>(applyCallee.getSymbol()).getValue();
  return applySymName.starts_with(Utils::DPSTensorGetKernelSimdWidth);
}

static bool isCallToForeach(Operation *op) {
  auto callOp = dyn_cast<KGEN::CallOp>(op);
  if (!callOp)
    return false;

  StringRef funcName =
      cast<mlir::FlatSymbolRefAttr>(callOp.getCalleeSymbol()).getValue();
  return funcName.starts_with(Utils::DPSTensorForEach);
}

static LogicalResult
fixElemwiseKernelSimdWidth(TypeManager &typeManager, MOGG::KernelOp kernelOp,
                           const llvm::SmallSetVector<DType, 4> &allDtypes) {

  KGEN::CallOp forEachCall = nullptr;

  kernelOp.walk([&](KGEN::CallOp callOp) {
    // Find the foreach call.
    if (isCallToForeach(callOp)) {
      forEachCall = callOp;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  // If there is no forEachCall then this isn't an elementwise kernel.
  // Just skip it.
  if (!forEachCall)
    return success();

  // It has a parameter which is the simd_width of the output type.
  // Replace it with the minimum simd_width of all kernel dtypes.
  constexpr unsigned simdWidthParamIndex = 4;

  TypedAttr oldSimdWithAttr = forEachCall.getParamValues()[simdWidthParamIndex];
  TypedAttr newSimdWithAttr = nullptr;

  // By default the simdwidth is of the form apply(get_kernel_simd_width[dtype,
  // ...]). Just replace the old dtype with the dtypes in allDtypes. And
  // aggregate all with a min operation.
  auto applyExpr = dyn_cast<KGEN::ParamOperatorAttr>(oldSimdWithAttr);
  // If we can't find the call, just emit a warning and move on without any
  // error. The user might have called foreach with its own special simd_width
  // for some reason.
  if (!isCallToGetKernelSimdWidth(applyExpr)) {
    kernelOp.emitWarning(
        "Failed to find the call to get_kernel_simd_width for the simd_width "
        "parameter");
    return success();
  }

  auto applyCallee = cast<KGEN::SymbolConstantAttr>(applyExpr.getOperand(0));
  SmallVector<TypedAttr> newCalleeParams(applyCallee.getParamValues());

  for (DType dtype : allDtypes) {
    // The first parameter is the dtype.
    newCalleeParams[0] = typeManager.lookupTypeBinding(dtype);
    auto newCallee = KGEN::SymbolConstantAttr::get(
        applyCallee.getSymbol(), applyCallee.getType(), newCalleeParams);

    TypedAttr simdWidthOfDtype = KGEN::ParamOperatorAttr::get(
        KGEN::POC::Apply, {newCallee}, applyExpr.getType());

    if (newSimdWithAttr == nullptr)
      newSimdWithAttr = simdWidthOfDtype;
    else
      newSimdWithAttr = KGEN::ParamOperatorAttr::get(
          KGEN::POC::Max, {newSimdWithAttr, simdWidthOfDtype});
  }

  // Now change the parameter of the foreach call.
  SmallVector<TypedAttr> newForEachParams(forEachCall.getParamValues());
  newForEachParams[simdWidthParamIndex] = newSimdWithAttr;
  auto newForEachCallee = KGEN::SymbolConstantAttr::get(
      forEachCall.getCalleeSymbol(), forEachCall.getCalleeSignature(),
      newForEachParams);
  forEachCall.setCalleeAttr(newForEachCallee);

  return success();
}

//-------------------------------------------------
// mogg.experimental.kernel fusion utils.
//-------------------------------------------------

static std::optional<SmallVector<int>> getFusionIndices(ExpKernelOp src,
                                                        ExpKernelOp dst) {
  LLVM_DEBUG({
    llvm::dbgs() << "Examining fusion indices between experimental kernels:\n";
    llvm::dbgs() << "  Source: " << src.getSummaryAttr().getValue() << " at ";
    llvm::dbgs() << "\n  Destination: " << dst.getSummaryAttr().getValue()
                 << " at ";
    llvm::dbgs() << "\n";
  });

  SmallVector<int> indices;
  for (auto [i, operand] : llvm::enumerate(dst.getOperands())) {
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
  auto srcKernel = cast<ExpKernelOp>(src);
  auto dstKernel = cast<ExpKernelOp>(dst);

  return getFusionIndices(srcKernel, dstKernel).has_value();
}

static void fuseAndReplaceOperandWith(MOGG::ExpKernelOp dst, unsigned opIdx,
                                      MOGG::ExpKernelOp src) {
  // Top-level operands.
  dst->insertOperands(opIdx, src->getOperands());

  // Block arguments.
  Value toReplace = dst.getExecBody().getArgument(opIdx);
  for (auto v : llvm::reverse(src.getExecBody().getArguments()))
    dst.getExecBody().insertArgument(opIdx, v.getType(), v.getLoc());

  // All uses.
  unsigned numSrcVal = src->getNumOperands();
  ValueRange fusedArgs =
      dst.getExecBody().getArguments().slice(opIdx, numSrcVal);
  for (auto &use : toReplace.getUses()) {
    unsigned insertIdx = use.getOperandNumber();
    use.getOwner()->insertOperands(insertIdx, fusedArgs);
    use.getOwner()->eraseOperand(insertIdx + numSrcVal);
  }

  // cleanup
  dst.getExecBody().eraseArgument(opIdx + numSrcVal);
  dst->eraseOperands(opIdx + numSrcVal);
};

static void fuseExpElementwiseKernel(ArrayRef<Operation *> subgraph) {
  auto dst = cast<ExpKernelOp>(subgraph.back());
  PatternRewriter rewriter(dst.getContext());
  MOGG::BindLambdaOp dstBindOp = dst.getElemwiseBodyLambda();
  Block *dstFuseBody = &dstBindOp.getLambda().front();
  for (auto op : llvm::reverse(subgraph.drop_back())) {
    auto src = cast<ExpKernelOp>(op);

    SmallVector<int> indices = *getFusionIndices(src, dst);
    SmallVector<Value> fusedArgs(src->getNumOperands() + 1);

    for (int index : llvm::reverse(indices)) {
      // Simply fuses the elementwise lambda.
      Block *srcFuseBody = src.getElemwiseBodyLambda().getBody();

      auto yieldOp = cast<MOGG::OutputOp>(srcFuseBody->getTerminator());
      Value fusedVal = yieldOp->getOperand(0);
      yieldOp->erase();

      // Replace to the block argument to fused result.
      BlockArgument dstArg = dstFuseBody->getArgument(index);
      dstArg.replaceAllUsesWith(fusedVal);
      dstFuseBody->eraseArgument(index);
      for (auto ba : llvm::reverse(srcFuseBody->getArguments().drop_back())) {
        fusedArgs[ba.getArgNumber()] =
            dstFuseBody->insertArgument(index, ba.getType(), ba.getLoc());
      }
      // Indices are reused.
      fusedArgs.back() = dstFuseBody->getArguments().back();
      rewriter.inlineBlockBefore(srcFuseBody, dstFuseBody, dstFuseBody->begin(),
                                 fusedArgs);

      fuseAndReplaceOperandWith(dst, index, src);
    }

    dst.setSummaryAttr(getSummaryString({src, dst}, FusionType::HARD));
    src.erase();
  }
}

struct FuseElementwisePass
    : public impl::FuseElementwiseBase<FuseElementwisePass> {
  using FuseElementwiseBase::FuseElementwiseBase;

  FuseElementwisePass() = default;
  FuseElementwisePass(const FuseElementwisePass &pass) {}

  Statistic numberOfFusedKernels{this, "numberOfFusedKernels",
                                 "Number of fused kernels"};

  LogicalResult fixupKGENKernelsSimdWidth(MO::GraphOp graphOp) {
    // Fixup simd_width for fused kernels when needed.
    auto &analysis = getAnalysis<MOGG::LoadMojoAnalysisPass>();
    ErrorOr<MOGG::MojoLibraryAnalysis *> errOr =
        analysis.initIfNeeded(graphOp->getContext(), kernelLibraryPaths);
    if (errOr.isError())
      return failure();
    MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis = *errOr.takeValue();
    auto typeManager =
        TypeManager::get(graphOp->getContext(), mojoLibraryAnalysis);
    if (typeManager.isError())
      return failure();

    auto walkResult = graphOp.walk([&](MOGG::KernelOp kernelOp) -> WalkResult {
      llvm::SmallSetVector<DType, 4> allDtypes;
      auto findDType = [&allDtypes](Type ty) {
        if (auto tensorTy = dyn_cast<MO::TensorType>(ty))
          allDtypes.insert(tensorTy.getDtype());
        else if (auto bufferTy = dyn_cast<MO::BufferType>(ty))
          allDtypes.insert(bufferTy.getDtype());
      };
      for (Type ty : kernelOp->getOperandTypes())
        findDType(ty);
      for (Type ty : kernelOp.getResultTypes())
        findDType(ty);

      // Only fix kernels with multiple dtypes
      if (allDtypes.size() <= 1)
        return WalkResult::advance();

      if (failed(fixElemwiseKernelSimdWidth(*typeManager, kernelOp, allDtypes)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    return success(!walkResult.wasInterrupted());
  }

  void runOnOperation() override {
    MO::GraphOp graphOp = getOperation();
    MO::ParametricTopologyAnalysis topologyAnalysis(graphOp);

    MOGG::CompatibilityFunction compatibilityCheck =
        MOGG::useExperimentalMOGGPath() ? expKernelsAreCompatible
                                        : kernelsAreCompatible;

    /// TODO: MOCO-1469 Remove this limit.
    static constexpr uint64_t kMaximumFusionSetSize = 64;
    auto subgraphs = sourceDestinationSequenceFusionAnalysis(
        graphOp.getBody(), topologyAnalysis, isAnElementwiseKernel,
        compatibilityCheck, false, kMaximumFusionSetSize);

    for (auto &subgraph : subgraphs)
      topologyAnalysis.computeTopologicalSorting(subgraph);

    for (ArrayRef subgraph : subgraphs) {
      numberOfFusedKernels += subgraph.size();
      if (isa<ExpKernelOp>(subgraph.back())) {
        fuseExpElementwiseKernel(subgraph);
        continue;
      }

      // KGEN fusion path.
      auto dstKernelOp = cast<KernelOp>(subgraph.back());
      for (auto op : llvm::reverse(subgraph)) {
        if (op == dstKernelOp)
          continue;

        auto srcKernelOp = cast<KernelOp>(op);

        LLVM_DEBUG({
          llvm::dbgs()
              << "\n==================================================\n";
          llvm::dbgs() << "Performing KGEN fusion between kernels:\n";
          llvm::dbgs() << "  Source: "
                       << srcKernelOp.getSummaryAttr().getValue() << "\n";
          llvm::dbgs() << "  Destination: "
                       << dstKernelOp.getSummaryAttr().getValue() << "\n";
        });

        fuseSrcKernelIntoDstKernelKGEN(srcKernelOp, dstKernelOp);

        dstKernelOp.setSummaryAttr(
            getSummaryString({srcKernelOp, dstKernelOp}, FusionType::HARD));
        srcKernelOp.erase();
      }

      dstKernelOp.walk([](Block *block) { mlir::sortTopologically(block); });
    }

    if (failed(fixupKGENKernelsSimdWidth(graphOp)))
      signalPassFailure();
  }

  // The KGEN equivalent of fusion consists of replacing calls of the old input
  // lambda with calls to the elementwise lambda in the source kernel. This is
  // possible because they have the same signature.
  void fuseSrcKernelIntoDstKernelKGEN(KernelOp srcKernel, KernelOp dstKernel) {
    stageKernelsForFusion({srcKernel, dstKernel});

    auto fusionIndices = *getFusionIndices(srcKernel, dstKernel);

    LLVM_DEBUG({
      llvm::dbgs() << "  Fusion indices for KGEN fusion: [";
      llvm::interleaveComma(fusionIndices, llvm::dbgs());
      llvm::dbgs() << "]\n";
    });

    IRMapping mapper;
    OpBuilder builder{dstKernel.getContext()};

    auto dstConverterOps = dstKernel.getOps<ToKGENOp>();
    if (dstConverterOps.empty()) {
      dstKernel.setInsertionPointToTop(builder);
    } else {
      SmallVector<ToKGENOp> dstConverterOpsArray(dstConverterOps.begin(),
                                                 dstConverterOps.end());
      builder.setInsertionPointAfter(dstConverterOpsArray.back());
    }

    KGEN::ParamDeclareRegionOp srcOutputLambda = srcKernel.findLambda(
        cast<StringAttr>(srcKernel.getOutputFusionInterfaceAttr()[0]));
    KGEN::ParamDeclareRegionOp srcElementwiseLambda =
        srcKernel.findLambda(srcKernel.getFusionBodyLambdaAttr());
    assert(srcElementwiseLambda);

    // Add each of the new operands to the kernel body we are fusing into and
    // the corresponding input lambdas.
    llvm::DenseMap<StringAttr, int> srcInputLambdaPositions;
    uint64_t srcOperandIndex = 0;
    for (auto [srcOperand, srcArgument] : llvm::zip(
             srcKernel.getOperands(), srcKernel.getBody().getArguments())) {
      if (isa<MO::ChainType>(srcOperand.getType()))
        continue;
      Value dstArgument = dstKernel.addOperand(srcOperand);
      mapper.map(srcArgument, dstArgument);
      srcInputLambdaPositions[srcKernel.getInputFusionInterfaceAt(
          srcOperandIndex++)] = dstKernel.getNumOperands() - 1 -
                                dstKernel.maybeGetInChain().has_value();
    }

    llvm::SmallVector<KGEN::ParamDeclareOp> newParamDeclOps;

    StringAttr renamedSrcElementwiseLambda;
    mlir::SymbolTable dstSymTable{dstKernel};

    DenseSet<Operation *> skipUsing;
    for (Operation &srcOp : srcKernel.getBody().getOps()) {
      // Ignore the one foreach call and any users of it.
      if (isCallToForeach(&srcOp) || isa<MOGG::OutputOp>(srcOp) ||
          // Since this pass currently only expects src kernels w/ one result we
          // can get away with just pruning everything away connected its
          // output placeholder.
          isa<MOGG::OutputPlaceholdersOp>(srcOp)) {
        // For deleting machinery to handle foreach `raising`
        SmallVector<Operation *> opAndSubgraphs =
            MO::getRootedComponents(&srcOp);
        skipUsing.insert(opAndSubgraphs.begin(), opAndSubgraphs.end());
      }

      if (skipUsing.contains(&srcOp))
        continue;

      auto lambda = dyn_cast<KGEN::ParamDeclareRegionOp>(srcOp);
      if (lambda == srcOutputLambda)
        continue;

      auto newOp = builder.clone(srcOp, mapper);

      // Symbols have to be added through the table to avoid duplication.
      if (lambda) {
        auto newName = dstSymTable.insert(newOp, builder.getInsertionPoint());
        auto oldName = lambda.getParamDecl().getName();
        if (auto it = srcInputLambdaPositions.find(oldName);
            it != srcInputLambdaPositions.end()) {
          dstKernel.setInputFusionInterface(it->second, newName);
        }

        if (lambda == srcElementwiseLambda)
          renamedSrcElementwiseLambda = newName;
      }

      if (auto newDeclOp = dyn_cast<KGEN::ParamDeclareOp>(newOp)) {
        newParamDeclOps.push_back(newDeclOp);
      }
    }

    assert(renamedSrcElementwiseLambda);

    auto &dstKernelBody = dstKernel.getBody().front();
    dstKernel.setInsertionPointToTop(builder);

    // Creating at most one empty tensors op. This will be used as a
    // replacement for the old argument from the src op.
    auto fakeTensors = dstKernel.getCreateEmptyTensor();
    if (!fakeTensors) {
      fakeTensors = builder.create<MOGG::CreateEmptyTensorsOp>(
          dstKernel.getLoc(),
          dstKernelBody.getArgument(fusionIndices.front()).getType());
    } else {
      auto fakeTensorTypes = SmallVector<Type>(fakeTensors->getResultTypes());
      fakeTensorTypes.push_back(
          dstKernelBody.getArgument(fusionIndices.front()).getType());
      auto newFakeTensors = builder.create<MOGG::CreateEmptyTensorsOp>(
          dstKernel.getLoc(), fakeTensorTypes);
      fakeTensors->replaceAllUsesWith(newFakeTensors.getResults().drop_back());
      fakeTensors.erase();
      fakeTensors = newFakeTensors;
    }

    for (int fusionIndex : llvm::reverse(fusionIndices)) {
      StringAttr dstKernelInputLambdaName =
          dstKernel.getInputFusionInterfaceAt(fusionIndex);
      KGEN::ParamDeclareRegionOp dstInputLambda =
          dstKernel.findLambda(dstKernelInputLambdaName);

      // Now replace all references to the input lambda with the new lambda.
      mlir::AttrTypeReplacer walker;
      walker.addReplacement([&](KGEN::ParamDeclRefAttr attr)
                                -> std::optional<KGEN::ParamDeclRefAttr> {
        if (attr.getName() == dstKernelInputLambdaName)
          return KGEN::ParamDeclRefAttr::get(
              renamedSrcElementwiseLambda,
              srcElementwiseLambda.getParamDecl().getType());
        return std::nullopt;
      });
      walker.recursivelyReplaceElementsIn(dstKernel, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);

      dstInputLambda.erase();

      // By composition we've already guaranteed the pointer is not going to be
      // accessed. What this means is that although this tensor is used by some
      // function calls, it's never actually read/written from/to. This is
      // because functions (including lambdas) have access to more than their
      // operands. They can capture values outside of their scope and that's how
      // lambdas actually access the right tensor. This one will never be
      // accessed but it is necessary to not have to temper with the KGEN code
      // (changing the signature of whatever was taking in that argument). In
      // other words, we create an empty tensors op and feed it to the code that
      // still requires a tensor although it does not write to the underlying
      // data.
      dstKernelBody.getArgument(fusionIndex)
          .replaceAllUsesWith(
              fakeTensors.getResult(fakeTensors->getNumResults() - 1));
      dstKernel.removeOperandAtIndex(fusionIndex);
    }

    // Get rid of the unused declarations for some of the removed lambdas.
    clearUnusedParamDecls(dstKernel, newParamDeclOps);

    // Handle re-wiring chains when both src/dest use them.
    if (srcKernel.maybeGetInChain().has_value() &&
        srcKernel.maybeGetOutChain().has_value() &&
        dstKernel.maybeGetInChain().has_value() &&
        dstKernel.maybeGetOutChain().has_value()) {
      Value srcInChain = srcKernel.getInChain();
      Value dstInChain = dstKernel.getInChain();
      // TODO(GEX-2060): Replace with call to helper to get an OpOperand
      for (unsigned i = 0; i < dstKernel->getNumOperands(); ++i) {
        if (dstKernel->getOperand(i) == dstInChain) {
          dstKernel->setOperand(i, srcInChain);
          break;
        }
      }

      srcKernel.getOutChain().replaceAllUsesWith(dstKernel.getOutChain());
    }

    // Set read and write effect attributes on the destination kernel directly
    dstKernel.setHasReadEffect(srcKernel.getHasReadEffect() ||
                               dstKernel.getHasReadEffect());
    dstKernel.setHasWriteEffect(srcKernel.getHasWriteEffect() ||
                                dstKernel.getHasWriteEffect());
    LLVM_DEBUG(llvm::dbgs()
               << "SUCCESS: Fused " << srcKernel.getSummaryAttr().getValue()
               << " into " << dstKernel.getSummaryAttr().getValue() << "\n");
  }
};

} // namespace

} // namespace M::MOGG
