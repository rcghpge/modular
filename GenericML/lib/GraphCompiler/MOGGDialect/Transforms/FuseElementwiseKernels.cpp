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
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoTargetFunctions.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/TypeManager.h"
#include "KGEN/KGENDialect/KGENOps.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/IRMapping.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEELEMENTWISE
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

static std::optional<SmallVector<int>> getFusionIndices(KernelOp src,
                                                        KernelOp dst) {
  if (src.getOutputFusionInterfaceAt(0).empty())
    return std::nullopt;

  SmallVector<int> indices;
  for (auto [i, operand] : llvm::enumerate(dst.getOperands())) {
    if (operand.getDefiningOp() != src)
      continue;

    if (dst.getInputFusionInterfaceAt(i).empty())
      return std::nullopt;

    indices.push_back(i);
  }

  return indices;
}

static inline bool isAnElementwiseKernel(Operation *op) {
  auto kernelOp = dyn_cast<KernelOp>(op);
  if (!kernelOp)
    return false;

  if (kernelOp.getIsViewOp().value_or(false))
    return false;

  if (!kernelOp.getFusionBodyLambda())
    return false;

  if (!kernelOp.getOutputParamDecls().empty())
    return false;

  // We disable fusion for in-place allocated kernel.
  if (kernelOp.isInPlaceAlloc())
    return false;

  return true;
}

static inline bool kernelsAreCompatible(Operation *src, Operation *dst) {
  auto srcKernel = cast<KernelOp>(src);
  auto dstKernel = cast<KernelOp>(dst);

  if (!getFusionIndices(srcKernel, dstKernel).has_value())
    return false;

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

    /// TODO: MOCO-1469 Remove this limit.
    static constexpr uint64_t kMaximumFusionSetSize = 64;
    auto subgraphs = sourceDestinationSequenceFusionAnalysis(
        graphOp.getBody(), topologyAnalysis, isAnElementwiseKernel,
        kernelsAreCompatible, false, kMaximumFusionSetSize);

    for (auto &subgraph : subgraphs)
      topologyAnalysis.computeTopologicalSorting(subgraph);

    for (auto &subgraph : subgraphs) {
      numberOfFusedKernels += subgraph.size();
      auto dstKernelOp = cast<KernelOp>(subgraph.back());
      for (auto op : llvm::reverse(subgraph)) {
        if (op == dstKernelOp)
          continue;

        auto srcKernelOp = cast<KernelOp>(op);
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

    IRMapping mapper;
    OpBuilder builder{dstKernel.getContext()};

    MOGG::OutputPlaceholdersOp dstOutput = dstKernel.getOutputPlaceholder();
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
      Value dstArgument = dstKernel.addOperand(srcOperand);
      mapper.map(srcArgument, dstArgument);
      srcInputLambdaPositions[srcKernel.getInputFusionInterfaceAt(
          srcOperandIndex++)] = dstKernel.getNumOperands() - 1;
    }

    llvm::SmallVector<KGEN::ParamDeclareOp> newParamDeclOps;

    StringAttr renamedSrcElementwiseLambda;
    mlir::SymbolTable dstSymTable{dstKernel};

    DenseSet<Operation *> skipUsing;
    for (Operation &srcOp : srcKernel.getBody().getOps()) {
      // Ignore the one foreach call and any users of it.
      if (isCallToForeach(&srcOp) || isa<MOGG::OutputOp>(srcOp)) {
        // For deleting machinery to handle foreach `raising`
        SmallVector<Operation *> opAndSubgraphs =
            MO::getRootedComponents(&srcOp);
        skipUsing.insert(opAndSubgraphs.begin(), opAndSubgraphs.end());
      }

      if (skipUsing.contains(&srcOp))
        continue;

      if (auto srcOutput = dyn_cast<MOGG::OutputPlaceholdersOp>(srcOp)) {
        // The duplicate ToKGEN will be CSE-ed.
        mapper.map(srcOutput.getResult(0), dstOutput.getResult(0));
        continue;
      }

      auto lambda = dyn_cast<KGEN::ParamDeclareRegionOp>(srcOp);
      if (lambda == srcOutputLambda)
        continue;

      auto newOp = builder.clone(srcOp, mapper);

      // Symbols have to be added through the table to avoid duplication.
      if (lambda) {
        auto newName = dstSymTable.insert(newOp, builder.getInsertionPoint());
        auto oldName = lambda.getParamDecl().getName();
        if (srcInputLambdaPositions.contains(oldName)) {
          dstKernel.setInputFusionInterface(srcInputLambdaPositions[oldName],
                                            newName);
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
    builder.setInsertionPointAfter(dstOutput);

    // Creating at most one empty tensors op. This will be used as a replacement
    // for the old argument from the src op.
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
  }
};

} // namespace

} // namespace M::MOGG
