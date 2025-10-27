//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Converts MO ops to MGP and MEF ops.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOToMOGG.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MOGGCompiler.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MOGGTypeTraits.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MOToKGENLowering.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MOToMOGGLowering.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"
#include "Support/AssertStream.h"
#include "Support/Context.h"
#include "Support/ML/TensorShape.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace M;
using namespace mlir;

namespace M::MO {

#define GEN_PASS_DEF_MOTOMOGG
#include "GenericML/GraphCompiler/Conversion/MOToMOGG.h.inc"

} // namespace M::MO

namespace {

/// Return whether we support the types for lowering the given op.
bool opHasCompatibleTypes(Operation *op) {
  auto isDTypeAllowed = [&](MO::TensorType tensorTy) {
    Type dtype = tensorTy.getElementType();
    return MOGG::MOGGTypeTraits::isSupportedMOGGType(dtype);
  };

  // Returns true if a type is compatible with jit.
  auto isShapeAllowed = [&](MO::TensorType tensorTy) {
    if (!tensorTy || !tensorTy.hasRank() || tensorTy.getRank() > kMaxRank ||
        tensorTy.hasUnknownShape())
      return false;

    // The symbolic shape must be trivial and not a compound expression. I.E
    // N*M.
    return llvm::all_of(tensorTy.getDims(), MOGG::isDimAttrSupported);
  };

  bool hasCompatibleTypes = true;
  for (Type t : op->getOperandTypes()) {
    if (auto tensorTy = MO::dynCastFromTensorLike(t)) {
      hasCompatibleTypes &= isDTypeAllowed(tensorTy);
      hasCompatibleTypes &= isShapeAllowed(tensorTy);
    } else if (auto listTy = dyn_cast<MO::ListType>(t)) {
      hasCompatibleTypes &= isa<MO::TensorType>(listTy.getElementType());
    } else {
      hasCompatibleTypes &= isa<MO::OpaqueType>(t) || isa<MO::ChainType>(t) ||
                            isa<mlir::IndexType>(t);
    }
  }
  for (Type t : op->getResultTypes()) {
    if (auto tensorTy = dyn_cast<MO::TensorType>(t)) {
      hasCompatibleTypes &= isDTypeAllowed(tensorTy);
      hasCompatibleTypes &= isShapeAllowed(tensorTy);
    } else {
      hasCompatibleTypes &= isa<MO::OpaqueType>(t) || isa<MO::ChainType>(t);
    }
  }

  return hasCompatibleTypes;
}

bool shouldLower(Operation *op) {
  if (isa<MO::PreservedDuringKernelLowering>(op))
    return false;

  auto dialect = op->getDialect();
  if (!dialect)
    return false;

  if (!isa<MO::MODialect>(dialect))
    return false;

  return true;
}

// nullopt --> expected no lowering. Error --> unexpected error. Else lowering
std::optional<ErrorOr<Operation *>>
lowerOpToMOGG(Operation *op, OpBuilder &insertPoint,
              MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis, bool forceTrivial,
              bool forceSync) {
  if (!shouldLower(op))
    return std::nullopt;

  if (!opHasCompatibleTypes(op)) {
    std::string errStr;
    llvm::raw_string_ostream os(errStr);
    os << "Found at least one incompatible input/output type for a JIT-able "
          "operation which is not supported"
       << ". Operand types: " << op->getOperandTypes()
       << ". Results Types: " << op->getResultTypes();
    return Error(errStr);
  }

  // Done to match expected behavior of non-extensibility api
  forceSync = forceSync || forceTrivial;

  bool isaShapeFunc = isa<MO::InvokeShapeFuncOp>(op);
  bool willEmitMojo = MOGG::useExperimentalMojoPath();

  // Shape functions are only lowered under experimental Mojo path.
  // LayoutTransforms are not yet lowered.
  if (isa<MO::LayoutTransformOp>(op) || (isaShapeFunc && !willEmitMojo))
    return std::nullopt;

  MOGG::MOGGExpKernelBuilder builder(op, insertPoint, forceSync,
                                     mojoLibraryAnalysis);

  auto kernelOrErr = builder.lowerToKernel();
  return kernelOrErr;
}

struct MOToMOGGPass : public MOGG::MojoLibraryAnalysisPass<
                          M::MO::impl::MOToMOGGBase<MOToMOGGPass>> {
  using Base::Base;

  // Replaces almost all rebinds which do not operate with unknown shapes.
  // Returns whether we were successful.
  bool replaceMORebinds() {
    // Load mojo library analysis
    auto analysisOr = getLibraryAnalysis();
    if (analysisOr.isError()) {
      emitError(getOperation().getLoc(),
                llvm::formatv("[{0}] Failed to load Mojo analysis: {1}",
                              getPassName(), analysisOr.getError()));
      return false;
    }

    auto res = getOperation().walk([&](MO::RebindOp op) {
      // Don't mess any rebind that deals with `?`. When this pass is run
      // this exists due to PT, we need the rebind to give dimensions a proper
      // name.
      MO::TensorType inTy = op.getInput().getType();
      MO::TensorType outTy = op.getResult().getType();
      if (inTy.hasUnknownShape() || outTy.hasUnknownShape())
        return WalkResult::advance();

      // Preserve layout attributes which are needed for best matmul/conv impl.
      if (outTy.getLayoutAttr() || inTy.getLayoutAttr())
        return WalkResult::advance();

      // Don't mess up the signature of ops which are not lowered to MOGG.
      // This is to ensure things still type-check since we keep the old op IR.
      // The exception are rebinds which we need to propagate information.
      llvm::SmallPtrSet<Operation *, 2> replacementExceptions;
      for (Operation *user : op->getUsers()) {
        if (isa<MO::RebindOp>(user))
          continue;

        if (!shouldLower(user))
          replacementExceptions.insert(user);
      }

      replacementExceptions.insert(op);
      op.getResult().replaceAllUsesExcept(op.getInput(), replacementExceptions);
      return WalkResult::advance();
    });

    return !res.wasInterrupted();
  }

  bool lowerMOOpsToMOGG() {
    // Load mojo library analysis
    auto analysisOr = getLibraryAnalysis();
    if (analysisOr.isError()) {
      emitError(getOperation().getLoc(),
                llvm::formatv("[{0}] Failed to load Mojo analysis: {1}",
                              getPassName(), analysisOr.getError()));
      return false;
    }
    MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis = *analysisOr.takeValue();

    // Run MOGG lowering on any valid operation
    auto result = getOperation().walk([&](Operation *op) {
      OpBuilder insertPoint{op};
      auto maybeOp = lowerOpToMOGG(op, insertPoint, mojoLibraryAnalysis,
                                   /*forceTrivial=*/false, forceSync);

      // Expected no lowering.
      if (!maybeOp.has_value())
        return WalkResult::advance();

      ErrorOr<Operation *> &results = maybeOp.value();
      if (results.isError()) {
        // Lowering error, abort pass!
        //
        // NOTE: getError() returns a `const char*`, which InFlightDiagnostics
        // lazily adds to report, storing only the pointer. As a result, if the
        // Error is deallocated and the memory goes along with it, we get
        // corrupted information.
        //
        // To get around this, force a copy by the InFlightDiagnostics by
        // wrapping things in Twine.
        op->emitOpError("[MO_TO_MOGG] ") << llvm::Twine(results.getError());
        return WalkResult::interrupt();
      } else {
        op->replaceAllUsesWith(results.takeValue()->getResults());
        op->erase();
        return WalkResult::advance();
      }
    });

    return !result.wasInterrupted();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns{ctx};

    // Replace almost all rebinds without unknown shapes with their original
    // inputs. This removes most rebinds in the graph and is done to prevent
    // rebinds from blocking fusion.
    bool replacementSuccess = replaceMORebinds();
    if (!replacementSuccess)
      return signalPassFailure();

    // Run MOGG lowering on any valid operation
    bool moggLoweringSuccess = lowerMOOpsToMOGG();
    if (!moggLoweringSuccess)
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> M::MO::createMOToMOGG(
    const MOToMOGGOptions &opts,
    std::shared_ptr<MOGG::MojoLibraryAnalysis> libraryAnalysis) {
  return std::make_unique<MOToMOGGPass>(opts, libraryAnalysis);
}
