//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Converts MO ops to MGP and MEF ops.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/Conversion/MOToMOGG.h"
#include "GraphCompiler/GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/MOGGTypeTraits.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/MOToMOGGLowering.h"
#include "GraphCompiler/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/AssertStream.h"
#include "Support/Context.h"
#include "Support/ML/TensorShape.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace M;
using namespace mlir;

/// Check if a KGEN expression opcode is supported by the MOGG compiler.
static bool isKGENExprSupported(KGEN::POC opcode) {
  switch (opcode) {
  case KGEN::POC::Add:
  case KGEN::POC::MulNoWrap:
  case KGEN::POC::Cond:
  case KGEN::POC::EQ:
  case KGEN::POC::Div:
    return true;
  default:
    return false;
  }
}

/// Check whether all of the expression nested within `dimAttr` are supported by
/// the MOGG compiler.
static bool isDimAttrSupported(TypedAttr dimAttr) {
  if (auto paramOpAttr = dyn_cast<KGEN::ParamOperatorAttr>(dimAttr)) {
    // The opcode must be supported.
    if (!isKGENExprSupported(paramOpAttr.getOpcode()))
      return false;

    // Make sure operands are supported.
    return llvm::all_of(paramOpAttr.getOperands(), isDimAttrSupported);
  }

  // MOGG only supports compiling parameter of bool type.
  if (auto paramDeclAttr = dyn_cast<KGEN::ParamDeclRefAttr>(dimAttr))
    if (!MOSH::ShapeAttr::isDim(paramDeclAttr))
      return false;

  return true;
}

namespace M::MO {

#define GEN_PASS_DEF_MOTOMOGG
#include "GraphCompiler/GraphCompiler/Conversion/MOToMOGG.h.inc"

} // namespace M::MO

namespace {

/// Return whether we support the types for lowering the given op.
static bool opHasCompatibleTypes(Operation *op) {
  auto isDTypeAllowed = [&](MO::TensorType tensorTy) {
    Type dtype = tensorTy.getElementType();
    return MOGG::MOGGTypeTraits::isSupportedMOGGType(dtype);
  };

  // Returns true if a type is compatible with jit.
  auto isShapeAllowed = [&](MO::TensorType tensorTy) {
    if (!tensorTy || !tensorTy.hasRank() || tensorTy.getRank() > kMaxRank)
      return false;

    // The symbolic shape must be trivial and not a compound expression. I.E
    // N*M.
    return llvm::all_of(tensorTy.getDims(), isDimAttrSupported);
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
              MOGG::MojoRegistry &registry, bool forceTrivial, bool forceSync) {
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

  MOGG::MOGGKernelBuilder builder(op, insertPoint, forceSync, registry);

  auto kernelOrErr = builder.lowerToKernel();
  return kernelOrErr;
}

struct MOToMOGGPass : public M::MO::impl::MOToMOGGBase<MOToMOGGPass> {
  using Base::Base;

  // Replaces almost all rebinds which do not operate with unknown shapes.
  // Returns whether we were successful.
  bool replaceMORebinds() {
    auto res = getOperation().walk([&](MO::RebindOp op) {
      // Don't mess any rebind that deals with `?`. When this pass is run
      // this exists due to PT, we need the rebind to give dimensions a proper
      // name.
      MO::TensorType inTy = op.getInput().getType();
      MO::TensorType outTy = op.getResult().getType();

      // Preserve layout attributes which are needed for best matmul/conv impl.
      if (outTy.getLayoutAttr() || inTy.getLayoutAttr())
        return WalkResult::advance();

      // Preserve rebinds that refine types by replacing symbolic dimensions
      // with static. This happens when LegalizeRMOOps lowers operations like
      // rmo.mo.slice to mo.slice + mo.rebind
      if (inTy.hasRank() && outTy.hasRank() &&
          inTy.getRank() == outTy.getRank()) {
        auto inShape = inTy.getTypedShapeAttr();
        auto outShape = outTy.getTypedShapeAttr();

        for (auto [inDim, outDim] :
             llvm::zip_equal(inShape.getValues(), outShape.getValues())) {
          // Check if the input dimension is symbolic and the output dimension
          // is static (integer literal)
          if (isa<KGEN::ParamDeclRefAttr>(inDim) &&
              MO::TensorType::isIntegerDimParam(outDim)) {
            return WalkResult::advance();
          }
        }
      }

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
    MOGG::MojoRegistry registry(getOperation());

    // Run MOGG lowering on any valid operation
    auto result = getOperation().walk([&](Operation *op) {
      OpBuilder insertPoint{op};
      auto maybeOp = lowerOpToMOGG(op, insertPoint, registry,
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

    // The device-infos mapping is only needed during MO-level passes.
    // Remove it so it doesn't surface in downstream formats (e.g. MEF).
    getOperation()->removeAttr(MO::DeviceInfoMapping::kAttrName);
  }
};

} // namespace
