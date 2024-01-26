//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ShapeInferencePassesCommon.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/Package/Package.h"
#include "Support/MArchTarget/Host.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/TargetSelect.h"

using namespace M;
using namespace MO;

namespace M::MO::ShapeInferencePassCommon {

//===----------------------------------------------------------------------===//
// Common helpers
//===----------------------------------------------------------------------===//

Value tryGetShapeValue(Value val, Location loc) {
  // Skip at most a single intermediate rebind
  if (auto rebindOp = val.getDefiningOp<RebindOp>())
    val = rebindOp.getInput();

  if (auto toTensorOp = val.getDefiningOp<Shape::ToTensorOp>())
    return toTensorOp.getInput();

  auto tensorVal = dyn_cast<TensorValue>(val);
  if (!tensorVal)
    return {};

  // Never treat a `!mo.tensor<[1], f32>` as a shape.
  if (!tensorVal.getType().getElementType().isIntOrIndex())
    return {};

  if (auto maybeIntVec = tryGetConstantIntVector(tensorVal)) {
    MLIRContext *ctx = val.getContext();
    auto shapeType = ShapeType::get(ctx);
    auto shapeAttr = ShapeAttr::get(maybeIntVec.value(), shapeType);
    // We don't insert via pattern rewriter because it might prevent rewriter
    // from reaching convergence (e.g., if this is called by LowerConcat, and
    // some other concat inputs aren't constant/shape)
    OpBuilder builder(ctx);
    builder.setInsertionPointAfterValue(val);
    return builder.create<KGEN::ParamConstantOp>(loc, shapeAttr);
  }

  return {};
}

ShapeAttr tryGetShapeAttr(Value val, Location loc) {
  auto shapeVal = tryGetShapeValue(val, loc);
  if (!shapeVal)
    return {};

  auto paramConstOp = shapeVal.getDefiningOp<KGEN::ParamConstantOp>();
  if (!paramConstOp)
    return {};

  return dyn_cast<ShapeAttr>(paramConstOp.getValue());
}

LogicalResult elaborateModule(ModuleOp moduleOp) {
  // TODO refactor this with KGENStubBuilder
  MLIRContext *ctx = moduleOp.getContext();
  llvm::InitializeNativeTarget();
  // TODO propagate the target info properly for cross compilation
  ErrorOr<TargetInfoAttr> target =
      getTargetInfoFor(ctx, llvm::sys::getDefaultTargetTriple(),
                       llvm::sys::getHostCPUName(), getHostCPUFeatures());
  if (target.isError())
    return failure();

  setTargetInfo(moduleOp, *target);

  // Extend the environment (defines) for the module.
  KGEN::extendWithModularEnvAttr(moduleOp);

  auto ownedRuntime = LLCL::createRuntimeIfNeeded(
      LLCL::RuntimeOptions().withLeakCheckedAllocator());
  mlir::PassManager pm(ctx);
  KGEN::populateElaborateModulePasses(
      pm, *ownedRuntime, *target, {/*enableSearch=*/false},
      [](KGEN::PackageLinkOp packageLink, TargetInfoAttr) {
        // TODO(#22819): For now, do not compile packages on-demand, and instead
        // stitch the pre-elaborated module into the module being built.
        return KGEN::loadPreElaboratedBytecodeForLinking(
            packageLink.getPreElaborationModuleAttr());
      });
  return pm.run(moduleOp);
}

void copyElaboratedShapeFuncsToFrontOfModule(
    ShapeFuncMaterializationState &state, ModuleOp moduleOp) {
  for (KGEN::FuncOp shapeFunc : state.getUsedShapeFuncsAfterElaboration()) {
    KGEN::FuncOp shapeFuncCopy = shapeFunc.clone();
    shapeFuncCopy.setPrivate(); // Make sure it gets symbolic DCEed
    moduleOp.getBody()->push_front(shapeFuncCopy);
  }
}

ErrorOrSuccess isApplyOperator(KGEN::ParamOperatorAttr attr) {
  if (!attr)
    return Error("not a parameter operator");
  if (attr.getOpcode() != KGEN::POC::Apply)
    return Error("not an apply operator");
  return success();
}

//===----------------------------------------------------------------------===//
// Common Patterns
//===----------------------------------------------------------------------===//

LogicalResult FoldConsecutiveRebinds::deviceAwareMatchAndRewrite(
    RebindOp op, DeviceAwareRewriter &rewriter) const {
  Location loc = op->getLoc();
  auto producerOp = op.getInput().getDefiningOp<RebindOp>();
  if (!producerOp) {
    return rewriter.notifyMatchFailure(loc, "not preceded by other mo.rebind");
  }

  // Don't fold if producer input has unknown shape, because we want to
  // minimize rebind that takes input with `?` (shape propagation can't
  // propagate through `?`s). For instance,
  //
  // input : tensor<[?]>
  //
  //   r1 = rebind(input) : tensor<[?]>  -> tensor<[D1]>
  //   r2 = rebind(r1)    : tensor<[D1]> -> tensor<[D2]>
  // v.s.
  //   r1 = rebind(input) : tensor<[?]> -> tensor<[D1]>
  //   r2 = rebind(input) : tensor<[?]> -> tensor<[D2]>
  //
  // In the first case, shape propagation will understand that D1 == D2, but
  // in the second case, it will not.
  if (producerOp.getInput().getType().hasUnknownShape()) {
    return rewriter.notifyMatchFailure(
        loc, "preceding rebind has input with unknown shape");
  }

  rewriter.replaceOpWithNewOp<RebindOp>(op, op.getType(), producerOp.getInput(),
                                        op.getOutputParamDecls());
  return success();
}

MaterializeShapeFuncs::MaterializeShapeFuncs(
    MLIRContext *ctx, ShapeFuncMaterializationState *state)
    : OpInterfaceRewritePattern(ctx), state(state) {}
LogicalResult
MaterializeShapeFuncs::matchAndRewrite(ShapeInferenceMaterialization op,
                                       PatternRewriter &rewriter) const {
  state->setRewriter(rewriter);
  return op.materializeShapeInference(*state);
}

} // namespace M::MO::ShapeInferencePassCommon
