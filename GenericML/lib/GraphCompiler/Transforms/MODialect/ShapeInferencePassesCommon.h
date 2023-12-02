//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_SHAPEINFERENCEPASSES_COMMON_H
#define GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_SHAPEINFERENCEPASSES_COMMON_H

#include "GenericML/GraphCompiler/MODialect/MODevices.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

using namespace M;
using namespace MO;
namespace M::MO::ShapeInferencePassCommon {

//===----------------------------------------------------------------------===//
// Common helpers
//===----------------------------------------------------------------------===//

/// Extract the value (of MOSH::ShapeType) represented by `val` if possible,
/// else return `null`.
/// NOTE possible side-effect: if `val` is a tensor constant, a
/// `KGEN::ParamConstantOp` will be created to hold the shape value.
Value tryGetShapeValue(Value val, Location loc);

/// Extract the `MOSH::ShapeAttr` represented by `val` if possible, else return
/// `null`.
/// NOTE possible side-effect: if `val` is a tensor constant, a
/// `KGEN::ParamConstantOp` will be created with `loc` to hold the shape value.
ShapeAttr tryGetShapeAttr(Value val, Location loc);

/// Run all the elaboration passes on given module, and return whether it
/// succeeded.
LogicalResult elaborateModule(ModuleOp moduleOp);

/// Copy the elaborated shape functions into the front of `moduleOp` and make
/// sure they can be symbolically DCEed after getting inlined later.
void copyElaboratedShapeFuncsToFrontOfModule(
    ShapeFuncMaterializationState &state, ModuleOp moduleOp);

/// Checks if the attribute is a static invocation using POC::Apply.
ErrorOrSuccess isApplyOperator(KGEN::ParamOperatorAttr attr);

//===----------------------------------------------------------------------===//
// Common Patterns
//===----------------------------------------------------------------------===//

/// A pattern that folds/flattens a chain of rebinds into a tree of rebinds that
/// share the same input.
///
/// It serves 2 purposes:
/// 1. Get rid of intermediate rebinds to unknown shapes (see example below).
/// 2. Minimizes the number of rebinds we must look through when checking for
/// constants.
///
/// Example:
///
/// input : tensor<[D0]>
///
///   reb1 = rebind(input) : tensor<[D0]> -> tensor<[?]>
///   res  = rebind(reb1)  : tensor<[?]>  -> tensor<[D2]>
/// ===>
///   reb1 = rebind(input) : tensor<[D0]> -> tensor<[?]>
///   res  = rebind(input) : tensor<[D0]> -> tensor<[D2]>
class FoldConsecutiveRebinds
    : public DeviceAwareRewritePattern<OpRewritePattern, RebindOp> {
  using DeviceAwareRewritePattern::DeviceAwareRewritePattern;

  LogicalResult
  deviceAwareMatchAndRewrite(RebindOp op,
                             DeviceAwareRewriter &rewriter) const override;
};

class MaterializeShapeFuncs
    : public mlir::OpInterfaceRewritePattern<ShapeInferenceMaterialization> {
public:
  using OpInterfaceRewritePattern<
      ShapeInferenceMaterialization>::OpInterfaceRewritePattern;

  MaterializeShapeFuncs(MLIRContext *ctx, ShapeFuncMaterializationState *state);

  LogicalResult matchAndRewrite(ShapeInferenceMaterialization op,
                                PatternRewriter &rewriter) const final;

private:
  ShapeFuncMaterializationState *state;
};

} // namespace M::MO::ShapeInferencePassCommon

#endif // GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_SHAPEINFERENCEPASSES_COMMON_H
