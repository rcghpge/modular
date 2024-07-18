//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Implementations of `Staticization` interface for various `mo` ops.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/ShapeFunctions.h"
#include "GenericML/Support/Dims.h"

namespace M::MO {

//===----------------------------------------------------------------------===//
// InvokeShapeFuncOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> InvokeShapeFuncOp::getDataDependentInputIndices() {
  SmallVector<int32_t> idxs;
  if (ArrayAttr arrayAttr = getDataDeptTensorsAttr()) {
    for (llvm::APInt index : arrayAttr.getAsValueRange<IntegerAttr>())
      idxs.push_back(index.getLimitedValue());
  }
  return idxs;
}

ShapeMaterializeResult
InvokeShapeFuncOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  StringRef shapeFuncName = getShapeFuncName();
  auto shapeMaterializeFunc = tryGetShapeMaterializeFunc(shapeFuncName);

  // Give up if this shape function doesn't have a C++ shape function impl.
  if (!shapeMaterializeFunc.has_value())
    return ShapeMaterializeResult::nonMaterializable();

  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  DictionaryAttr kgenParams = getKgenParamsAttr();
  return shapeMaterializeFunc.value()(inputAttrs, kgenParams, graphOp, loc);
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> ReshapeOp::getDataDependentInputIndices() { return {1}; }

ShapeMaterializeResult
ReshapeOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr targetShape = inputAttrs[1];
  return materializeReshapeShape(graphOp, loc, inputShape, targetShape);
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> SplitOp::getDataDependentInputIndices() { return {1, 2}; }

ShapeMaterializeResult
SplitOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr splitSizes = inputAttrs[1];
  TypedAttr axis = inputAttrs[2];
  return materializeSplitOutputShapes(graphOp, loc, inputShape, splitSizes,
                                      axis);
}

} // namespace M::MO
