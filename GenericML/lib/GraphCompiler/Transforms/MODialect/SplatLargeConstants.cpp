//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"

namespace M::MO {

#define GEN_PASS_DEF_SPLATLARGECONSTANTS
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

struct SplatLargeConstants
    : public impl::SplatLargeConstantsBase<SplatLargeConstants> {

  SplatLargeConstants() : SplatLargeConstantsBase() {}

  SplatLargeConstants(const SplatLargeConstantsOptions &opts)
      : SplatLargeConstantsBase(opts) {}

  void runOnOperation() override {
    GraphOp graphOp = getOperation();
    MLIRContext *ctx = graphOp->getContext();
    OpBuilder builder{graphOp.getContext()};
    auto shapeType = ShapeType::get(ctx);

    graphOp.walk([&](ConstantOp constantOp) {
      auto constantValue = constantOp.getValue();
      if (constantValue.size() < numElements)
        return;

      // Scalar constants have nothing to broadcast.
      if (constantValue.size() == 1)
        return;

      builder.setInsertionPoint(constantOp);
      Location loc = constantOp->getLoc();
      auto elementType = constantValue.getElementType();
      auto constantOpType = constantOp.getType();

      int bitwidth = elementType.getIntOrFloatBitWidth();
      int bytewidth = bitwidth / 8;
      SmallVector<uint8_t> zeroValue(bytewidth, 0);
      auto newArrayType = ArrayType::get(1, elementType);
      auto splatValue = M::ArrayElementsAttr::get(zeroValue, newArrayType);
      auto newConstantType =
          constantOpType.cloneWith(ShapeAttr::get({1}, shapeType));
      auto newConstantOp =
          ConstantOp::create(builder, loc, newConstantType, splatValue);
      auto broadcastOp = Static::BroadcastToOp::create(
          builder, loc, constantOpType, newConstantOp);
      constantOp->replaceAllUsesWith(broadcastOp);
      constantOp->erase();
    });
  }
};

} // namespace M::MO
