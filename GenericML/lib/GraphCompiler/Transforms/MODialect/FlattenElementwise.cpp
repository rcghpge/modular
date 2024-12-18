//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Analysis/MODialect/ParametricTopologyAnalysis.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

namespace M::MO {

#define GEN_PASS_DEF_FLATTENELEMENTWISE
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

template <typename ElemwiseInterfaceT>
void maybeFlattenElemwiseOp(ElemwiseInterfaceT elemwise, OpBuilder &builder) {
  TypedValue<TensorType> res = elemwise.getResult();
  TensorType tt = res.getType();

  // abort if this elementwise op is already flattened
  if (tt.getRank() <= 1)
    return;

  // abort if we can't flatten shape due to unknown dim
  if (tt.hasUnknownShape())
    return;

  // abort if the elemwise op cannot have its resulting tensor type changed
  // trivially.
  if (isa<MO::CastOp, MO::RangeOp, MO::UnsqueezeShapeOp>(elemwise))
    return;

  // compute shape of new flattened tensor
  MOSH::ShapeAttr shape = tt.getTypedShapeAttr();
  TypedAttr flattenedSize = shape.getSymbolicNumElements();
  MOSH::ShapeAttr newShape =
      MOSH::ShapeAttr::get({flattenedSize}, shape.getType());
  TensorType newTt = TensorType::get(newShape, tt.getDtype(), tt.getDeviceRef(),
                                     tt.getMetadata());

  // add reshapes for each operand
  for (auto [i, operand] : llvm::enumerate(elemwise->getOperands())) {
    builder.setInsertionPoint(elemwise);
    auto newReshapeOp = builder.create<MO::Static::ReshapeOp>(elemwise.getLoc(),
                                                              newTt, operand);

    elemwise->setOperand(i, newReshapeOp.getResult());
  }

  // add reshape for result
  builder.setInsertionPointAfter(elemwise);
  auto newReshapeOp =
      builder.create<MO::Static::ReshapeOp>(elemwise.getLoc(), tt, res);
  res.replaceAllUsesExcept(newReshapeOp.getResult(), {newReshapeOp});

  // update elementwise op to use flattened shape
  res.setType(newTt);
}

struct FlattenElementwise
    : public impl::FlattenElementwiseBase<FlattenElementwise> {
  using FlattenElementwiseBase::FlattenElementwiseBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module->getContext();
    OpBuilder builder{ctx};

    module->walk([&](ElementWiseUnary elemwise) {
      maybeFlattenElemwiseOp(elemwise, builder);
    });
    module->walk([&](ElementWiseBinary elemwise) {
      maybeFlattenElemwiseOp(elemwise, builder);
    });
  }
};

} // namespace M::MO
