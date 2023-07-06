//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPH_COMPILER_MOGGDIALECT_MOGG_HELPERS_H
#define GENERICML_GRAPH_COMPILER_MOGGDIALECT_MOGG_HELPERS_H

#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"

#include "mlir/IR/IRMapping.h"

using namespace M;

namespace M::MOGG {

namespace {

/// Inline the given lambda into the call site of the invoke and return the new
/// results which have replaced the old operation.
[[maybe_unused]] SmallVector<Value> inlineInvoke(MOGG::LambdaInvokeOp invokeOp,
                                                 MOGG::LambdaOp invokeTarget) {
  IRMapping mapper;

  // Remap the values on the call site to the values inside the block.
  for (auto [callValue, bodyValue] :
       llvm::zip(invokeOp.getOperands(),
                 invokeTarget.getBody().front().getArguments()))
    mapper.map(bodyValue, callValue);

  mlir::OpBuilder inserter{invokeOp};

  // Clone everything in the lambda.
  for (Operation &op : invokeTarget.getBody().front()) {
    // Don't add the terminator obviously
    if (isa<MOGG::OutputOp>(op)) {
      assert(invokeOp->getNumResults() == op.getNumOperands() &&
             "[LAMBDA_INLINE] Return doesn't have same number of results as "
             "caller.");
      if (invokeOp.use_empty())
        continue;

      SmallVector<Value> newVals;
      for (Value v : op.getOperands())
        newVals.push_back(mapper.lookup(v));
      invokeOp.replaceAllUsesWith(newVals);
      return newVals;
    } else {
      inserter.insert(op.clone(mapper));
    }
  }

  return {};
}

[[maybe_unused]] MOGG::CallOp createMOGGReshape(mlir::OpBuilder &builder,
                                                mlir::Location loc,
                                                Value tensorIn,
                                                MO::TensorType newShape) {
  MLIRContext *ctx = builder.getContext();
  // Create the new output shape.
  auto shape = MOGG::ShapeType::get(ctx, newShape.getTypedShapeAttr());

  // Materialize the runtime value of the shape in the MOGG kernel.
  auto getShape = builder.create<MOGG::ParamShapeToValueOp>(loc, shape);
  Value runtimeShape = getShape.getResult();

  // Add the non-blocking flag.
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("single_thread_blocking_override",
                                       builder.getBoolAttr(true)));
  mlir::DictionaryAttr kgenParams = builder.getDictionaryAttr(attrs);

  // Create the call to the new reshape.
  return builder.create<MOGG::CallOp>(
      loc,
      /*outputTypes=*/TypeRange{newShape},
      /*kernelTarget=*/MO::Static::ReshapeOp::getOperationName(),
      /*inputOperands=*/ValueRange{tensorIn, runtimeShape},
      /*lambdaParams=*/mlir::ArrayAttr{},
      /*otherParams=*/kgenParams);
}

// Create a call to the slice kernel directly in MOGG.
[[maybe_unused]] MOGG::CallOp
createMOGGSlice(mlir::OpBuilder &builder, mlir::Location loc, Value tensorIn,
                Type outType, const SmallVector<int64_t> &start,
                const SmallVector<int64_t> &stop,
                const SmallVector<int64_t> &step) {
  // Create a constant for the axis and each of the dimensions being sliced.
  Value startVal = builder.create<MO::ConstantOp>(loc, start);
  Value stopVal = builder.create<MO::ConstantOp>(loc, stop);
  Value stepVal = builder.create<MO::ConstantOp>(loc, step);

  // Add the non-blocking flag.
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("single_thread_blocking_override",
                                       builder.getBoolAttr(true)));
  mlir::DictionaryAttr kgenParams = builder.getDictionaryAttr(attrs);

  // Create the call to the new reshape.
  return builder.create<MOGG::CallOp>(
      loc,
      /*outputTypes=*/TypeRange{outType},
      /*kernelTarget=*/MO::SliceOp::getOperationName(),
      /*inputOperands=*/ValueRange{tensorIn, startVal, stopVal, stepVal},
      /*lambdaParams=*/mlir::ArrayAttr{},
      /*otherParams=*/kgenParams);
}

} // namespace

} // namespace M::MOGG

#endif // GENERICML_GRAPH_COMPILER_MOGGDIALECT_MOGG_HELPERS_H
