//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MODevices.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Pass/Pass.h"

namespace M::MO {

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_DECOMPOSEVIEWOPS
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

} // namespace M::MO

using namespace M;
using namespace M::MO;

namespace {

static void decomposeSlice(SliceOp slice, OpBuilder &builder) {
  TensorType inputType = slice.getInput().getType();
  if (!inputType.hasRank())
    return;
  size_t rank = inputType.getRank();

  TensorValue origStart = slice.getStart();
  TensorValue origStop = slice.getStop();
  TensorValue origStep = slice.getStep();

  SmallVector<int64_t> scalarShape;
  // Get the scalar equivalent type for the indices. We will slice out the
  // correct value for each.
  TensorType startScalar = origStart.getType().cloneWith(
      scalarShape, origStart.getType().getElementType());
  TensorType stopScalar = origStop.getType().cloneWith(
      scalarShape, origStop.getType().getElementType());
  TensorType stepScalar = origStep.getType().cloneWith(
      scalarShape, origStep.getType().getElementType());

  MO::ConstantOp one =
      builder.create<MO::ConstantOp>(slice.getLoc(), static_cast<int64_t>(1));

  // We slice the tensor one dimension at a time and so only update the shape
  // one dimension at a time as well.
  SmallVector<TypedAttr> shape;
  for (TypedAttr dim : inputType.getDims())
    shape.push_back(dim);

  auto zeroAxis = builder.getIndexAttr(0);

  Value lastVal = slice.getInput();
  for (size_t i = 0; i < rank; ++i) {
    auto axis = builder.getIndexAttr(i);

    int64_t asInt = i;

    // We are slicing the start / stop / step one dimenson at a time. For dim 0
    // we slice [0:1:1], dim 1 [1:2:1] and so on.
    MO::ConstantOp startPos =
        builder.create<MO::ConstantOp>(slice.getLoc(), asInt);
    MO::ConstantOp stopPos =
        builder.create<MO::ConstantOp>(slice.getLoc(), asInt + 1);

    // Slice out the start, step and stop.
    auto start =
        builder.create<SliceDimOp>(slice.getLoc(), startScalar, origStart,
                                   zeroAxis, startPos, stopPos, one);
    auto stop = builder.create<SliceDimOp>(slice.getLoc(), stopScalar, origStop,
                                           zeroAxis, startPos, stopPos, one);
    auto step = builder.create<SliceDimOp>(slice.getLoc(), stepScalar, origStep,
                                           zeroAxis, startPos, stopPos, one);

    shape[i] = slice.getType().getDim(i);

    // Create the new slice along just one dimension.
    SliceDimOp newSlice = builder.create<SliceDimOp>(
        slice.getLoc(), slice.getType().cloneWith(shape), lastVal, axis,
        start.getResult(), stop.getResult(), step.getResult());

    if (auto devRef = slice.getDeviceRef())
      newSlice.setDeviceRef(devRef);

    lastVal = newSlice.getResult();
  }

  slice.replaceAllUsesWith(lastVal);
  slice.erase();
}

struct DecomposeViewsPass
    : public impl::DecomposeViewOpsBase<DecomposeViewsPass> {
  using DecomposeViewOpsBase::DecomposeViewOpsBase;

  void runOnOperation() override {
    OpBuilder builder{&getContext()};
    getOperation().walk([&](Operation *op) {
      builder.setInsertionPoint(op);

      if (auto sliceOp = dyn_cast<SliceOp>(op))
        decomposeSlice(sliceOp, builder);
    });
  }
};

} // namespace
