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

    auto devRef = slice.getDeviceRef();

    auto tryCreateSliceDimOuter = [&]() {
      // In some special cases (for example axis=0, step=1, we can lower to a
      // simple pointer offset implemented by SliceDimOuter)

      // We don't want to do this if the input is a view, since we don't want to
      // block view fusion
      //
      // We also don't want to do this in the case that the tensor was produced
      // by a mutable.load op as in the case of a load + slice a full copy of
      // the tensor will be materialized as only ops that lower to a mogg.kernel
      // can be fused into.
      Operation *definingOp = slice.getInput().getDefiningOp();
      if (definingOp &&
          isa<BroadcastToOp, TransposeOp, SliceOp, SliceDimOp, MutableLoadOp>(
              definingOp))
        return false;

      // This can only lower to an offset if all leading dimensions are 1
      for (int j = 0; j < asInt; ++j) {
        if (inputType.getIntDim(j) != 1)
          return false;
      }

      // Give up if the step doesn't come from a constant
      auto constStep = step.getInput().getDefiningOp<ConstantOp>();
      if (!constStep)
        return false;

      auto maybeStepValue = tryGetConstantBlob<int64_t>(constStep.getValue(),
                                                        stepScalar.getDtype());
      // Give up if we can't get the step values
      if (!maybeStepValue)
        return false;

      // Give up if any steps leading up to and including this axis are not 1
      for (int j = 0; j <= asInt; ++j) {
        if (maybeStepValue.value()[j] != 1)
          return false;
      }

      auto lastTy = cast<TensorType>(lastVal.getType());

      // Remove the leading singleton dimensions
      auto squeezedInputTy =
          lastTy.cloneWith(inputType.getDims().drop_front(i));
      lastVal = builder.createOrFold<Static::ReshapeOp>(
          slice.getLoc(), squeezedInputTy, lastVal);

      ArrayRef<TypedAttr> shapeRef(shape);
      auto squeezedSliceTy = slice.getType().cloneWith(shapeRef.drop_front(i));

      // Create SliceDimOuter op now that the sliced dimension is the outermost
      SliceDimOuterOp newSlice = builder.create<SliceDimOuterOp>(
          slice.getLoc(), squeezedSliceTy, lastVal, start.getResult(),
          stop.getResult());

      if (devRef)
        newSlice.setDeviceRef(devRef);

      lastVal = newSlice.getResult();

      // Restore the leading singleton dimensions
      auto unsqueezedSliceTy = slice.getType().cloneWith(shape);
      lastVal = builder.createOrFold<Static::ReshapeOp>(
          slice.getLoc(), unsqueezedSliceTy, lastVal);

      return true;
    };

    if (!tryCreateSliceDimOuter()) {
      // Create the new slice along just one dimension.
      SliceDimOp newSlice = builder.create<SliceDimOp>(
          slice.getLoc(), slice.getType().cloneWith(shape), lastVal, axis,
          start.getResult(), stop.getResult(), step.getResult());

      if (devRef)
        newSlice.setDeviceRef(devRef);

      lastVal = newSlice.getResult();
    }
  }

  slice.replaceAllUsesWith(lastVal);
  slice.erase();
}

static void decomposeReshape(Static::ReshapeOp reshape, OpBuilder &builder) {
  builder.setInsertionPointAfter(reshape);
  TensorType outType = reshape.getType();

  // Remove all `1`s from the reshape by adding a singleton dimension after.
  SmallVector<TypedAttr> newOutShape;
  MOSH::ShapeAttr outShapeAttr = outType.getTypedShapeAttr();
  for (size_t idx = 0; idx < outShapeAttr.getSize(); ++idx) {
    TypedAttr dimVal = outShapeAttr[idx];
    auto asInt = dyn_cast<IntegerAttr>(dimVal);
    if (!asInt || asInt.getValue() != 1)
      newOutShape.push_back(dimVal);
  }

  TensorType outShapeWithoutOnes = outType.cloneWith(newOutShape);

  TypedValue<TensorType> output = reshape.getResult();
  output.setType(outShapeWithoutOnes);

  // The shape to progressively add the ones back in.
  TensorType rollingShape = outShapeWithoutOnes;

  // Create a new reshape built up from just these new ranks.
  for (size_t idx = 0; idx < outShapeAttr.getSize(); ++idx) {
    TypedAttr dimVal = outShapeAttr[idx];
    auto asInt = dyn_cast<IntegerAttr>(dimVal);
    if (!asInt || asInt.getValue() != 1)
      continue;

    MOSH::ShapeAttr shapeAttr = rollingShape.getTypedShapeAttr();
    SmallVector<TypedAttr> newShape{shapeAttr.getValues()};
    newShape.insert(newShape.begin() + idx, builder.getIndexAttr(1));
    rollingShape = rollingShape.cloneWith(newShape);

    auto op = builder.create<AddSingletonDimOp>(
        reshape.getLoc(), rollingShape, output,
        /*axis*/ builder.getIndexAttr(idx));

    output.replaceAllUsesExcept(op.getResult(), op);
    output = op.getResult();
  }

  // If the reshape is now a no-op, remove it.
  if (reshape.getType() == reshape.getInput().getType()) {
    reshape.getResult().replaceAllUsesWith(reshape.getInput());
    reshape.erase();
    return;
  }

  size_t inRank = reshape.getInput().getType().getRank();
  size_t outRank = reshape.getType().getRank();

  // Try split dimension. Super specific for 2D -> 3D split for now.
  if (inRank == 2 && outRank == 3) {
    MOSH::ShapeAttr outShapeAttr = reshape.getType().getTypedShapeAttr();
    MOSH::ShapeAttr inShapeAttr =
        reshape.getInput().getType().getTypedShapeAttr();

    // Pattern match [N, K] -> [n1, n2, K]
    if (outShapeAttr[0] != inShapeAttr[0] &&
        outShapeAttr[1] != inShapeAttr[1] &&
        outShapeAttr[1] != inShapeAttr[0] &&
        outShapeAttr[2] == inShapeAttr[1]) {

      Value split = builder.create<SplitDimOp>(
          reshape.getLoc(), reshape.getType(), reshape.getInput(),
          /*axis*/ builder.getIndexAttr(0));
      reshape.replaceAllUsesWith(split);
      reshape.erase();
    }
  }
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
      else if (auto reshape = dyn_cast<Static::ReshapeOp>(op))
        decomposeReshape(reshape, builder);
    });
  }
};

} // namespace
