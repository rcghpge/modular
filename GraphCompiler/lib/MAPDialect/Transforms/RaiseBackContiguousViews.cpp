//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Raises unfused slice/reshape `map.iter.foreach` view kernels back to
// MO/MOGG ops the pre-advanced-fusion pipeline already lowers without a
// copy. Mirrors the raise-back step of the legacy `mogg-fuse-views` pass
// (`GraphCompiler/lib/MOGGDialect/Transforms/FuseViews.cpp`), scoped the same
// way: slice and reshape only.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_RAISEBACKCONTIGUOUSVIEWS
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// Returns the sole `TransformOpTy` op computing `op`'s load index if `op`'s
// body is exactly what MOToMAP's lowering produces for a standalone,
// never-fused view kernel of that kind -- one transform op consuming the
// loop's own index, feeding a single load, feeding the store -- and null
// otherwise. `ViewFuser` always changes this shape (more ops) when it fuses
// anything into or out of a kernel, so matching the exact 3-op shape is
// sufficient to know this kernel was never touched.
template <typename TransformOpTy>
TransformOpTy matchUnfusedView(IterForeachOp op) {
  if (op.getNumResults() != 1)
    return {};
  Block &body = op.getBody().front();
  if (body.getOperations().size() != 3)
    return {};
  auto it = body.begin();
  auto transform = dyn_cast<TransformOpTy>(&*it++);
  auto load = dyn_cast<LoadOp>(&*it++);
  auto store = dyn_cast<StoreOp>(&*it);
  if (!transform || !load || !store)
    return {};
  return transform;
}

// Mirrors the legacy raise-back's own heuristic for identifying "the" sliced
// dimension: the first input dim of size != 1. Returns the input rank if
// every input dim has size 1 (no such dimension).
int64_t findSliceDim(ArrayRef<int64_t> inDims) {
  for (auto [i, dim] : llvm::enumerate(inDims))
    if (dim != 1)
      return i;
  return inDims.size();
}

// Returns the sliced dimension `k` if `slice` is a contiguous slice: every
// dimension but `k` keeps its input's full extent, and `k`'s start/stop/step
// are all statically known with step 1. Requires every input/output dim to
// be statically known too -- a slice with any symbolic dim always takes the
// general (already zero-copy) MAPToMOGG path instead of raising back.
std::optional<int64_t> matchContiguousSliceDim(SliceIndexOp slice) {
  MAP::SliceSpecAttr spec = slice.getSpec();
  SmallVector<int64_t> inDims = spec.getSignature().getIn().getIntDims();
  SmallVector<int64_t> outDims = spec.getSignature().getOut().getIntDims();
  if (llvm::any_of(inDims, ShapedType::isDynamic) ||
      llvm::any_of(outDims, ShapedType::isDynamic))
    return std::nullopt;

  int64_t k = findSliceDim(inDims);
  for (auto [i, inDim] : llvm::enumerate(inDims))
    if (static_cast<int64_t>(i) != k && inDim != outDims[i])
      return std::nullopt;

  ArrayRef<int64_t> starts = spec.getStarts().asArrayRef();
  ArrayRef<int64_t> stops = spec.getStops().asArrayRef();
  ArrayRef<int64_t> steps = spec.getSteps().asArrayRef();
  if (k == static_cast<int64_t>(inDims.size()) ||
      ShapedType::isDynamic(starts[k]) || ShapedType::isDynamic(stops[k]) ||
      steps[k] != 1)
    return std::nullopt;

  return k;
}

// Raises an unfused, contiguous `map.iter.foreach` slice kernel back to
// `mogg.slice`, a zero-copy buffer slice `MOGGToMGP` lowers to
// `mgp.tensor.slice`. Returns whether `op` was raised.
bool raiseContiguousSlice(IterForeachOp op) {
  SliceIndexOp slice = matchUnfusedView<SliceIndexOp>(op);
  if (!slice)
    return false;
  std::optional<int64_t> k = matchContiguousSliceDim(slice);
  if (!k)
    return false;

  MAP::SliceSpecAttr spec = slice.getSpec();
  ArrayRef<int64_t> starts = spec.getStarts().asArrayRef();
  ArrayRef<int64_t> stops = spec.getStops().asArrayRef();

  Value input = op.getInputs()[0];
  Value startsOperand = op.getInputs()[1];
  auto hostDevice =
      cast<MO::TensorType>(startsOperand.getType()).getDeviceRef();

  OpBuilder builder(op);
  Value startK =
      MO::ConstantOp::create(builder, op.getLoc(), starts[*k], hostDevice);
  Value stopK =
      MO::ConstantOp::create(builder, op.getLoc(), stops[*k], hostDevice);

  auto resultType = cast<MO::TensorType>(op.getResultTypes()[0]);
  auto sliceOp = MOGG::SliceOp::create(builder, op.getLoc(), resultType, input,
                                       startK, stopK);
  op->replaceAllUsesWith(sliceOp->getResults());
  op->erase();
  return true;
}

// Raises an unfused `map.iter.foreach` reshape kernel back to
// `mo.static.reshape` -- unconditional, unlike slice, since `MOGGToMGP`
// always turns a reshape into a non-copying primitive regardless of
// contiguity. Returns whether `op` was raised.
bool raiseReshape(IterForeachOp op) {
  ReshapeIndexOp reshape = matchUnfusedView<ReshapeIndexOp>(op);
  if (!reshape)
    return false;

  OpBuilder builder(op);
  auto reshapeOp = MO::Static::ReshapeOp::create(
      builder, op.getLoc(), op.getResultTypes(), op.getInputs());
  op->replaceAllUsesWith(reshapeOp->getResults());
  op->erase();
  return true;
}

struct RaiseBackContiguousViewsPass
    : public MAP::impl::RaiseBackContiguousViewsBase<
          RaiseBackContiguousViewsPass> {
  using Base::Base;

  void runOnOperation() override {
    MO::GraphOp graphOp = getOperation();
    // Walk rather than iterate the body's direct ops: a map.iter.foreach
    // view kernel can sit inside control flow (e.g. mo.parallel), and
    // iterating only the graph's own top-level ops would miss those.
    SmallVector<IterForeachOp> foreachOps;
    graphOp.walk([&](IterForeachOp op) { foreachOps.push_back(op); });
    for (IterForeachOp op : foreachOps) {
      if (raiseContiguousSlice(op)) {
        ++numSlicesRaised;
      } else if (raiseReshape(op)) {
        ++numReshapesRaised;
      }
    }
  }
};

} // namespace

} // namespace M::MAP
