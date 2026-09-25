//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Input-fusion wrapper for a pure-elementwise producer fusing into a
// pure-elementwise consumer foreach -- the case that always shares one index
// space. A view on either side is `ViewFuser`'s job.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Support/InputFusion.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MODevices.h"

#include "mlir/Support/LLVM.h"

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEELEMENTWISE
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// Above this many elements, re-reading a tensor operand is expensive. Mirrors
// MOGG's FuseElementwiseKernels.cpp (JSC-30 / MSTDL-2759).
constexpr int64_t kLargeTensorElementThreshold = 4096;

bool isLargeTensorOperand(Type type) {
  auto tensorType = dyn_cast<MO::TensorType>(type);
  if (!tensorType)
    return false;
  std::optional<int64_t> numElements = tensorType.getNumElements();
  return !numElements.has_value() ||
         *numElements > kLargeTensorElementThreshold;
}

// Cheap to duplicate iff at most one input is a large tensor: duplication
// re-reads all inputs once per extra consumer, a win only when at most one
// re-read is expensive.
bool isCheapToDuplicate(IterForeachOp op) {
  int numLargeOperands = 0;
  for (Value input : op.getInputs())
    if (isLargeTensorOperand(input.getType()))
      if (++numLargeOperands > 1)
        return false;
  return true;
}

struct FuseElementwisePass
    : public MAP::impl::FuseElementwiseBase<FuseElementwisePass> {
  using Base::Base;
  FuseElementwisePass(const FuseElementwisePass &other) : Base(other) {}

  void runOnOperation() override {
    InputFusionConfig config;
    // Both kernels pure elementwise (no index transform). The consumer being a
    // foreach falls out of the load having a foreach parent.
    config.shouldFuse = [](StoreOp srcStore, LoadOp dstLoad) {
      auto src = srcStore->getParentOfType<IterForeachOp>();
      auto dst = dstLoad->getParentOfType<IterForeachOp>();
      return src && dst && !isViewForeach(src) && !isViewForeach(dst);
    };
    // Never duplicate a host kernel (mirrors MOGG).
    config.canDuplicate = [](IterForeachOp src) {
      return !MO::isOnHost(src) && isCheapToDuplicate(src);
    };
    // Bound duplication so a cheap multi-use producer with fan-out can't grow a
    // few enormous kernels (mirrors MOGG's kMaximumFusionSetSize).
    config.maxFusionSetSize = maxFusionSetSize;
    numKernelsFused = fuseInput(getOperation(), config);
  }

  Statistic numKernelsFused{this, "num-kernels-fused",
                            "Number of kernels fused"};
};

} // namespace

} // namespace M::MAP
