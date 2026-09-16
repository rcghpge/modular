//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Input-fusion wrapper for a sole-use `map.iter.foreach` producer fusing into a
// `map.iter.opaque` consumer's existing prologue, so e.g.
// `gather(add(a, b), indices)` becomes one kernel whose data-operand prologue
// computes the add inline. A view producer is `ViewFuser`'s to claim first.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Support/InputFusion.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"

#include "mlir/Support/LLVM.h"

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEPROLOGUE
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

struct FuseProloguePass : public MAP::impl::FusePrologueBase<FuseProloguePass> {
  using Base::Base;
  FuseProloguePass(const FuseProloguePass &other) : Base(other) {}

  void runOnOperation() override {
    InputFusionConfig config;
    // A foreach producer fusing into an opaque (the load lives in the opaque's
    // prologue; a foreach-body load would have no opaque parent here).
    config.shouldFuse = [](StoreOp srcStore, LoadOp dstLoad) {
      return srcStore->getParentOfType<IterForeachOp>() &&
             dstLoad->getParentOfType<IterOpaqueOp>();
    };
    // Only a sole-use producer fuses into a prologue.
    config.canDuplicate = [](IterForeachOp) { return false; };
    numKernelsFused = fuseInput(getOperation(), config);
  }

  Statistic numKernelsFused{this, "num-kernels-fused",
                            "Number of kernels fused"};
};

} // namespace

} // namespace M::MAP
