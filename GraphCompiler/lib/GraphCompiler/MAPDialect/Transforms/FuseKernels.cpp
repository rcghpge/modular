//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Fuses MAP kernels by inlining producers into consumers. Scaffolding only.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/Conversion/MOToMAP.h"
#include "GraphCompiler/GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEKERNELS
#include "GraphCompiler/GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

struct FuseKernelsPass : public MAP::impl::FuseKernelsBase<FuseKernelsPass> {
  using Base::Base;

  void runOnOperation() override {
    if (!MO::useAdvancedFusion())
      return;

    // TODO: run the advanced fusion patterns over the MAP dialect. No-op for
    // now.
  }
};

} // namespace

} // namespace M::MAP
