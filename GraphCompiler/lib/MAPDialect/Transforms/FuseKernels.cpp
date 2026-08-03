//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Fuses MAP kernels by inlining producers into consumers.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/MOToMAP/MOToMAP.h"
#include "GraphCompiler/MAPDialect/Support/ElementwiseFuser.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEKERNELS
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

struct FuseKernelsPass : public MAP::impl::FuseKernelsBase<FuseKernelsPass> {
  using Base::Base;

  void runOnOperation() override {
    if (!MO::useAdvancedFusion())
      return;

    getOperation()->walk(
        [&](MO::GraphOp graphOp) { ElementwiseFuser(graphOp).fuse(); });
  }
};

} // namespace

} // namespace M::MAP
