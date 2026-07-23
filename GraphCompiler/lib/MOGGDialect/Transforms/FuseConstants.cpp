//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSECONSTANTS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct FuseConstants : public impl::FuseConstantsBase<FuseConstants> {
  using FuseConstantsBase::FuseConstantsBase;

  void runOnOperation() override {
    MO::GraphOp graph = getOperation();
    for (MOGG::KernelOp kernel : graph.getOps<MOGG::KernelOp>())
      fuseConstant(kernel);
  }
};

} // namespace

} // namespace M::MOGG
