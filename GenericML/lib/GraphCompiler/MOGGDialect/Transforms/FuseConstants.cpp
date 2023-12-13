//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSECONSTANTS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

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
