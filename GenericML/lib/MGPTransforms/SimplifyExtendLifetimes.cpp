//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"

#include "GraphRT/GRTDialect/GRTOps.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_SIMPLIFYEXTENDLIFETIMES
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

} // namespace M::MGP

namespace M::MGP {

namespace {

struct SimplifyExtendLifetimes
    : public MGP::impl::SimplifyExtendLifetimesBase<SimplifyExtendLifetimes> {
  using SimplifyExtendLifetimesBase::SimplifyExtendLifetimesBase;

  void runOnOperation() override {
    getOperation()->walk([&](MGP::ExtendLifetimeOp extend) {
      TypedValue<MEF::ChainType> chain = extend.getGate();
      MOGG::GenericCompiledRegionOp kernel =
          chain.getDefiningOp<MOGG::GenericCompiledRegionOp>();
      if (!kernel)
        return;

      kernel->insertOperands(kernel->getNumOperands(), extend.getInputs());
      for (Value v : extend.getInputs()) {
        kernel.getBody().addArgument(v.getType(), v.getLoc());
        kernel.setInOutSignatureAttr(kernel.getInOutSignature().append(
            {InOutSignatureAttr::InOutSemantics::kNone}));
      }

      extend.getResult().replaceAllUsesWith(chain);
      extend.erase();
    });
  }
};

} // namespace

} // namespace M::MGP
