//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_DUMPMOJO
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

struct DumpMojo : public impl::DumpMojoBase<DumpMojo> {
  using Base::Base;

  void runOnOperation() override {
    auto resourceManager = DenseResourceElementsHandle::getManagerInterface(
        getOperation()->getContext());
    auto mojoTextResource = resourceManager.getBlobManager().lookup("mojo");
    if (!mojoTextResource)
      return;

    auto mojoTextRef = mojoTextResource->getBlob()->getData();
    auto mojoText = StringRef(mojoTextRef.data(), mojoTextRef.size());
    llvm::errs() << mojoText << "\n";
  }
};

} // namespace M::MOGG
