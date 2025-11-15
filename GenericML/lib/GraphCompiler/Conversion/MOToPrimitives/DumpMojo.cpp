//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_DUMPMOJO
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

namespace M::MGP {

struct DumpMojo
    : public MOGG::MojoLibraryAnalysisPass<impl::DumpMojoBase<DumpMojo>> {
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

} // namespace M::MGP
