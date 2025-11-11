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
    uint64_t moduleIdx = 0;
    for (auto modelOp : getOperation().getOps<MGP::ModelOp>()) {
      auto mojoModuleRange =
          modelOp.getInitBlock()->getOps<MOGG::MojoModuleOp>();
      if (mojoModuleRange.empty())
        continue;

      /// TODO: This makes no sense as an op. Turn this op and
      /// LoadCompiledKernelOp into attributes of the MGP::ModelOp.
      ASSERT_STREAM(llvm::range_size(mojoModuleRange) == 1,
                    << "Expected at most one mojo module per model op");
      auto mojoModule = *mojoModuleRange.begin();
      auto mojoTextAttr =
          cast<DenseResourceElementsAttr>(mojoModule.getModule());
      auto mojoText =
          StringRef(mojoTextAttr.getData().data(), mojoTextAttr.size());

      llvm::errs() << llvm::formatv("##### Module {} #####\n{}\n", ++moduleIdx,
                                    mojoText);
    }
  }
};

} // namespace M::MGP
