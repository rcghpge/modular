//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MOGGPreElab/MojoKernelAdaptor.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;
using namespace MOGGPreElab;

namespace M::KGEN::MOGGPreElab {
#define GEN_PASS_DEF_DUMPKERNELS
#include "KGEN/MOGGPreElab/MOGGPreElabPasses.h.inc"
} // namespace M::KGEN::MOGGPreElab

class DumpKernelsPass
    : public M::KGEN::MOGGPreElab::impl::DumpKernelsBase<DumpKernelsPass> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    llvm::DenseMap<StringAttr, std::pair<Operation *, Operation *>> kernels;
    moduleOp.walk([&](Operation *op) {
      auto structName = op->getAttrOfType<StringAttr>(kMOGGKernelStructName);
      if (!structName)
        return;

      if (op->hasAttr(kMOGGExecuteFunctionLabel))
        kernels[structName].first = op;
      else if (op->hasAttr(kMOGGShapeFunctionLabel))
        kernels[structName].second = op;
    });

    for (auto &[structName, entry] : kernels) {
      auto &[execute, shape] = entry;
      if (isa<LIT::FnOp>(execute)) {
        llvm::errs() << MojoKernelAdaptor<LIT::FnOp>(
            structName.strref(), dyn_cast_or_null<LIT::FnOp>(execute),
            dyn_cast_or_null<LIT::FnOp>(shape), nullptr);
      } else {
        llvm::errs() << MojoKernelAdaptor<KGEN::GeneratorOp>(
            structName.strref(), dyn_cast_or_null<KGEN::GeneratorOp>(execute),
            dyn_cast_or_null<KGEN::GeneratorOp>(shape), nullptr);
      }
      llvm::errs() << "\n";
    }
  }
};
