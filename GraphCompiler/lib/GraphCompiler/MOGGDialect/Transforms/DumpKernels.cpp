//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "KGEN/MOGGPreElab/MojoKernelAdaptor.h"

using namespace M;
using namespace KGEN;
using namespace MOGGPreElab;

namespace M::MOGG {

#define GEN_PASS_DEF_DUMPKERNELS
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct KernelFuncs {
  Operation *execute;
  Operation *shape;
  Operation *elementwise;
};

struct DumpKernelsPass : public impl::DumpKernelsBase<DumpKernelsPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    llvm::DenseMap<StringAttr, KernelFuncs> kernels;
    moduleOp.walk([&](Operation *op) {
      auto structName = op->getAttrOfType<StringAttr>(kMOGGKernelStructName);
      if (!structName)
        return;

      if (op->hasAttr(kMOGGExecuteFunctionLabel))
        kernels[structName].execute = op;
      else if (op->hasAttr(kMOGGShapeFunctionLabel))
        kernels[structName].shape = op;
      else if (op->hasAttr(kMOGGElementwiseFunctionLabel))
        kernels[structName].elementwise = op;
    });

    for (auto &[structName, entry] : kernels) {
      llvm::errs() << MojoKernelAdaptor(
          structName.strref(), dyn_cast_or_null<LIT::FnOp>(entry.execute),
          dyn_cast_or_null<LIT::FnOp>(entry.shape),
          dyn_cast_or_null<LIT::FnOp>(entry.elementwise));
      llvm::errs() << "\n";
    }
  }
};

} // namespace

} // namespace M::MOGG
