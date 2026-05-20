//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h"

using namespace M;
using namespace KGEN;
using namespace M::MOGG;

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

    // If mogg.kernel.decl ops are present, use KernelDeclAdaptor.
    if (!moduleOp.getOps<KernelDeclOp>().empty() ||
        !moduleOp.getOps<PrimitiveDeclOp>().empty()) {
      for (KernelDeclOp decl : moduleOp.getOps<KernelDeclOp>()) {
        llvm::errs() << KernelDeclAdaptor(decl) << "\n";
      }
      for (PrimitiveDeclOp decl : moduleOp.getOps<PrimitiveDeclOp>()) {
        llvm::errs() << KernelFunctionDeclAdaptor(decl) << "\n";
      }
      return;
    }

    // Fallback: use MojoKernelAdaptor on annotated FnOps.
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
