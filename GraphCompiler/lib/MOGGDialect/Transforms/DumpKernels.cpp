//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"

using namespace M;
using namespace KGEN;
using namespace M::MOGG;

namespace M::MOGG {

#define GEN_PASS_DEF_DUMPKERNELS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct DumpKernelsPass : public impl::DumpKernelsBase<DumpKernelsPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    for (KernelDeclOp decl : moduleOp.getOps<KernelDeclOp>())
      llvm::errs() << KernelDeclAdaptor(decl) << "\n";
    for (PrimitiveDeclOp decl : moduleOp.getOps<PrimitiveDeclOp>())
      llvm::errs() << KernelFunctionDeclAdaptor(decl) << "\n";
  }
};

} // namespace

} // namespace M::MOGG
