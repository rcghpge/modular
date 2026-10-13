//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "max/mcl/MOF/API/c/detail/mojo_analysis.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "Support/ErrorOr.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"

using namespace M;

M::ErrorOr<mlir::OwningOpRef<mlir::ModuleOp>>
MAXG_createKernelDeclsModule(MlirContext context) {
  mlir::MLIRContext *ctx = unwrap(context);
  mlir::OpBuilder builder(ctx);
  // Use an UnknownLoc; the module is a kernel-decls container, not associated
  // with any user source location.
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::ModuleOp::create(builder.getUnknownLoc()));
  return std::move(module);
}

M::ErrorOrSuccess MAXG_addKernelPackage(MlirModule module,
                                        std::filesystem::path path) {
  mlir::ModuleOp mod = unwrap(module);
  mlir::MLIRContext *ctx = mod.getContext();

  // First call to a fresh module loads built-ins; subsequent calls append the
  // delta. We detect the "first call" case as "module has no kernel decls
  // yet" so the caller doesn't need to track this themselves. Passing an
  // empty path is valid on the first call: it loads built-ins only.
  bool isFirstCall = mod.getOps<MOGG::KernelDeclOp>().empty() &&
                     mod.getOps<MOGG::PrimitiveDeclOp>().empty();

  MOGG::ImportPackagesOptions opts;
  if (!path.empty())
    opts.mojoPkgPaths.push_back(path.string());
  opts.append = !isFirstCall;

  mlir::PassManager pm(ctx);
  pm.addPass(MOGG::createImportPackages(opts));
  if (failed(pm.run(mod)))
    return Error("MAXG_addKernelPackage: failed to import kernels from '" +
                 path.string() + "'");
  return success();
}

M::ErrorOrSuccess MAXG_verifyCustomOp(MlirModule module, MlirOperation op) {
  return MOGG::verifyCustomOp(unwrap(module), unwrap(op));
}

void MAXG_seedKernelDecls(MlirModule target, MlirModule source) {
  MOGG::copyKernelDecls(unwrap(source), unwrap(target));
}
