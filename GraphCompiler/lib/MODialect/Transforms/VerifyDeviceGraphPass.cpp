//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"

#define DEBUG_TYPE "mo-verify-device-graph"

namespace M::MO {

#define GEN_PASS_DEF_VERIFYDEVICEGRAPHPASS
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

using namespace M;
using namespace MO;

namespace {

//===----------------------------------------------------------------------===//
// VerifyDeviceGraphPass
//===----------------------------------------------------------------------===//
//
// When a `mo.graph` carries the `isDeviceGraph` attribute, every
// kernel-backed op in its body must correspond to a `mogg.kernel.decl` whose
// `build_graph` field is populated. Ops without a kernel registration (such
// as dialect-intrinsic ops like `mo.constant` or `mo.output`) are skipped.
// Graphs without the `isDeviceGraph` attribute are no-ops.

struct VerifyDeviceGraphPass final
    : public impl::VerifyDeviceGraphPassBase<VerifyDeviceGraphPass> {
  using Base::Base;

  void runOnOperation() override {
    GraphOp graphOp = getOperation();
    if (!graphOp.getIsDeviceGraph())
      return;

    auto moduleOp = graphOp->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      graphOp.emitError()
          << "device-graphable mo.graph has no parent ModuleOp; cannot "
             "look up kernel registrations";
      return signalPassFailure();
    }

    MOGG::MojoRegistry registry(moduleOp);

    graphOp.getBodyBlock()->walk(
        [&](mlir::Operation *op) {
          // Terminators do not require validation.
          if (op->hasTrait<OpTrait::IsTerminator>())
            return;

          // Control flow is not yet supported by the device-graph pipeline.
          // Terminators (e.g. `mo.output`, `mo.yield`) are tagged as control
          // ops but pass through trivially, so skip them. This restriction
          // will be relaxed once CUDA conditional nodes (or an equivalent)
          // are wired up.
          if (isa<MOControlOpInterface>(op)) {
            op->emitError()
                << "device-graphable mo.graph cannot contain control-flow op '"
                << op->getName().getStringRef()
                << "' (control flow is not yet supported)";
            signalPassFailure();
            return;
          }

          // For `mo.custom`, the kernel registration is named by the
          // `symbol` attribute; for built-in MO ops, the op name itself is
          // the registration string.

          StringRef registration =
              TypeSwitch<Operation *, StringRef>(op)
                  .Case([](CustomOp custom) { return custom.getSymbol(); })
                  .Default(op->getName().getStringRef());

          if (!registry.containsKernel(registration))
            return;

          DeviceRefAttr device = MOGG::MojoRegistry::getDeviceRefForOp(op);
          auto kernelOr = registry.getResolvedKernel(registration, device);
          if (kernelOr.isError()) {
            op->emitError(kernelOr.getError());
            signalPassFailure();
            return;
          }
          MOGG::KernelDeclOp kernel = *kernelOr;
          if (!kernel.getBuildGraph()) {
            op->emitError()
                << "op '" << op->getName().getStringRef()
                << "' is in a device-graphable mo.graph but its kernel "
                   "registration ('"
                << registration << "', kernel symbol '" << kernel.getSymName()
                << "') does not expose a `build_graph` entry point";
            signalPassFailure();
          }
        });
  }
};

} // namespace
