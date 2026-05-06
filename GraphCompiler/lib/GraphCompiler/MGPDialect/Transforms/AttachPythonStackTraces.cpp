//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GraphCompiler/GraphCompiler/RMODialect/RMOAttrs.h"
#include "Support/Diagnostics/FormatScopedDiagnosticHandler.h"
#include "mlir/IR/Visitors.h"

#define DEBUG_TYPE "attach-python-stack-traces"

namespace M::MGP {

#define GEN_PASS_DEF_ATTACHPYTHONSTACKTRACES
#include "GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

}; // namespace M::MGP

using namespace M;
using namespace MGP;

namespace {

//===----------------------------------------------------------------------===//
// AttachPythonStackTraces
//===----------------------------------------------------------------------===//
//
// For each op in the module that has an OpaqueLoc wrapping a PythonFrameAttr:
//  - Embeds the Python stack trace into the message of any mgp.assert op.
//  - Encodes the trace as FusedLoc metadata so it survives downstream
//    lowering passes (e.g. KernelStubOp::emitMojoMOGGABI can retrieve it
//    via collectSourceNotes). See GEX-2678.
//  - Strips the OpaqueLoc (replacing it with the fallback location), since
//    OpaqueLoc does not serialize correctly to bytecode. See GEX-1435.
//  - If OpaqueLocs are nested inside a FusedLoc, each is individually
//    replaced with its fallback, preserving the location tree structure.

struct AttachPythonStackTraces
    : public impl::AttachPythonStackTracesBase<AttachPythonStackTraces> {
  using AttachPythonStackTracesBase::AttachPythonStackTracesBase;

  /// Recursively strip OpaqueLocs from a location tree, converting each to
  // its fallback location or a FusedLoc carrying the Python stack trace.
  static std::optional<Location> stripOpaqueLocs(mlir::MLIRContext *ctx,
                                                 Location loc) {
    if (auto opaqueLoc = mlir::dyn_cast<mlir::OpaqueLoc>(loc)) {
      Location fallback = opaqueLoc.getFallbackLocation();
      auto pythonFrame = RMO::PythonFrameAttr::fromOpaque(opaqueLoc);
      char *stackTrace = pythonFrame ? pythonFrame.stack_trace() : nullptr;
      if (stackTrace && *stackTrace)
        return mlir::FusedLoc::get(ctx, {fallback},
                                   StringAttr::get(ctx, stackTrace));
      return fallback;
    }

    if (auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
      SmallVector<Location> children;
      bool changed = false;
      for (Location child : fusedLoc.getLocations()) {
        if (auto newChild = stripOpaqueLocs(ctx, child)) {
          children.push_back(*newChild);
          changed = true;
        } else {
          children.push_back(child);
        }
      }
      if (changed)
        return mlir::FusedLoc::get(ctx, children, fusedLoc.getMetadata());
      return std::nullopt;
    }

    return std::nullopt;
  }

  void runOnOperation() override {
    mlir::MLIRContext *ctx = &getContext();

    getOperation()->walk([&ctx](Operation *op) {
      Location loc = op->getLoc();

      // Embed the trace into the assert message for better diagnostics.
      // Only inspect the op's immediate OpaqueLoc; nested OpaqueLocs inside a
      // FusedLoc don't apply here.
      if (auto assertOp = dyn_cast<AssertOp>(op)) {
        if (auto opaqueLoc = mlir::dyn_cast<mlir::OpaqueLoc>(loc)) {
          auto pythonFrame = RMO::PythonFrameAttr::fromOpaque(opaqueLoc);
          char *stackTrace = pythonFrame ? pythonFrame.stack_trace() : nullptr;
          if (stackTrace && *stackTrace) {
            std::string fullMsg = "\nSource Traceback:\n";
            llvm::raw_string_ostream ss(fullMsg);
            ss << stackTrace << '\n';
            StringAttr msg = mlir::cast<StringAttr>(assertOp.getMessageAttr());
            mlir::Diagnostic diag(loc, mlir::DiagnosticSeverity::Error);
            diag << msg.str();
            FormatScopedDiagnosticHandler::emitDiagnosticSeverityToStream(ss,
                                                                          diag);
            assertOp.setMessageAttr(StringAttr::get(ctx, fullMsg));
          }
        }
      }

      if (auto newLoc = stripOpaqueLocs(ctx, loc))
        op->setLoc(*newLoc);
    });
  }
};

} // namespace
