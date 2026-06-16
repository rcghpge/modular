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

static llvm::StringRef extractStackTrace(mlir::OpaqueLoc opaqueLoc) {
  if (auto pythonFrame = RMO::PythonFrameAttr::fromOpaque(opaqueLoc))
    return pythonFrame.stack_trace();
  return {};
}

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

  void runOnOperation() override {
    mlir::MLIRContext *ctx = &getContext();

    getOperation()->walk([&ctx](Operation *op) {
      Location loc = op->getLoc();

      // Embed the trace into the assert message for better diagnostics.
      // Only inspect the op's immediate OpaqueLoc; nested OpaqueLocs inside a
      // FusedLoc don't apply here.
      if (auto assertOp = dyn_cast<AssertOp>(op)) {
        if (auto opaqueLoc = mlir::dyn_cast<mlir::OpaqueLoc>(loc)) {
          llvm::StringRef stackTrace = extractStackTrace(opaqueLoc);
          if (!stackTrace.empty()) {
            std::string fullMsg = "\nSource Traceback:\n";
            llvm::raw_string_ostream ss(fullMsg);
            ss << stackTrace << '\n';
            StringAttr msg = assertOp.getMessageAttr();
            mlir::Diagnostic diag(loc, mlir::DiagnosticSeverity::Error);
            diag << msg.str();
            FormatScopedDiagnosticHandler::emitDiagnosticSeverityToStream(ss,
                                                                          diag);
            assertOp.setMessageAttr(StringAttr::get(ctx, fullMsg));
          }
        }
      }
    });

    // Convert OpaqueLocs wrapping PythonFrameAttr into serializable FusedLocs
    // so the traces survive downstream lowering and bytecode.
    RMO::materializePythonFrameLocations(getOperation());
  }
};

} // namespace
