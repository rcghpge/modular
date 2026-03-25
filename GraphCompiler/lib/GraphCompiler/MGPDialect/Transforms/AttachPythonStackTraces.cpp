//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GraphCompiler/GraphCompiler/RMODialect/RMOAttrs.h"
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

struct AttachPythonStackTraces
    : public impl::AttachPythonStackTracesBase<AttachPythonStackTraces> {
  using AttachPythonStackTracesBase::AttachPythonStackTracesBase;

  void runOnOperation() override {
    mlir::MLIRContext *ctx = &getContext();

    WalkResult result = getOperation()->walk([&ctx](Operation *op) {
      Location loc = op->getLoc();
      auto opaqueLoc = mlir::dyn_cast<mlir::OpaqueLoc>(loc);
      if (!opaqueLoc)
        return WalkResult::advance();

      Location fallback = opaqueLoc.getFallbackLocation();
      auto pythonFrame = RMO::PythonFrameAttr::fromOpaque(opaqueLoc);
      char *stackTrace = pythonFrame ? pythonFrame.stack_trace() : nullptr;

      if (stackTrace && *stackTrace) {
        // Embed the trace into the assert message for better diagnostics.
        if (auto assertOp = dyn_cast<AssertOp>(op)) {
          std::string fullMsg =
              "\nBuild Time Traceback (most recent call last):\n";
          llvm::raw_string_ostream ss(fullMsg);
          ss << stackTrace;
          StringAttr msg = mlir::cast<StringAttr>(assertOp.getMessageAttr());
          ss << "\nMAX Runtime Error: " << msg.str() << '\n';
          assertOp.setMessageAttr(StringAttr::get(ctx, fullMsg));
        }
        // Encode the trace as FusedLoc metadata before stripping OpaqueLoc.
        op->setLoc(mlir::FusedLoc::get(ctx, {fallback},
                                       StringAttr::get(ctx, stackTrace)));
      } else {
        op->setLoc(fallback);
      }

      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace
