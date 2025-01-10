//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/RMODialect/RMOAttrs.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPTransforms/MGPTransforms.h"
#include "mlir/IR/Visitors.h"

#define DEBUG_TYPE "add-stack-traces-to-asserts"

namespace M::MGP {

#define GEN_PASS_DEF_ADDSTACKTRACESTOASSERTSPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

}; // namespace M::MGP

using namespace M;
using namespace MGP;

namespace {

//===----------------------------------------------------------------------===//
// AddStackTracesToAssertsPass
//===----------------------------------------------------------------------===//
//
// Adds stack traces to all assert messages in the model.

struct AddStackTracesToAssertsPass
    : public impl::AddStackTracesToAssertsPassBase<
          AddStackTracesToAssertsPass> {
  using AddStackTracesToAssertsPassBase::AddStackTracesToAssertsPassBase;

  void runOnOperation() override {
    mlir::MLIRContext *ctx = &getContext();

    WalkResult result = getOperation()->walk([&ctx](AssertOp op) {
      Location loc = op.getLoc();
      auto opaque_loc = mlir::dyn_cast<mlir::OpaqueLoc>(loc);
      if (!opaque_loc)
        return WalkResult::advance();

      // Grab the python frame from location.
      void *attr_ptr = mlir::OpaqueLoc::getUnderlyingLocationOrNull<
          RMO::PythonFrameAttr::ImplType *>(opaque_loc);
      if (!attr_ptr)
        return WalkResult::advance();

      auto attr = RMO::PythonFrameAttr::getFromOpaquePointer(attr_ptr);

      std::string full_msg =
          "\nBuild Time Traceback (most recent call last):\n";
      llvm::raw_string_ostream ss(full_msg);

      ss << attr.stack_trace();

      StringAttr msg = mlir::cast<StringAttr>(op.getMessageAttr());
      ss << "\nMAX Runtime Error: " << msg.str() << '\n';

      op.setMessageAttr(StringAttr::get(ctx, full_msg));

      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();

    // An OpaqueLoc should serialize to its fallback location when emitting
    // bytecode, but it seems that does not work correctly. This might be an
    // issue in upstream MLIR, we're not sure. For the time being, manually
    // replace the OpaqueLoc with its fallback to avoid this issue. See GEX-1435
    // for more details.
    result = getOperation()->walk([](Operation *op) {
      Location loc = op->getLoc();
      auto opaque_loc = mlir::dyn_cast<mlir::OpaqueLoc>(loc);
      if (!opaque_loc)
        return WalkResult::advance();

      op->setLoc(opaque_loc.getFallbackLocation());
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace
