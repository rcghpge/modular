//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

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

      // Note: This is currently tuned for how the python graph api generates
      // location information. It will likely need to be customized for use with
      // other frontends. If other frontends do not organize location info in
      // the same way, the traceback will end up being empty.

      // We are printing the location in a python style.
      // Python prints the stack in reverse order (most recent call last).
      llvm::SmallVector<mlir::NameLoc> locs;
      mlir::AttrTypeWalker walker;
      walker.addWalk([&](mlir::NameLoc name_loc) { locs.push_back(name_loc); });
      walker.walk(loc);
      if (locs.empty()) {
        // No location information or unexpected format.
        // Skip updating this assert op.
        return WalkResult::advance();
      }

      std::string full_msg = "\nBuild Time Traceback (most recent call last):";
      llvm::raw_string_ostream ss(full_msg);
      for (auto &name_loc : llvm::reverse(locs)) {
        std::string context = name_loc.getName().str();

        auto file_loc =
            mlir::dyn_cast<mlir::FileLineColLoc>(name_loc.getChildLoc());
        if (!file_loc)
          continue;

        std::string module_file = file_loc.getFilename().str();
        size_t i = module_file.find(":");

        if (i != std::string::npos) {
          std::string module = module_file.substr(0, i);
          std::string file = module_file.substr(i + 1, std::string::npos);
          ss << "\n  File " << file << ", line " << file_loc.getLine()
             << ", in " << module;
        } else {
          ss << "\n  File " << module_file << ", line " << file_loc.getLine();
        }

        ss << '\n';
        if (!context.empty()) {
          ss << context;

          for (size_t i = 0; i < file_loc.getColumn(); ++i)
            ss << ' ';
          ss << '^';
        }
      }

      StringAttr msg = mlir::cast<StringAttr>(op.getMessageAttr());
      ss << "\nMAX Runtime Error: " << msg.str() << '\n';

      op.setMessageAttr(StringAttr::get(ctx, full_msg));

      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace
