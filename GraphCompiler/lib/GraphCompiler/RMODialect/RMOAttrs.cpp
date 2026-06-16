//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the RMO dialect attributes.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/RMODialect/RMOAttrs.h"
#include "GraphCompiler/GraphCompiler/RMODialect/RMODialect.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"

using namespace M;
using namespace RMO;

//===----------------------------------------------------------------------===//
// RMODialect attribute support
//===----------------------------------------------------------------------===//

void RMODialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "GraphCompiler/GraphCompiler/RMODialect/RMOAttrs.cpp.inc"
      >();
  addAttributes<PythonFrameAttr>();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GraphCompiler/GraphCompiler/RMODialect/RMOAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// Python frame location materialization
//===----------------------------------------------------------------------===//

void RMO::materializePythonFrameLocations(mlir::Operation *root) {
  mlir::MLIRContext *ctx = root->getContext();
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(
      [ctx](mlir::OpaqueLoc opaqueLoc) -> mlir::LocationAttr {
        mlir::Location fallback = opaqueLoc.getFallbackLocation();
        if (auto frame = PythonFrameAttr::fromOpaque(opaqueLoc)) {
          llvm::StringRef stackTrace = frame.stack_trace();
          if (!stackTrace.empty())
            return mlir::FusedLoc::get(ctx, {fallback},
                                       mlir::StringAttr::get(ctx, stackTrace));
        }
        return fallback;
      });
  root->walk([&replacer](mlir::Operation *op) {
    replacer.replaceElementsIn(op, /*replaceAttrs=*/false,
                               /*replaceLocs=*/true, /*replaceTypes=*/false);
  });
}
