//===- MLIRToBEF.cpp - MLIR to BEF file format converter ------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the main entrypoints for the MLIRToBEF library.
// The converter is implemented in three phases.  The first phase identifies all
// of the strings and attributes that need to be emitted to the string/attribute
// pool.  The second phase optimizes and emits the strings and attributes to
// the file and remembers their offsets.  The third phase emits all of the
// regions in the MLIR program.
//
// MLIR ops are converted to primitive info and stored in BEF.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/MLIRToBEF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Translation.h"
using namespace M;

LogicalResult M::translateMLIRToBEF(ModuleOp module, raw_ostream &output) {
  module.emitError("MLIRToBef not implemented yet");
  return failure();
}

void M::registerMLIRToBEFTranslation() {
  static mlir::TranslateFromMLIRRegistration toBEF(
      "export-bef", [](ModuleOp module, llvm::raw_ostream &os) {
        return translateMLIRToBEF(module, os);
      });
}
